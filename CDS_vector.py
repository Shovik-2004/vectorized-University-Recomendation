import json
import re
from typing import List, Dict, Any

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm



# Add this helper function to both python scripts

def convert_sat_to_act(sat_score: float) -> float:
    """Converts an SAT score to an equivalent ACT score using a simplified concordance table."""
    if sat_score == 0: return 0.0
    # This is a simplified table. For a production system, use a more granular one.
    concordance = {
        1600: 36, 1560: 35, 1520: 34, 1490: 33, 1450: 32, 1420: 31, 1390: 30,
        1360: 29, 1330: 28, 1300: 27, 1260: 26, 1230: 25, 1200: 24, 1160: 23,
        1130: 22, 1100: 21, 1060: 20, 1030: 19, 990: 18, 960: 17, 920: 16, 900: 15,
        880: 14, 850: 13, 820: 12, 800: 11, 780: 10, 750: 9, 720: 8, 700: 7, 650: 6, 600: 5, 550: 4, 500: 3, 400: 2, 300: 1, 200: 0
    }
    # Find the closest SAT score in the table and return its corresponding ACT score
    closest_sat = min(concordance.keys(), key=lambda k: abs(k - sat_score))
    return float(concordance[closest_sat])


# You can create a reverse function as well if needed
def convert_act_to_sat(act_score: float) -> float:
    """Converts an ACT score to an equivalent SAT score."""
    if act_score == 0: return 0.0
    # Simplified reverse table
    concordance = {
        36: 1600, 35: 1560, 34: 1520, 33: 1490, 32: 1450, 31: 1420, 30: 1390,
        29: 1360, 28: 1330, 27: 1300, 26: 1260, 25: 1230, 24: 1200, 23: 1160,
        22: 1130, 21: 1100, 20: 1060, 19: 1030, 18: 990, 17: 960, 16: 920, 15: 900,
        14: 880, 13: 850, 12: 820, 11: 800, 10: 780, 9: 750, 8: 720, 7: 700, 6: 650, 5: 600, 4: 550, 3: 500, 2: 400, 1: 300, 0: 200
    }
    closest_act = min(concordance.keys(), key=lambda k: abs(k - act_score))
    return float(concordance[closest_act])

# -------------------- Helpers --------------------

def to_float(val, default=0.0) -> float:
    """Parse numbers that may come as strings with %, $, commas, or be null/'Not provided'."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in {"na", "n/a", "null", "none", "not provided", "not provided in document"}:
        return default
    s = s.replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return default


_ratio_pattern = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:to|:)\s*(\d+(?:\.\d+)?)\s*$", re.I)

def parse_ratio(val) -> float:
    """
    Parse 'Student to faculty ratio' like '8 to 1' or '8:1' -> 8/1 = 8.0
    Returns 0.0 if not parseable.
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    m = _ratio_pattern.match(str(val))
    if not m:
        return 0.0
    a, b = float(m.group(1)), float(m.group(2))
    if b == 0:
        return 0.0
    return a / b


def list_to_text(label: str, values: Any) -> str:
    """Turn a list field into a labeled text snippet; safely handles None."""
    if not values:
        return ""
    if isinstance(values, list):
        items = ", ".join([str(v) for v in values if v is not None])
    else:
        items = str(values)
    return f"{label}: {items}."


def dict_to_text(label: str, data: Dict[str, Any]) -> str:
    """Turns a dictionary (like admissions factors) into a text snippet."""
    if not data or not isinstance(data, dict):
        return ""
    parts = [f"{key.replace('_', ' ')} is {value}" for key, value in data.items()]
    return f"{label}: {'; '.join(parts)}."


# -------------------- Model --------------------

class UniversityEmbedder(nn.Module):
    """
    Balanced fusion embedder:
      - Text -> BERT (768) -> Linear -> proj_dim
      - Numbers -> MLP -> proj_dim
      - Fused = 0.3 * text_proj + 0.7 * num_proj
    """
    def __init__(self, bert_model_name: str = "bert-base-uncased",
                 num_input_dim: int = 16,  # SAT(3)+ACT(3)+fee(1)+ratio(1)+GPA bins(8) = 16
                 proj_dim: int = 128, device: str = None):
        super().__init__()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_output_dim = self.bert.config.hidden_size  # 768

        # Numeric MLP
        self.num_mlp = nn.Sequential(
            nn.Linear(num_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, proj_dim)
        )

        # Project text to same size
        self.text_proj = nn.Linear(bert_output_dim, proj_dim)

        self.to(self.device)

    def normalize(self, feats: torch.Tensor) -> torch.Tensor:
       """
       Normalization with more spread to avoid collapse:
       SAT / 800  -> range ~0-2
       ACT / 18   -> range ~0-2
       Application fee / 200
       Student-faculty ratio / 50
       GPA bins / 25  -> exaggerates differences
       """
       sat = feats[:, 0:3] / 800.0
       act = feats[:, 3:6] / 18.0
       fee = torch.clamp(feats[:, 6:7] / 200.0, 0.0, 1.0)
       ratio = torch.clamp(feats[:, 7:8] / 50.0, 0.0, 1.0)
       gpa_bins = feats[:, 8:16] / 25.0  # 0-100 -> 0-4
       return torch.cat([sat, act, fee, ratio, gpa_bins], dim=1)


    def forward(self, text_fields: List[str], num_features: torch.Tensor) -> torch.Tensor:
        # TEXT
        encoded = self.tokenizer(
            text_fields,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.bert(**encoded)
        text_embeds = outputs.pooler_output  # [B, 768]
        text_proj = self.text_proj(text_embeds)  # [B, D]

        # NUMERIC
        num_features = num_features.to(self.device)
        num_norm = self.normalize(num_features)
        num_proj = self.num_mlp(num_norm)  # [B, D]

        # BALANCED FUSION
        fused = 0.3 * text_proj + 0.7 * num_proj
        return fused


# -------------------- Feature Builders --------------------

# K dictionary maps friendly names to the actual keys in the JSON file.
K = {
    "name": "Name of University/College",
    "city": "City",
    "address": "Address",
    "phone": "Admissions Phone Number",
    "toll_free": "Admissions Toll-Free Phone Number",
    "email": "Admissions E-mail Address",
    "url": "University home page URL",
    "type": "Coeducational, men's, or women's college", # Updated key
    "calendar": "academic year calender", # New key
    "degrees_offered": "degrees offered by your instituition", # New key

    "sat25": "SAT 25th percentile",
    "sat50": "SAT 50th percentile (median)",
    "sat75": "SAT 75th percentile",

    "act25": "ACT 25th percentile",
    "act50": "ACT 50th percentile (median)",
    "act75": "ACT 75th percentile",

    "fee": "Application fee",
    "ratio": "Student to faculty ratio",

    "gpa_375_up": "GPA 3.75 and higher",
    "gpa_350_374": "GPA 3.50-3.74",
    "gpa_325_349": "GPA 3.25-3.49",
    "gpa_300_324": "GPA 3.00-3.24",
    "gpa_250_299": "GPA 2.50-2.99",
    "gpa_200_249": "GPA 2.00-2.49",
    "gpa_100_199": "GPA 1.00-1.99",
    "gpa_below_10": "GPA Below 1.0",

    "avg_gpa": "average gpa", # New key
    "admissions_factors": "admissions_factors", # New key
    "financial_aid_url": "financial_aid", # New key
    "study": "Special study options",
    "activities": "Activities offered",
    "housing": "Housing information",
}

def build_text(uni: Dict[str, Any]) -> str:
    """Combines all textual and descriptive data into a single string for BERT."""
    parts = [
        uni.get(K["name"], ""),
        uni.get(K["city"], ""),
        uni.get(K["address"], ""),
        # --- FIX IS HERE ---
        # This now correctly handles cases where the college type is a list instead of a string.
        list_to_text("College type", uni.get(K["type"])),
        # -------------------
        uni.get(K["calendar"], ""),
        str(uni.get(K["email"], "")),
        str(uni.get(K["url"], "")),
        str(uni.get(K["financial_aid_url"], "")),
        f"Admissions Phone: {uni.get(K['phone'], 'N/A')}",
        f"Toll-Free: {uni.get(K['toll_free'], 'N/A')}",
        f"Application fee: {uni.get(K['fee'], 'N/A')}",
        f"Student to faculty ratio: {uni.get(K['ratio'], 'N/A')}",
        f"Average GPA: {uni.get(K['avg_gpa'], 'N/A')}",
        list_to_text("Degrees offered", uni.get(K["degrees_offered"], [])),
        dict_to_text("Admissions factors", uni.get(K["admissions_factors"], {})),
        list_to_text("Special study options", uni.get(K["study"], [])),
        list_to_text("Activities offered", uni.get(K["activities"], [])),
        list_to_text("Housing information", uni.get(K["housing"], [])),
    ]
    return " ".join([p for p in parts if p])


def build_numeric_features(uni: Dict[str, Any]) -> List[float]:
    """
    Extracts the 16 core numeric features for the MLP, with logic to handle
    missing SAT/ACT and GPA data through conversion and imputation.
    """
    # 1. Extract SAT and ACT scores
    sat25 = to_float(uni.get(K["sat25"], 0.0))
    sat50 = to_float(uni.get(K["sat50"], 0.0))
    sat75 = to_float(uni.get(K["sat75"], 0.0))

    act25 = to_float(uni.get(K["act25"], 0.0))
    act50 = to_float(uni.get(K["act50"], 0.0))
    act75 = to_float(uni.get(K["act75"], 0.0))

    # 2. Apply concordance logic to fill missing scores
    # If SAT is missing but ACT is present, estimate SAT from ACT
    if sat50 == 0.0 and act50 > 0.0:
        sat50 = convert_act_to_sat(act50)
        # Estimate the 25th and 75th percentiles based on the new median
        sat25 = sat50 - 40 
        sat75 = sat50 + 40

    # If ACT is missing but SAT is present, estimate ACT from SAT
    if act50 == 0.0 and sat50 > 0.0:
        act50 = convert_sat_to_act(sat50)
        # Estimate the 25th and 75th percentiles
        act25 = max(act50 - 1, 0) # Ensure it doesn't go below 0
        act75 = min(act50 + 1, 36) # Ensure it doesn't go above 36

    # 3. Extract fee and ratio
    fee = to_float(uni.get(K["fee"], 0.0))
    ratio = parse_ratio(uni.get(K["ratio"], None))

    # 4. Extract GPA bins
    gpa_bins = [
        to_float(uni.get(K["gpa_375_up"], 0.0)),
        to_float(uni.get(K["gpa_350_374"], 0.0)),
        to_float(uni.get(K["gpa_325_349"], 0.0)),
        to_float(uni.get(K["gpa_300_324"], 0.0)),
        to_float(uni.get(K["gpa_250_299"], 0.0)),
        to_float(uni.get(K["gpa_200_249"], 0.0)),
        to_float(uni.get(K["gpa_100_199"], 0.0)),
        to_float(uni.get(K["gpa_below_10"], 0.0)),
    ]

    # 5. Impute GPA data for top-tier schools if it's missing
    if sum(gpa_bins) == 0.0 and sat50 > 1450:
        # If all GPA bins are zero but it's a top SAT school,
        # impute a high value for the top bin.
        gpa_bins[0] = 90.0  # Assumes 90% of students have a GPA of 3.75+

    # 6. Combine all features into the final list
    features = [sat25, sat50, sat75, act25, act50, act75, fee, ratio] + gpa_bins
    assert len(features) == 16, f"Expected 16 features, but got {len(features)}"
    return features


# -------------------- Processing --------------------

def process_universities(input_path: str, output_path: str, batch_size: int = 1):
    """
    Reads a JSON array [ {...}, {...}, ... ] from input_path,
    writes JSONL with name + embedding to output_path.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = UniversityEmbedder(device=device)
    model.eval()

    writer = open(output_path, "w", encoding="utf-8")

    texts_batch = []
    nums_batch = []
    names_batch = []

    def flush_batch():
        if not texts_batch:
            return
        with torch.no_grad():
            fused = model(texts_batch, torch.tensor(nums_batch, dtype=torch.float))
            fused = fused.cpu().numpy()
        for name, emb in zip(names_batch, fused):
            rec = {"university_name": name, "embedding": emb.tolist()}
            writer.write(json.dumps(rec) + "\n")
        texts_batch.clear()
        nums_batch.clear()
        names_batch.clear()

    with open(input_path, "r", encoding="utf-8") as f:
        universities = json.load(f)

    for uni in tqdm(universities, desc="Vectorizing universities"):
        # Handle various possible keys for the university name
        name = (
            uni.get("Name of University/College")
            or uni.get("name")
            or uni.get("name_of_university")
            or uni.get("Name of College/University")
            or f"Unknown University {len(names_batch) + 1}"
        )

        text = build_text(uni)
        nums = build_numeric_features(uni)

        texts_batch.append(text)
        nums_batch.append(nums)
        names_batch.append(name)

        if len(texts_batch) >= batch_size:
            flush_batch()

    flush_batch() # Process any remaining items in the last batch
    writer.close()
    torch.save(model.state_dict(), "university_embedder.pth")
    print(f"✅ Saved embeddings to {output_path}")
    print(f"✅ Saved model weights to university_embedder.pth")


# -------------------- Run --------------------

if __name__ == "__main__":
    # Input: a single JSON file containing an array of university objects [ {...}, {...}, ... ]
    INPUT_JSON_FILE = "CDS1.json"
    OUTPUT_JSONL_FILE = "CDS_vectors.jsonl"
    
    process_universities(INPUT_JSON_FILE, OUTPUT_JSONL_FILE, batch_size=8)