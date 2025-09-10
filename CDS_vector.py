import json
import re
from typing import List, Dict, Any

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


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
    """Extracts the 16 core numeric features for the MLP."""
    sat25 = to_float(uni.get(K["sat25"], 0.0))
    sat50 = to_float(uni.get(K["sat50"], 0.0))
    sat75 = to_float(uni.get(K["sat75"], 0.0))

    act25 = to_float(uni.get(K["act25"], 0.0))
    act50 = to_float(uni.get(K["act50"], 0.0))
    act75 = to_float(uni.get(K["act75"], 0.0))

    fee = to_float(uni.get(K["fee"], 0.0))
    ratio = parse_ratio(uni.get(K["ratio"], None))

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
    print(f"✅ Saved embeddings to {output_path}")


# -------------------- Run --------------------

if __name__ == "__main__":
    # Input: a single JSON file containing an array of university objects [ {...}, {...}, ... ]
    INPUT_JSON_FILE = "CDS1.json"
    OUTPUT_JSONL_FILE = "CDS_vectors.jsonl"
    
    process_universities(INPUT_JSON_FILE, OUTPUT_JSONL_FILE, batch_size=8)