import json
import numpy as np
import torch
import torch.nn as nn
import faiss
from transformers import BertTokenizer, BertModel
from typing import List
import textwrap


# -------------------- Embedder Class --------------------
class UniversityEmbedder(nn.Module):
    """
    Balanced fusion embedder:
      - Text -> BERT (768) -> Linear -> proj_dim
      - Numbers -> MLP -> proj_dim
      - Fused = 0.3 * text_proj + 0.7 * num_proj
    """
    def __init__(self, bert_model_name: str = "bert-base-uncased",
                 num_input_dim: int = 16,
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


# -------------------- Load Mapping --------------------
def load_mapping(mapping_path: str):
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- GPA → bins --------------------
def gpa_to_bins(gpa: float):
    bins = [0.0] * 8
    if gpa >= 3.75: bins[0] = 100.0
    elif 3.50 <= gpa < 3.75: bins[1] = 100.0
    elif 3.25 <= gpa < 3.50: bins[2] = 100.0
    elif 3.00 <= gpa < 3.25: bins[3] = 100.0
    elif 2.50 <= gpa < 3.00: bins[4] = 100.0
    elif 2.00 <= gpa < 2.50: bins[5] = 100.0
    elif 1.00 <= gpa < 2.00: bins[6] = 100.0
    else: bins[7] = 100.0
    return bins


# -------------------- Vectorize User Query (with debug) --------------------
def vectorize_user_query(model, gpa: float, sat: float = None, act: float = None, text: str = "", debug=False):
    sat_vec = [sat, sat, sat] if sat is not None else [0.0, 0.0, 0.0]
    act_vec = [act, act, act] if act is not None else [0.0, 0.0, 0.0]
    fee = [0.0]
    ratio = [0.0]
    gpa_bins = gpa_to_bins(gpa)

    features = sat_vec + act_vec + fee + ratio + gpa_bins
    features = torch.tensor([features], dtype=torch.float)

    text_fields = [text] if text else [""]

    with torch.no_grad():
        user_vec = model(text_fields, features).cpu().numpy()

    # 🔑 Normalize query for cosine similarity
    faiss.normalize_L2(user_vec)

    if debug:
        print("\n🟢 DEBUG: Raw numeric features:", features.tolist())
        print("🟢 DEBUG: Normalized numeric features:", model.normalize(features).tolist())
        print("🟢 DEBUG: Final fused + normalized user vector (first 10 dims):", user_vec[0][:10])

    return user_vec

# -------------------- Search (with debug) --------------------
def search_user_query(model, index, mapping, gpa, sat=None, act=None, text="", top_k=30, debug=False):
    query_vec = vectorize_user_query(model, gpa, sat, act, text, debug=debug)
    sims, I = index.search(query_vec, top_k)  # similarity scores (higher = better)

    all_results = []
    for j, i in enumerate(I[0]):
        # CORRECTED LINE: Access mapping as a list using the index 'i'
        if i < len(mapping):
            uni_name = mapping[i] 
            sim = float(sims[0][j])
            all_results.append((uni_name, sim))

            if debug and j < 5: # Limit debug printing for clarity
                print(f"🟢 DEBUG (Top 5): {uni_name} -> Cosine similarity {sim:.4f}")
        else:
            print(f"⚠️ Warning: Index {i} out of bounds for mapping list.")
            continue


    # Categorize results into three lists based on proximity
    ambitious = all_results[0:10]
    practical = all_results[10:20]
    safe = all_results[20:30]

    return ambitious, practical, safe

# -------------------- Helper functions to display details --------------------
def get_university_name(uni_data):
    """Robustly gets the university name from a dictionary entry."""
    keys_to_check = [
        "name_of_university",
        "Name of University/College",
        "Name of College/University",
        "name"
    ]
    for key in keys_to_check:
        if key in uni_data:
            return uni_data[key]
    return None

def display_university_details(details):
    """Formats and prints the details of a selected university."""
    print("\n" + "="*60)
    name = get_university_name(details)
    if name:
        print(f"🎓 Details for {name}")
        print("-" * 60)

    for key, value in details.items():
        # Skip redundant name keys
        if key.lower().startswith('name'):
            continue
            
        formatted_key = key.replace('_', ' ').replace('gpa', 'GPA').title()
        
        if isinstance(value, dict):
            print(f"\n--- {formatted_key} ---")
            for sub_key, sub_value in value.items():
                formatted_sub_key = sub_key.replace('_', ' ').title()
                print(f"  {formatted_sub_key}: {sub_value}")
        elif isinstance(value, list) and value:
            print(f"\n--- {formatted_key} ---")
            # Use textwrap to neatly format long lists
            print(textwrap.fill(", ".join(map(str, value)), width=80, initial_indent="  ", subsequent_indent="  "))
        elif value is not None and value != "":
            print(f"{formatted_key}: {value}")
    
    print("="*60 + "\n")


# -------------------- Main --------------------
if __name__ == "__main__":
    # Load FAISS index + mapping
    index = faiss.read_index("CDS_vectors.index")
    mapping = load_mapping("CDS_vectors_mapping.json")

    # Load model
    model = UniversityEmbedder()
    model.eval()

    # --- [NEW] Load the full JSON data to display details later ---
    with open('CDS1.json', 'r', encoding='utf-8') as f:
        full_university_data = json.load(f)

    # Create a dictionary for quick lookups, using a robust name getter
    university_details_map = {}
    for uni_data in full_university_data:
        name = get_university_name(uni_data)
        if name:
            university_details_map[name.lower()] = uni_data


    print("🎓 University Recommendation System (Cosine Similarity)")
    print("-----------------------------------------------------")

    # --- Get user inputs ---
    user_gpa = float(input("Enter your GPA (e.g., 3.6): "))

    sat_or_act = input("Do you want to enter SAT or ACT? (type 'sat' or 'act'): ").strip().lower()
    user_sat, user_act = None, None
    if sat_or_act == "sat":
        user_sat = float(input("Enter your SAT score (out of 1600): "))
    elif sat_or_act == "act":
        user_act = float(input("Enter your ACT score (out of 36): "))

    user_text = input("Enter any preference text (location, major, etc.): ").strip()

    # --- Search ---
    ambitious_unis, practical_unis, safe_unis = search_user_query(
        model, index, mapping,
        gpa=user_gpa, sat=user_sat, act=user_act, text=user_text,
        top_k=30, debug=True
    )

    print("\n\n\n🔍 Recommended Universities")
    print("==============================")

    # --- Display categorized results ---
    print("\n🌠 Ambitious Universities (Top 10)")
    print("-----------------------------------")
    if ambitious_unis:
        for name, sim in ambitious_unis:
            print(f"{name}  (cosine similarity: {sim:.4f})")
    else:
        print("Not enough results found for this category.")

    print("\n🎯 Practical Universities (Ranks 11-20)")
    print("--------------------------------------")
    if practical_unis:
        for name, sim in practical_unis:
            print(f"{name}  (cosine similarity: {sim:.4f})")
    else:
        print("Not enough results found for this category.")

    print("\n✅ Safe Universities (Ranks 21-30)")
    print("-----------------------------------")
    if safe_unis:
        for name, sim in safe_unis:
            print(f"{name}  (cosine similarity: {sim:.4f})")
    else:
        print("Not enough results found for this category.")

    # -------------------- [NEW] Interactive Details Section --------------------
    while True:
        choice = input("\n👉 Enter a university name for full details, or type 'exit' to quit: ").strip()

        if choice.lower() == 'exit':
            print("👋 Goodbye!")
            break

        # Find the university details using the case-insensitive map
        details = university_details_map.get(choice.lower())

        if details:
            display_university_details(details)
        else:
            print(f"❌ University '{choice}' not found. Please ensure the name is spelled correctly.")