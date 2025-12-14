#!/usr/bin/env python3
# topic_keywords_builder.py (Enhanced + Stopword Removal)
#
# Cluster-based c-TF-IDF topic keyword extraction with:
#   - stopword removal
#   - minimal token filtering (junk removal)
#   - smarter preprocessing
#
# Input:
#   - slot_{slot}.parquet (must contain: text_processed, cluster)
#
# Output:
#   - results/topic_keywords/{slot}_keywords.json
#   - results/topic_keywords/{slot}_keywords.csv

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm

from data_preprocessing import Config

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except:
    STOPWORDS = set()


CUSTOM_STOPWORDS = {
    "using", "used", "based", "approach", "method", "methods",
    "result", "results", "paper", "analysis", "study",
    "model", "models", "data", "system", "systems",
    "approaches", "use", "via"
}

STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

# -----------------------------
# Token Filtering
# -----------------------------
def clean_tokens(text: str) -> str:
    tokens = str(text).lower().split()
    cleaned = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) < 3:
            continue
        if t.isnumeric():
            continue
        cleaned.append(t)

    return " ".join(cleaned)


# -----------------------------
# Slot Loading
# -----------------------------
def load_slot_parquet(slot: str) -> pd.DataFrame:
    path = Path(Config.OUTPUT_SLOTS_DIR) / f"slot_{slot}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def get_all_slots() -> List[str]:
    files = sorted(Path(Config.OUTPUT_SLOTS_DIR).glob("slot_*.parquet"))
    return [f.stem.replace("slot_", "") for f in files]


# -----------------------------
# c-TF-IDF computation
# -----------------------------
def compute_c_tf_idf(docs: List[str], num_docs: int):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    transformer = TfidfTransformer(norm=None)
    tfidf_matrix = transformer.fit_transform(X)

    return tfidf_matrix.toarray(), vocab


# -----------------------------
# Extract Keywords for One Slot
# -----------------------------
def extract_keywords_for_slot(slot: str, top_n: int = 15) -> Dict[int, List[str]]:
    df = load_slot_parquet(slot)

    if "cluster" not in df.columns:
        raise ValueError(f"[Error] Slot {slot} missing 'cluster' column.")

    df = df[df["cluster"] != -1]
    if df.empty:
        return {}

    df["cleaned"] = df["text_processed"].apply(clean_tokens)

    grouped = df.groupby("cluster")

    docs = [] 
    cluster_labels = []

    for cid, sdf in grouped:
        text = " ".join(sdf["cleaned"].astype(str).tolist())
        docs.append(text)
        cluster_labels.append(cid)

    if not docs:
        return {}

    tfidf_matrix, vocab = compute_c_tf_idf(docs, len(docs))

    keywords_per_cluster = {}
    for row_idx, cid in enumerate(cluster_labels):
        row = tfidf_matrix[row_idx]
        top_idx = row.argsort()[::-1][:top_n]

        words = [vocab[i] for i in top_idx if vocab[i] not in STOPWORDS]
        keywords_per_cluster[cid] = words

    return keywords_per_cluster


# -----------------------------
# Save Results
# -----------------------------
def save_results(slot: str, kw: Dict[int, List[str]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{slot}_keywords.json"
    csv_path = out_dir / f"{slot}_keywords.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(kw, f, indent=2, ensure_ascii=False)

    rows = []
    for cid, words in kw.items():
        for w in words:
            rows.append({"slot": slot, "cluster": cid, "keyword": w})

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"[OK] Saved keywords for slot {slot}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="./results/topic_keywords")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    slots = get_all_slots()

    print(f"[1/2] Found {len(slots)} slots")

    for slot in tqdm(slots, desc="Extracting keywords"):
        try:
            kw = extract_keywords_for_slot(slot, top_n=args.top_n)
            save_results(slot, kw, out_dir)
        except Exception as e:
            print(f"[Error] Slot {slot}: {e}")

    print("[2/2] Done.")


if __name__ == "__main__":
    main()
