import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm

from data_preprocessing import Config


def load_slot_parquet(slot: str) -> pd.DataFrame:
    path = Path(Config.OUTPUT_SLOTS_DIR) / f"slot_{slot}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Slot parquet not found: {path}")
    return pd.read_parquet(path)


def get_all_slots() -> List[str]:
    p = Path(Config.OUTPUT_SLOTS_DIR)
    files = sorted(p.glob("slot_*.parquet"))
    slots = [f.stem.replace("slot_", "") for f in files]
    return slots


# ---------------------------------------------------------------
# c-TF-IDF 
# ---------------------------------------------------------------
def compute_c_tf_idf(
    docs: List[str],
    m: int
) -> (np.ndarray, List[str]):


    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)  # term frequencies
    vocab = vectorizer.get_feature_names_out()

    transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
    c_tf_idf = transformer.fit_transform(X)
    return c_tf_idf.toarray(), vocab


# ---------------------------------------------------------------
#  keyword extraction
# ---------------------------------------------------------------
def extract_keywords_for_slot(
    slot: str,
    top_n: int = 15
) -> Dict[int, List[str]]:
    """
    returns:
        { cluster_label: [top keywords] }
    """
    df = load_slot_parquet(slot)

    if "cluster" not in df.columns:
        raise ValueError(
            f"Slot {slot} does not contain 'cluster' column. "
            f"Run clustering before building keywords."
        )

    df = df[df["cluster"] != -1].copy()
    if df.empty:
        return {}

    grouped = df.groupby("cluster")

    docs = []
    cluster_labels = []

    for cluster_id, subdf in grouped:
        text = " ".join(subdf["text_processed"].astype(str).tolist())
        docs.append(text)
        cluster_labels.append(cluster_id)

    m = len(docs)
    if m == 0:
        return {}

    ctfidf_matrix, vocab = compute_c_tf_idf(docs, m)

    keywords_per_cluster = {}
    for idx, cluster_id in enumerate(cluster_labels):
        row = ctfidf_matrix[idx]
        top_idx = row.argsort()[::-1][:top_n]
        keywords = [vocab[i] for i in top_idx]
        keywords_per_cluster[cluster_id] = keywords

    return keywords_per_cluster


def save_results(slot: str, keywords: Dict[int, List[str]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{slot}_keywords.json"
    csv_path = out_dir / f"{slot}_keywords.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)

    rows = []
    for cid, keys in keywords.items():
        for k in keys:
            rows.append({"slot": slot, "cluster": cid, "keyword": k})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"> Saved keywords for slot {slot}: {json_path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cluster-based c-TF-IDF topic keyword extraction"
    )
    parser.add_argument("--top_n", type=int, default=15, help="상위 N 키워드")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/topic_keywords",
        help="결과 저장 디렉토리"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    slots = get_all_slots()

    print(f"[1/2] Found {len(slots)} slots.")
    for slot in tqdm(slots, desc="Extracting keywords"):
        try:
            keywords = extract_keywords_for_slot(slot, top_n=args.top_n)
            save_results(slot, keywords, out_dir)
        except Exception as e:
            print(f"[Error] Slot {slot} failed: {e}")

    print("[2/2] Done.")


if __name__ == "__main__":
    main()
