import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import umap
import hdbscan

from data_preprocessing import Config

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------
# UMAP + HDBSCAN 파라미터 (속도 최적화 버전)
# -------------------------------------------------------

UMAP_PARAMS = dict(
    n_neighbors=15,
    min_dist=0.1,
    n_components=5,     
    metric="cosine",
    random_state=42
)

HDBSCAN_PARAMS = dict(
    min_cluster_size=10,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=False
)

# -------------------------------------------------------
# 함수 정의
# -------------------------------------------------------

def load_embeddings(slot: str) -> np.ndarray:
    path = Path(Config.OUTPUT_EMBEDDINGS_DIR) / f"embeddings_{slot}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return np.load(path, mmap_mode="r").astype("float32")


def load_slot_parquet(slot: str) -> pd.DataFrame:
    path = Path(Config.OUTPUT_SLOTS_DIR) / f"slot_{slot}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Slot parquet not found: {path}")
    return pd.read_parquet(path)


def save_slot_parquet(slot: str, df: pd.DataFrame):
    path = Path(Config.OUTPUT_SLOTS_DIR) / f"slot_{slot}.parquet"
    df.to_parquet(path, index=False)


# -------------------------------------------------------
# Main Clustering Function
# -------------------------------------------------------

def cluster_slot(slot: str):
    print(f"\n=== Processing slot {slot} ===")

    df = load_slot_parquet(slot)
    embs = load_embeddings(slot)

    if len(df) != embs.shape[0]:
        raise ValueError(
            f"[FATAL] Row / embedding mismatch in slot {slot}: "
            f"{len(df)} rows vs {embs.shape[0]} embeddings."
        )

    print(f"> Loaded {len(df)} rows & embeddings, running UMAP...")

    mapper = umap.UMAP(**UMAP_PARAMS)
    reduced = mapper.fit_transform(embs)

    print("> Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(reduced)

    df["cluster"] = labels

    save_slot_parquet(slot, df)

    print(f"> Slot {slot}: clustering complete. Unique clusters: {len(set(labels))}")


# -------------------------------------------------------
# Orchestrator
# -------------------------------------------------------

def main():
    print("[1/2] Searching for embedding files...")

    emb_paths = sorted(Path(Config.OUTPUT_EMBEDDINGS_DIR).glob("embeddings_*.npy"))
    if not emb_paths:
        raise RuntimeError("No embeddings found. Run embedding stage first.")

    slots = [p.stem.replace("embeddings_", "") for p in emb_paths]
    print(f"> Found {len(slots)} slots: {slots}")

    print("[2/2] Running clustering on each slot...")

    for slot in tqdm(slots, desc="Clustering slots"):
        try:
            cluster_slot(slot)
        except Exception as e:
            print(f"[ERROR] Slot {slot} failed: {e}")

    print("\n>>> All slot clustering finished successfully.")
    print(">>> You can now run topic_keywords_builder.py")


if __name__ == "__main__":
    main()
