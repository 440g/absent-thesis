import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import pickle


from data_preprocessing import Config


def load_slot_parquet(slot: str) -> pd.DataFrame:
    path = Path(Config.OUTPUT_SLOTS_DIR) / f"slot_{slot}.parquet"
    return pd.read_parquet(path)


def load_embeddings(slot: str, meta: Dict[str, Dict]) -> np.ndarray:
    return np.load(meta[slot]["embeddings_path"], mmap_mode="r").astype("float32")


def load_slot_metadata() -> Dict[str, Dict]:
    meta_path = Path(Config.OUTPUT_FAISS_DIR) / "slot_metadata.pkl"
    with open(meta_path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------
# Cluster centroid
# -------------------------------------------------
def compute_cluster_centroids(
    df: pd.DataFrame,
    embs: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    cluster_id → centroid vector
    """
    centroids = {}
    grouped = df.groupby("cluster")

    for cid, sdf in grouped:
        if cid == -1:
            continue
        doc_idx = sdf.index.to_list()
        vecs = embs[doc_idx]

        c = vecs.mean(axis=0)
        norm = np.linalg.norm(c) + 1e-12
        centroids[cid] = (c / norm).astype("float32")

    return centroids


# -------------------------------------------------
# Cluster matching algorithm between two slots
# -------------------------------------------------
def match_clusters_between_slots(
    slotA: str,
    slotB: str,
    centA: Dict[int, np.ndarray],
    centB: Dict[int, np.ndarray],
    similarity_threshold: float = 0.45,
) -> Dict[int, Dict]:
    """
    Return:
      match_result = {
         clusterA: {
            "match": clusterB or None,
            "similarity": value
         }
      }
    """

    clustersA = list(centA.keys())
    clustersB = list(centB.keys())

    if not clustersA or not clustersB:
        return {}

    # similarity matrix
    sim_mat = np.zeros((len(clustersA), len(clustersB)), dtype="float32")

    for i, cA in enumerate(clustersA):
        vA = centA[cA]
        for j, cB in enumerate(clustersB):
            vB = centB[cB]
            sim_mat[i, j] = float(np.dot(vA, vB))

    matchedB = set()
    results = {}

    for i, cA in enumerate(clustersA):
        sims = sim_mat[i]
        best_j = sims.argmax()
        best_sim = sims[best_j]

        if best_sim < similarity_threshold:
            results[cA] = {"match": None, "similarity": float(best_sim)}
        else:
            cB = clustersB[best_j]
            if cB in matchedB:
                results[cA] = {"match": None, "similarity": float(best_sim)}
            else:
                matchedB.add(cB)
                results[cA] = {"match": cB, "similarity": float(best_sim)}

    return results


# -------------------------------------------------
# Volume computation
# -------------------------------------------------
def compute_volume(df: pd.DataFrame) -> Dict[int, int]:
    """
    cluster → doc count
    """
    res = {}
    grouped = df.groupby("cluster")
    for cid, sdf in grouped:
        if cid == -1:
            continue
        res[cid] = len(sdf)
    return res


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def main():
    import pickle

    meta = load_slot_metadata()
    DROP_SLOTS = {"1986_1988", "1989_1991"}
    slots = [s for s in sorted(meta.keys()) if s not in DROP_SLOTS]


    out_dir = Path("./results/topic_trends")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_volume_rows = []
    matching_results = {}

    print("[1/3] Computing centroids and volumes per slot...")
    slot_centroids = {}
    slot_volumes = {}

    for s in tqdm(slots, desc="Slots"):
        df = load_slot_parquet(s)
        embs = load_embeddings(s, meta)

        centroids = compute_cluster_centroids(df, embs)
        volumes = compute_volume(df)

        slot_centroids[s] = centroids
        slot_volumes[s] = volumes

        for cid, vol in volumes.items():
            start, end = s.split("_")
            year_label = f"{start}" if start == end else s
            all_volume_rows.append(
                {"slot": s, "year": year_label, "cluster": cid, "volume": vol}
            )

    print("[2/3] Matching clusters between consecutive slots...")
    for i in tqdm(range(len(slots) - 1), desc="Matching"):
        sA = slots[i]
        sB = slots[i + 1]

        centA = slot_centroids[sA]
        centB = slot_centroids[sB]

        matchAB = match_clusters_between_slots(sA, sB, centA, centB)

        matching_results[f"{sA}->{sB}"] = matchAB

    print("[3/3] Saving outputs...")

    df_vol = pd.DataFrame(all_volume_rows)
    df_vol.to_csv(out_dir / "cluster_volume.csv", index=False)
    print(f"  - saved volume data: {out_dir/'cluster_volume.csv'}")

    with open(out_dir / "cluster_matching.json", "w", encoding="utf-8") as f:
        json.dump(matching_results, f, indent=2, ensure_ascii=False)
    print(f"  - saved matching results: {out_dir/'cluster_matching.json'}")

    print("Done.")


if __name__ == "__main__":
    main()
