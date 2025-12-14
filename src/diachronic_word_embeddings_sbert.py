import os
import re
import gc
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from collections import Counter

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from data_preprocessing import Config


def load_slot_metadata() -> Dict[str, Dict]:
    meta_path = Path(Config.OUTPUT_FAISS_DIR) / "slot_metadata.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"slot_metadata.pkl not found at {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta


def load_slot_parquet(slot: str) -> pd.DataFrame:
    path = Path(Config.OUTPUT_SLOTS_DIR) / f"slot_{slot}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Slot parquet not found for {slot}: {path}")
    return pd.read_parquet(path)


def load_slot_embeddings_and_index(slot: str, meta: Dict[str, Dict]) -> Tuple[np.ndarray, faiss.Index, List[str]]:
    info = meta[slot]
    emb_path = info["embeddings_path"]
    index_path = info["faiss_index_path"]
    ids = info["ids"]

    if not Path(emb_path).exists():
        raise FileNotFoundError(f"Embeddings file not found for {slot}: {emb_path}")
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found for {slot}: {index_path}")

    vectors = np.load(emb_path, mmap_mode="r").astype("float32")
    index = faiss.read_index(index_path)

    if vectors.shape[0] != len(ids):
        raise ValueError(
            f"[FATAL] Slot {slot}: vectors.shape[0] ({vectors.shape[0]}) "
            f"!= len(ids) ({len(ids)})"
        )

    return vectors, index, ids


# ============================================================
# Anchor word 선택
# ============================================================
STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "has",
    "are", "was", "were", "been", "their", "there", "here", "such", "than",
    "into", "onto", "within", "without", "between", "about", "above", "below",
    "which", "while", "where", "when", "also", "using", "use", "used",
    "can", "could", "may", "might", "shall", "will", "would", "should"
}


def build_anchor_words(meta: Dict[str, Dict],
                       max_anchors: int = 300,
                       min_slots_ratio: float = 0.5,
                       min_global_freq: int = 20) -> List[str]:
    slots = sorted(meta.keys())
    n_slots = len(slots)
    required_slots = max(1, int(n_slots * min_slots_ratio))

    slot_vocab = {} 
    global_freq = Counter()
    slot_presence = Counter()

    print("[1/4] Building anchor words (per-slot frequency scan)...")

    for slot in tqdm(slots, desc="Scanning slots for anchors"):
        df = load_slot_parquet(slot)
        cnt = Counter()

        for text in df["text_processed"].astype(str):
            tokens = text.split()
            cnt.update(tokens)

        top_in_slot = Counter(dict(cnt.most_common(5000)))
        slot_vocab[slot] = top_in_slot

        for w, c in top_in_slot.items():
            global_freq[w] += c
            slot_presence[w] += 1

        del df, cnt, top_in_slot
        gc.collect()

    candidates = []
    for w, c in global_freq.items():
        if w in STOPWORDS:
            continue
        if not w.isalpha():
            continue
        if c < min_global_freq:
            continue
        if slot_presence[w] < required_slots:
            continue
        candidates.append((w, c, slot_presence[w]))

    candidates.sort(key=lambda x: (-x[1], -x[2]))
    anchors = [w for w, _, _ in candidates[:max_anchors]]

    print(f"> Total candidate anchors before truncation: {len(candidates)}")
    print(f"> Using top {len(anchors)} anchors (max={max_anchors})")

    return anchors


# ============================================================
# Drift 
# ============================================================
def compute_centroid(vectors: np.ndarray) -> np.ndarray:

    if vectors.shape[0] == 0:
        raise ValueError("Empty vectors for centroid computation")
    c = np.mean(vectors, axis=0)
    n = np.linalg.norm(c) + 1e-12
    return c / n


def faiss_neighbors_for_vector(q_vec: np.ndarray,
                               emb_matrix: np.ndarray,
                               index: faiss.Index,
                               doc_ids: List[str],
                               topk: int = 15) -> List[str]:

    q = q_vec.astype("float32").reshape(1, -1)
    D, I = index.search(q, topk + 1)
    idxs = I[0].tolist()
    ids = [doc_ids[i] for i in idxs]
    return ids[1: topk + 1]


def find_docs_containing_word(df: pd.DataFrame, word: str) -> np.ndarray:
    pattern = rf"(^| ){re.escape(word)}( |$)"
    mask = df["text_processed"].astype(str).str.contains(pattern, regex=True)
    return df[mask].index.to_numpy()


# ============================================================
# 메인
# ============================================================
def compute_semantic_drift(anchors: List[str],
                           meta: Dict[str, Dict],
                           topk_neighbors: int = 15) -> List[Dict]:
    """
    anchors: semantic drift를 측정할 단어 리스트
    meta: slot metadata (slot -> info)
    """
    slots = sorted(meta.keys())
    results = []

    print("[2/4] Computing semantic drift (anchor-wise, slot-wise streaming)...")

    for w in tqdm(anchors, desc="Anchors"):
        centroids = []
        neighbor_lists = []

        for slot in slots:
            df = load_slot_parquet(slot)
            emb_matrix, index, doc_ids = load_slot_embeddings_and_index(slot, meta)

            doc_idxs = find_docs_containing_word(df, w)
            if doc_idxs.size == 0:
                centroids.append(None)
                neighbor_lists.append([])
            else:
                slot_vecs = emb_matrix[doc_idxs]
                c = compute_centroid(slot_vecs)
                centroids.append(c)

                nn_ids = faiss_neighbors_for_vector(c, emb_matrix, index, doc_ids, topk=topk_neighbors)
                neighbor_lists.append(nn_ids)

            del df, emb_matrix, index, doc_ids, doc_idxs
            gc.collect()

        cos_diffs = []
        jacc_diffs = []

        for i in range(len(slots) - 1):
            v1 = centroids[i]
            v2 = centroids[i + 1]
            if v1 is None or v2 is None:
                continue

            cos_sim = float(np.dot(v1, v2))
            cos_diff = 1.0 - cos_sim
            cos_diffs.append(cos_diff)

            n1 = set(neighbor_lists[i])
            n2 = set(neighbor_lists[i + 1])
            if not n1 and not n2:
                jac_diff = 0.0
            else:
                inter = len(n1 & n2)
                union = len(n1 | n2) + 1e-12
                jaccard_sim = inter / union
                jac_diff = 1.0 - jaccard_sim
            jacc_diffs.append(jac_diff)

        if cos_diffs:
            cosine_drift = float(np.mean(cos_diffs))
        else:
            cosine_drift = None

        if jacc_diffs:
            neighbor_drift = float(np.mean(jacc_diffs))
        else:
            neighbor_drift = None

        if cosine_drift is not None and neighbor_drift is not None:
            combined = 0.6 * neighbor_drift + 0.4 * cosine_drift
        else:
            combined = None

        results.append({
            "word": w,
            "cosine_drift": cosine_drift,
            "neighbor_drift": neighbor_drift,
            "combined_score": combined,
        })

    return results


# ============================================================
# 메인 엔트리
# ============================================================
def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    print("[0/4] Loading slot metadata...")
    meta = load_slot_metadata()
    print(f"> Slots found: {len(meta)}")

    anchors = build_anchor_words(
        meta,
        max_anchors=300,
        min_slots_ratio=0.5,
        min_global_freq=20 
    )
    print(f"> Final anchor count: {len(anchors)}")

    if not anchors:
        raise RuntimeError("No anchor words selected. Try lowering thresholds.")

    rows = compute_semantic_drift(
        anchors,
        meta,
        topk_neighbors=15
    )

    print("[3/4] Aggregating and saving results...")

    df = pd.DataFrame(rows)
    df["combined_score_safe"] = df["combined_score"].fillna(-1.0)
    df = df.sort_values("combined_score_safe", ascending=False).drop(columns=["combined_score_safe"])
    df.reset_index(drop=True, inplace=True)

    out_path = Path("./results/sbert_diachronic_word_change.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[4/4] Saved results to: {out_path}")
    print(df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
