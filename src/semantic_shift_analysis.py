import argparse
import json
import re
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm

from data_preprocessing import Config


STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "on", "with", "as",
    "by", "from", "at", "this", "be", "are", "was", "it", "an", "or", "its",
}


def load_slot_metadata() -> Dict[str, Dict]:
    meta_path = Path(Config.OUTPUT_FAISS_DIR) / "slot_metadata.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"slot_metadata.pkl not found: {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta


def load_slot_parquet(slot: str) -> pd.DataFrame:
    path = Path(Config.OUTPUT_SLOTS_DIR) / f"slot_{slot}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_embeddings(slot: str, meta: Dict[str, Dict]) -> np.ndarray:
    emb_path = meta[slot]["embeddings_path"]
    return np.load(emb_path, mmap_mode="r").astype("float32")


def load_faiss_index(slot: str, meta: Dict[str, Dict]) -> faiss.Index:
    index_path = meta[slot]["faiss_index_path"]
    if not Path(index_path).exists():
        raise FileNotFoundError(index_path)
    return faiss.read_index(index_path)


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    """L2 정규화된 centroid 벡터 반환"""
    if vectors.size == 0:
        raise ValueError("vectors is empty")
    c = vectors.mean(axis=0)
    norm = np.linalg.norm(c) + 1e-12
    return (c / norm).astype("float32")


def compute_neighbors(
    query_vec: np.ndarray,
    index: faiss.Index,
    doc_ids: List[str],
    topk: int = 15,
) -> List[str]:
    """FAISS에서 query_vec 기준 topk neighbor 문서 id 반환"""
    q = query_vec.reshape(1, -1).astype("float32")
    D, I = index.search(q, topk + 1)
    idxs = I[0].tolist()
    ids = [doc_ids[i] for i in idxs if 0 <= i < len(doc_ids)]
    # 첫 번째는 자기 자신일 가능성이 높으니 제외
    return ids[1 : topk + 1]


# ------------------------------------------------
# anchor 단어 선정
# ------------------------------------------------
def extract_top_words_for_slot(
    df: pd.DataFrame,
    max_features: int = 3000,
) -> List[str]:
    """
    슬롯 하나에 대해 text_processed 전체를 하나의 문서로 보고 TF-IDF 상위 단어 추출
    """
    text_series = df["text_processed"].dropna().astype(str)
    if text_series.empty:
        return []
    corpus = [" ".join(text_series.tolist())]
    vec = TfidfVectorizer(max_features=max_features)
    X = vec.fit_transform(corpus)
    vocab = vec.get_feature_names_out()
    return list(vocab)


def select_anchor_words(
    meta: Dict[str, Dict],
    use_slots: List[str],
    topn_per_slot: int = 3000,
    min_slot_ratio: float = 0.7,
    min_global_freq: int = 10,
) -> List[str]:

    slot_topwords: Dict[str, set] = {}
    global_doc_freq: Dict[str, int] = {}

    for s in tqdm(use_slots, desc="Collecting top words per slot"):
        df = load_slot_parquet(s)
        top_words = extract_top_words_for_slot(df, max_features=topn_per_slot)
        slot_topwords[s] = set(top_words)

        text_series = df["text_processed"].astype(str)
        for w in top_words:
            mask = text_series.str.contains(fr"\b{re.escape(w)}\b", regex=True)
            global_doc_freq[w] = global_doc_freq.get(w, 0) + int(mask.sum())
        del df, text_series

    required = max(1, int(len(use_slots) * min_slot_ratio))
    slot_count: Dict[str, int] = {}
    for s, words in slot_topwords.items():
        for w in words:
            slot_count[w] = slot_count.get(w, 0) + 1

    anchors = []
    for w, cnt in slot_count.items():
        if cnt < required:
            continue
        if global_doc_freq.get(w, 0) < min_global_freq:
            continue
        if w in STOPWORDS:
            continue
        if not re.match(r"^[a-z0-9]+$", w):
            continue
        anchors.append(w)

    anchors = sorted(set(anchors))
    return anchors


# ------------------------------------------------
# 단어별 drift 계산 
# ------------------------------------------------
def compute_semantic_drift(
    anchors: List[str],
    meta: Dict[str, Dict],
    use_slots: List[str],
    topk_neighbors: int = 15,
    weight_neighbor: float = 0.6,
    weight_cosine: float = 0.4,
) -> List[Dict]:

    prev_centroid: Dict[str, np.ndarray] = {w: None for w in anchors}
    prev_neighbors: Dict[str, List[str]] = {w: [] for w in anchors}
    cos_sum: Dict[str, float] = {w: 0.0 for w in anchors}
    cos_count: Dict[str, int] = {w: 0 for w in anchors}
    jac_sum: Dict[str, float] = {w: 0.0 for w in anchors}
    jac_count: Dict[str, int] = {w: 0 for w in anchors}
    global_freq: Dict[str, int] = {w: 0 for w in anchors}

    for s_idx, slot in enumerate(tqdm(use_slots, desc="Slots for drift")):
        df = load_slot_parquet(slot)
        embs = load_embeddings(slot, meta)
        index = load_faiss_index(slot, meta)
        ids = [str(_id) for _id in meta[slot]["ids"]]

        if embs.shape[0] != len(df):
            raise ValueError(
                f"Slot {slot}: embeddings count {embs.shape[0]} != dataframe rows {len(df)}"
            )

        texts = df["text_processed"].astype(str).tolist()
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts) 
        vocab = vectorizer.vocabulary_  

        for w in anchors:
            if w not in vocab:
                cur_centroid = None
                cur_neighbors: List[str] = []
            else:
                col_idx = vocab[w]
                doc_idx = X[:, col_idx].nonzero()[0]
                if doc_idx.size == 0:
                    cur_centroid = None
                    cur_neighbors = []
                else:
                    vecs = embs[doc_idx]
                    cur_centroid = compute_centroid(vecs)
                    cur_neighbors = compute_neighbors(
                        cur_centroid,
                        index,
                        ids,
                        topk=topk_neighbors,
                    )
                    global_freq[w] += int(doc_idx.size)

            prev_c = prev_centroid[w]
            prev_n = prev_neighbors[w]

            if prev_c is not None and cur_centroid is not None:
                cos = 1.0 - float(np.dot(prev_c, cur_centroid))
                cos_sum[w] += cos
                cos_count[w] += 1

                n1, n2 = set(prev_n), set(cur_neighbors)
                if not n1 and not n2:
                    j = 0.0
                else:
                    inter = len(n1 & n2)
                    union = len(n1 | n2) + 1e-12
                    j = 1.0 - float(inter / union)
                jac_sum[w] += j
                jac_count[w] += 1

            prev_centroid[w] = cur_centroid
            prev_neighbors[w] = cur_neighbors

        del df, embs, index, ids, texts, vectorizer, X, vocab

    rows: List[Dict] = []
    for w in anchors:
        if cos_count[w] == 0 and jac_count[w] == 0:
            continue

        mean_cos = cos_sum[w] / cos_count[w] if cos_count[w] > 0 else 0.0
        mean_jac = jac_sum[w] / jac_count[w] if jac_count[w] > 0 else 0.0
        combined = weight_neighbor * mean_jac + weight_cosine * mean_cos

        rows.append(
            {
                "word": w,
                "global_freq": int(global_freq[w]),
                "cosine_drift": float(mean_cos),
                "neighbor_drift": float(mean_jac),
                "combined_score": float(combined),
            }
        )

    return rows


# ------------------------------------------------
# 메인
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SBERT semantic shift "
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./results/sbert_semantic_shift_words.csv"
    )
    parser.add_argument(
        "--anchor_topn",
        type=int,
        default=3000
    )
    parser.add_argument(
        "--min_slot_ratio",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--min_global_freq",
        type=int,
        default=10
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=15
    )
    parser.add_argument(
        "--weight_neighbor",
        type=float,
        default=0.6
    )
    parser.add_argument(
        "--weight_cosine",
        type=float,
        default=0.4)

    args = parser.parse_args()

    meta = load_slot_metadata()
    all_slots = sorted(meta.keys())

    if len(all_slots) <= 2:
        raise RuntimeError(f"Not enough slots. Need > 2, got {len(all_slots)}")

    use_slots = all_slots[2:]
    print(f"Total slots: {len(all_slots)} → Using {len(use_slots)} slots (drop first 2):")
    print("Dropped:", ", ".join(all_slots[:2]))
    print("Using :", ", ".join(use_slots))


    print("[1/3] Selecting anchor words...")
    anchors = select_anchor_words(
        meta,
        use_slots=use_slots,
        topn_per_slot=args.anchor_topn,
        min_slot_ratio=args.min_slot_ratio,
        min_global_freq=args.min_global_freq,
    )
    print(f"  - selected {len(anchors)} anchor words")

    if not anchors:
        raise RuntimeError("No anchor words selected. Try lowering thresholds.")


    print("[2/3] Computing semantic drift per word...")
    rows = compute_semantic_drift(
        anchors,
        meta,
        use_slots=use_slots,
        topk_neighbors=args.topk,
        weight_neighbor=args.weight_neighbor,
        weight_cosine=args.weight_cosine,
    )

    if not rows:
        raise RuntimeError("No drift rows computed. Check data and thresholds.")

    df = (
        pd.DataFrame(rows)
        .sort_values("combined_score", ascending=False)
        .reset_index(drop=True)
    )


    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("[3/3] Done.")
    print(f"  - saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
