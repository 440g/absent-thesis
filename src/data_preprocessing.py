# data_preprocessing.py
# ArXiv Semantic Shift Analysis - MPS / memory-efficient version
# - Slot-wise memmap embedding writing (no big in-memory arrays)
# - SBERT encode uses device=str(device) and torch.no_grad()
# - FAISS built from per-slot .npy files

import os
import json
import re
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import torch


class Config:
    DATA_FILE_PATH = './dataset/arxiv-metadata-oai-snapshot.json'
    OUTPUT_RAW_DATA_CACHE = './dataset/arxiv_raw_cache.pkl'
    OUTPUT_EXTRACTED_PARQUET = './dataset/extracted.parquet'
    OUTPUT_PROCESSED_DATA_PATH = './dataset/processed_arxiv_with_embeddings.pkl'
    OUTPUT_SLOTS_DIR = './dataset/slots'
    OUTPUT_FAISS_DIR = './dataset/faiss'
    OUTPUT_EMBEDDINGS_DIR = './dataset/embeddings'  

    MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
    BATCH_SIZE = 64
    SLOT_SIZE = 3 
    MIN_TEXT_LENGTH = 50  


def ensure_dirs():
    for d in [Path(Config.OUTPUT_SLOTS_DIR), Path(Config.OUTPUT_FAISS_DIR), Path(Config.OUTPUT_EMBEDDINGS_DIR)]:
        d.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 1. Load Raw JSON
# -----------------------------
def load_raw_data(path: str = Config.DATA_FILE_PATH) -> pd.DataFrame:
    if Path(Config.OUTPUT_RAW_DATA_CACHE).exists():
        print(f"> Raw data cache found. Loading from: {Config.OUTPUT_RAW_DATA_CACHE}")
        with open(Config.OUTPUT_RAW_DATA_CACHE, 'rb') as f:
            df = pickle.load(f)
        print(f"> Successfully loaded {len(df)} raw records.")
        return df

    if not Path(path).exists():
        raise FileNotFoundError(f"Not found: {path}")

    rows = []
    print(f"> Loading raw JSON from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSON"):
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    Path(Config.OUTPUT_RAW_DATA_CACHE).parent.mkdir(parents=True, exist_ok=True)
    with open(Config.OUTPUT_RAW_DATA_CACHE, 'wb') as f:
        pickle.dump(df, f)

    print(f"> Raw data loaded and cached. Total records: {len(df)}")
    return df


# -----------------------------
# 2. Extract & Clean
# -----------------------------
def extract_and_clean(df: pd.DataFrame, min_text_length: int = Config.MIN_TEXT_LENGTH) -> pd.DataFrame:
    def extract_year_field(created_val):
        if not isinstance(created_val, str):
            return None
        m = re.search(r"(19|20)\d{2}", created_val)
        return int(m.group()) if m else None

    if 'created' not in df.columns and 'versions' in df.columns:
        def extract_from_versions(vs):
            try:
                if isinstance(vs, list) and len(vs) > 0:
                    created = vs[0].get('created', '')
                    m = re.search(r"(19|20)\d{2}", created)
                    return int(m.group()) if m else None
            except Exception:
                return None
            return None
        df['year'] = df['versions'].apply(extract_from_versions)
    else:
        df['year'] = df['created'].apply(extract_year_field)

    df = df.dropna(subset=['year']).reset_index(drop=True)
    df['year'] = df['year'].astype(int)

    df['title'] = df.get('title', '').astype(str).str.strip()
    df['abstract'] = df.get('abstract', '').astype(str).str.strip()

    df['text'] = (df['title'] + ' ' + df['abstract']).str.strip()

    df = df[df['text'].str.len() >= min_text_length].reset_index(drop=True)

    df['text_processed'] = df['text'].apply(preprocess_for_topic_labeling)

    keep_cols = ['id', 'title', 'abstract', 'text', 'text_processed', 'year', 'categories']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    df_out = df[keep_cols].copy()
    df_out.to_parquet(Config.OUTPUT_EXTRACTED_PARQUET, index=False)
    print(f"> Extracted and cleaned. Records kept: {len(df_out)}")
    return df_out


# -----------------------------
# 3. Topic labeling preprocess 
# -----------------------------
def preprocess_for_topic_labeling(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if len(t) > 2]
    return ' '.join(tokens)


# -----------------------------
# 4. Slot splitting 
# -----------------------------
def split_into_slots(df: pd.DataFrame, slot_size: int = Config.SLOT_SIZE) -> pd.DataFrame:
    min_y, max_y = int(df['year'].min()), int(df['year'].max())
    slots = []

    start = min_y
    while start <= max_y:
        end = start + slot_size - 1
        slot_df = df[(df['year'] >= start) & (df['year'] <= end)].copy()
        if len(slot_df) > 0:
            slot_name = f"{start}_{end}"
            slot_df['slot'] = slot_name
            slots.append(slot_df)
        start = end + 1

    if not slots:
        return df.assign(slot='none')

    df_slots = pd.concat(slots, ignore_index=True)
    return df_slots


# -----------------------------
# 5. Compute embeddings 
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_embeddings_batch_slotwise(df: pd.DataFrame,
                                      model_name: str = Config.MODEL_NAME,
                                      batch_size: int = Config.BATCH_SIZE,
                                      embeddings_dir: str = Config.OUTPUT_EMBEDDINGS_DIR) -> Dict[str, str]:

    device = get_device()
    print(f"> Using device: {device}")
    print(f"> Loading SBERT model {model_name} on device {device}...")
    model = SentenceTransformer(model_name)
    model.to(device)
    model.max_seq_length = 512

    ensure_dirs()

    slot_embedding_paths = {}

    for slot, slot_df in tqdm(df.groupby('slot'), desc="Slots Embedding"):
        titles = slot_df['title'].tolist()
        abstracts = slot_df['abstract'].tolist()
        n = len(titles)
        if n == 0:
            continue

        with torch.no_grad():
            sample_te = model.encode([titles[0]], convert_to_numpy=True, device=str(device), normalize_embeddings=True)
            sample_ae = model.encode([abstracts[0] if abstracts[0] else " "], convert_to_numpy=True, device=str(device), normalize_embeddings=True)
        dim = sample_te.shape[1]

        emb_path = Path(embeddings_dir) / f"embeddings_{slot}.npy"
        memmap = np.lib.format.open_memmap(emb_path, mode='w+', dtype='float32', shape=(n, dim))

        idx = 0
        with torch.no_grad():
            for i in range(0, n, batch_size):
                bt = titles[i:i+batch_size]
                ba = abstracts[i:i+batch_size]
                te = model.encode(bt, convert_to_numpy=True, device=str(device), normalize_embeddings=True)
                ae = model.encode(ba, convert_to_numpy=True, device=str(device), normalize_embeddings=True)
                combined = (te + ae) / 2.0

                norms = np.linalg.norm(combined, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                combined = combined / norms

                memmap[idx: idx + combined.shape[0]] = combined.astype('float32')
                idx += combined.shape[0]

        del memmap

        slot_embedding_paths[slot] = str(emb_path)
        print(f"> Slot {slot}: saved embeddings to {emb_path} (n={n}, dim={dim})")

    return slot_embedding_paths


# -----------------------------
# 6. Build FAISS per-slot 
# -----------------------------
def save_embeddings_and_build_faiss_from_files(slot_embedding_paths: Dict[str, str],
                                               df: pd.DataFrame,
                                               embeddings_dir: str = Config.OUTPUT_EMBEDDINGS_DIR,
                                               faiss_dir: str = Config.OUTPUT_FAISS_DIR) -> Dict[str, Dict]:

    ensure_dirs()
    slot_metadata = {}

    for slot, emb_path in tqdm(slot_embedding_paths.items(), desc="Building FAISS per slot"):
        vectors = np.load(emb_path, mmap_mode='r').astype('float32')
        ids = df[df['slot'] == slot]['id'].astype(str).tolist()

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim) 
        index.add(vectors)

        index_path = Path(faiss_dir) / f"faiss_{slot}.index"
        faiss.write_index(index, str(index_path))

        slot_metadata[slot] = {
            'embeddings_path': str(emb_path),
            'faiss_index_path': str(index_path),
            'dim': dim,
            'num_vectors': int(vectors.shape[0]),
            'ids': ids
        }

        slot_parquet = Path(Config.OUTPUT_SLOTS_DIR) / f"slot_{slot}.parquet"
        slot_df = df[df['slot'] == slot].copy().reset_index(drop=True)
        slot_df.to_parquet(slot_parquet, index=False)

        print(f"> FAISS saved for slot {slot}: {index_path} (vectors={vectors.shape[0]})")

    meta_path = Path(faiss_dir) / 'slot_metadata.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(slot_metadata, f)

    print(f"> Saved embeddings and FAISS indices for {len(slot_metadata)} slots. Metadata: {meta_path}")
    return slot_metadata


# -----------------------------
# 7. Semantic shift tracking helper 
# -----------------------------
def track_semantic_shift(keyword: str, slot_metadata: Dict[str, Dict], top_k: int = 10, model_name: str = Config.MODEL_NAME):
    device = get_device()
    model = SentenceTransformer(model_name)
    model.to(device)
    q = model.encode(keyword, convert_to_numpy=True, device=str(device), normalize_embeddings=True)
    q = q.astype('float32')

    results = {}
    for slot, meta in slot_metadata.items():
        index = faiss.read_index(meta['faiss_index_path'])
        D, I = index.search(q.reshape(1, -1), top_k)
        ids = [meta['ids'][i] for i in I[0]]
        results[slot] = {
            'distances': D[0].tolist(),
            'ids': ids
        }
    return results


# -----------------------------
# 8. Orchestration
# -----------------------------
def run_pipeline(slot_size: int = Config.SLOT_SIZE):
    ensure_dirs()

    print("[1/5] Loading raw data...")
    df_raw = load_raw_data()

    print("[2/5] Extracting and cleaning...")
    df_extracted = extract_and_clean(df_raw)

    print("[3/5] Splitting into time slots...")
    df_slots = split_into_slots(df_extracted, slot_size)

    print("[4/5] Computing embeddings (slot-wise, batch) ...")
    slot_embedding_paths = compute_embeddings_batch_slotwise(df_slots)

    # persist intermediate processed data (pickle)
    Path(Config.OUTPUT_PROCESSED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    # Save df_slots (without keeping huge embedding lists)
    with open(Config.OUTPUT_PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(df_slots, f)

    print("[5/5] Building FAISS indices from per-slot files ...")
    metadata = save_embeddings_and_build_faiss_from_files(slot_embedding_paths, df_slots)

    print("Pipeline completed successfully.")
    return df_slots, metadata


if __name__ == '__main__':
    df_final, meta = run_pipeline()
    print(f"Final records: {len(df_final)}; slots: {len(meta)}")
