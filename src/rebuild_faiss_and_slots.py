import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from data_preprocessing import (
    Config,
    ensure_dirs,
    load_raw_data,
    extract_and_clean,
    split_into_slots,
)

def main():
    ensure_dirs()

    print("[1/3] Loading base dataframe ...")
    df_raw = load_raw_data()
    df_extracted = extract_and_clean(df_raw)
    df_slots = split_into_slots(df_extracted, Config.SLOT_SIZE)

    emb_dir = Path(Config.OUTPUT_EMBEDDINGS_DIR)
    faiss_dir = Path(Config.OUTPUT_FAISS_DIR)
    slot_dir = Path(Config.OUTPUT_SLOTS_DIR)

    emb_files = sorted(emb_dir.glob("embeddings_*.npy"))
    print(f"[2/3] Found {len(emb_files)} embedding files.")

    slot_metadata = {}

    for emb_path in tqdm(emb_files, desc="Rebuilding FAISS per slot"):
        slot = emb_path.stem.replace("embeddings_", "")
        slot_df = df_slots[df_slots["slot"] == slot].copy()
        if len(slot_df) == 0:
            print(f"> Skipping {slot} (no matching rows in dataframe)")
            continue

        vectors = np.load(emb_path, mmap_mode="r").astype("float32")
        dim = vectors.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        index_path = faiss_dir / f"faiss_{slot}.index"
        faiss.write_index(index, str(index_path))

        slot_metadata[slot] = {
            "embeddings_path": str(emb_path),
            "faiss_index_path": str(index_path),
            "dim": dim,
            "num_vectors": int(vectors.shape[0]),
            "ids": slot_df["id"].astype(str).tolist()
        }

        slot_parquet = slot_dir / f"slot_{slot}.parquet"
        slot_df.to_parquet(slot_parquet, index=False)

        print(f"> Rebuilt FAISS for {slot} (n={len(slot_df)})")

    meta_path = faiss_dir / "slot_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(slot_metadata, f)

    print(f"\n[3/3] Completed. Rebuilt {len(slot_metadata)} slots.")
    print(f"> Metadata saved to: {meta_path}")

if __name__ == "__main__":
    main()
