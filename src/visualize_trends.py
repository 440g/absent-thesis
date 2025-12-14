#!/usr/bin/env python3
# visualize_trends.py
#
# Visualization for:
#  - diachronic_word_embeddings_opt.py  (semantic drift)
#  - topic_keywords_builder.py
#  - topic_volume_trend_generator.py
#
# Compatible with patched pipeline (slot drops, new filenames, etc.)

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEMANTIC_SHIFT_PATH = Path("results/sbert_diachronic_word_change.csv")
TOPIC_KEYWORDS_DIR = Path("results/topic_keywords")
VOLUME_PATH = Path("results/topic_trends/cluster_volume.csv")
MATCHING_PATH = Path("results/topic_trends/cluster_matching.json")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DROP_SLOTS = {"1986_1988", "1989_1991"}


def plot_top_drifting_words(
    csv_path: Path = SEMANTIC_SHIFT_PATH,
    top_n: int = 20,
    out_path: Path = FIG_DIR / "top_drifting_words.png",
):
    df = pd.read_csv(csv_path)

    df = df[df["combined_score"].notnull()]

    if df.empty:
        print("[!] No valid drift scores found.")
        return

    df = df.sort_values("combined_score", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df["word"][::-1], df["combined_score"][::-1])
    plt.xlabel("Semantic drift score (combined)")
    plt.ylabel("Word")
    plt.title(f"Top {top_n} drifting words")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")



def load_volume_and_matching() -> Tuple[pd.DataFrame, Dict]:
    vol_df = pd.read_csv(VOLUME_PATH)

    vol_df = vol_df[~vol_df["slot"].isin(DROP_SLOTS)]

    with open(MATCHING_PATH, "r", encoding="utf-8") as f:
        match_all = json.load(f)

    filtered_match = {}
    for key, v in match_all.items():
        sA, sB = key.split("->")
        if sA in DROP_SLOTS or sB in DROP_SLOTS:
            continue
        filtered_match[key] = v

    return vol_df, filtered_match


def parse_year_from_slot(slot: str) -> float:
    s, e = slot.split("_")
    return (int(s) + int(e)) / 2.0


def build_topic_chains(
    vol_df: pd.DataFrame,
    matching: Dict,
    min_length: int = 2,
) -> pd.DataFrame:

    backward = {}
    forward = {}

    for key, m in matching.items():
        sA, sB = key.split("->")
        for cidA, info in m.items():
            if info["match"] is None:
                continue
            cidB = info["match"]
            cidA = int(cidA)
            cidB = int(cidB)
            forward[(sA, cidA)] = (sB, cidB)
            backward[(sB, cidB)] = (sA, cidA)

    all_pairs = {
        (row["slot"], int(row["cluster"]))
        for _, row in vol_df.iterrows()
        if row["cluster"] != -1
    }

    starts = [(s, c) for (s, c) in all_pairs if (s, c) not in backward]

    vol_lookup = {
        (row["slot"], int(row["cluster"])): int(row["volume"])
        for _, row in vol_df.iterrows()
    }

    chains = []
    chain_id = 0

    for s0, c0 in starts:
        s, c = s0, c0
        chain = []

        while True:
            vol = vol_lookup.get((s, c), 0)
            year_val = parse_year_from_slot(s)
            chain.append({
                "chain_id": chain_id,
                "slot": s,
                "year": year_val,
                "cluster": c,
                "volume": vol,
            })

            nxt = forward.get((s, c))
            if nxt is None:
                break
            s, c = nxt

        if len(chain) >= min_length:
            chains.extend(chain)
            chain_id += 1

    if not chains:
        print("[!] No topic chains constructed.")
        return pd.DataFrame(columns=["chain_id", "slot", "year", "cluster", "volume"])

    return pd.DataFrame(chains)


def plot_top_topic_volume_trends(
    chains_df: pd.DataFrame,
    top_k: int = 5,
    out_path: Path = FIG_DIR / "top_topic_volume_trends.png",
):
    if chains_df.empty:
        print("[!] Empty chains_df. Skip plotting.")
        return

    agg = chains_df.groupby("chain_id")["volume"].sum().reset_index()

    valid_chain_ids = [
        cid for cid in agg["chain_id"]
        if len(chains_df[chains_df["chain_id"] == cid]) >= 2
    ]
    agg = agg[agg["chain_id"].isin(valid_chain_ids)]

    top_ids = agg.sort_values("volume", ascending=False).head(top_k)["chain_id"].tolist()

    plt.figure(figsize=(10, 6))
    for cid in top_ids:
        sub = chains_df[chains_df["chain_id"] == cid].sort_values("year")
        plt.plot(sub["year"], sub["volume"], marker="o", label=f"Topic {cid}")

    plt.xlabel("Year")
    plt.ylabel("Volume (#papers)")
    plt.title(f"Top {top_k} Topic Volume Trends")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")



def load_keywords_for_slot(slot: str) -> Dict[str, List[str]]:
    path = TOPIC_KEYWORDS_DIR / f"{slot}_keywords.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_topic_keywords_over_time(
    chains_df: pd.DataFrame,
    chain_id: int,
    out_path: Path = None,
):
    sub = chains_df[chains_df["chain_id"] == chain_id].sort_values("year")
    if sub.empty:
        print(f"[!] No data for chain_id={chain_id}")
        return

    if out_path is None:
        out_path = FIG_DIR / f"topic_{chain_id}_keywords.png"

    plt.figure(figsize=(10, 0.8 * len(sub) + 2))
    y_positions = list(range(len(sub)))

    plt.scatter([0] * len(y_positions), y_positions, alpha=0)
    plt.xticks([])
    plt.yticks([])

    for y, (_, row) in zip(y_positions, sub.iterrows()):
        slot = row["slot"]
        cluster = str(int(row["cluster"]))
        kw_map = load_keywords_for_slot(slot)
        kw_list = kw_map.get(cluster, [])
        top_kw = ", ".join(kw_list[:8])
        label = f"{slot} (cluster {cluster})"

        plt.text(0.01, y, f"{label}\n{top_kw}", fontsize=9, va="center")

    plt.title(f"Topic {chain_id} Keywords Over Time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {out_path}")



def main():
    plot_top_drifting_words(top_n=20)

    vol_df, matching = load_volume_and_matching()
    chains_df = build_topic_chains(vol_df, matching, min_length=2)

    if not chains_df.empty:
        plot_top_topic_volume_trends(chains_df, top_k=5)

        agg = chains_df.groupby("chain_id")["volume"].sum().reset_index()
        agg = agg.sort_values("volume", ascending=False)
        top_topic_id = int(agg.iloc[0]["chain_id"])

        plot_topic_keywords_over_time(chains_df, chain_id=top_topic_id)
    else:
        print("[!] No topic chains constructed. Skipping plots.")


if __name__ == "__main__":
    main()
