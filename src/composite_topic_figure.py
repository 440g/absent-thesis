import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import pickle


META_PATH = Path("dataset/faiss/slot_metadata.pkl")
KEYWORD_DIR = Path("results/topic_keywords")
SLOT_DIR = Path("dataset/slots")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

with open(META_PATH, "rb") as f:
    META = pickle.load(f)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


SUMM_MODEL = "sshleifer/distilbart-cnn-6-6"

tokenizer = AutoTokenizer.from_pretrained(SUMM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(SUMM_MODEL).to(DEVICE)

def local_summarize(text, max_length=128, min_length=32):
    if not text.strip():
        return "(empty text)"

    # too long text → truncate
    text = text[:4000]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# -----------------------------------------------------
# Utility: Load slot keywords
# -----------------------------------------------------
def load_keywords(slot, cid):
    path = KEYWORD_DIR / f"{slot}_keywords.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        kw_map = json.load(f)
    return kw_map.get(str(cid)) or kw_map.get(int(cid)) or []


# -----------------------------------------------------
# summarization 
# -----------------------------------------------------

def slot_summary(slot, cid, max_docs=40):
    df = pd.read_parquet(f"{SLOT_DIR}/slot_{slot}.parquet")
    df = df[df.cluster == cid]

    if df.empty:
        return ["(no documents found)"]

    docs = df.text_processed.tolist()[:max_docs]
    full_text = " ".join(docs)

    summary = local_summarize(full_text)

    return [f"During {slot}, topic {cid} focused on: {summary}"]


# -----------------------------------------------------
# Plot: Keyword Timeline 
# -----------------------------------------------------
def draw_keyword_timeline(ax, slots, cids):

    slot_keywords = {s: load_keywords(s, c)[:12] for s, c in zip(slots, cids)}
    all_kw = sorted(set(k for ks in slot_keywords.values() for k in ks))
    if not all_kw:
        ax.text(0.5, 0.5, "No keywords available", ha="center", va="center")
        ax.axis("off")
        return

    kw_idx = {k: i for i, k in enumerate(all_kw)}

    for k in all_kw:
        xs, ys = [], []
        for si, slot in enumerate(slots):
            if k in slot_keywords[slot]:
                xs.append(si)
                ys.append(kw_idx[k])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.5, alpha=0.9)

    ax.set_yticks(range(len(all_kw)))
    ax.set_yticklabels(all_kw, fontsize=8)
    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels(slots, rotation=45)
    ax.set_title("Keyword Timeline (Connected)", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.3)


# -----------------------------------------------------
# Plot: Word Cloud per slot
# -----------------------------------------------------
def draw_wordclouds(fig, ax_region, slots, cids):
    n = len(slots)
    ax_region.axis("off")
    left0, bottom0, width0, height0 = ax_region.get_position().bounds

    sub_axes = []
    for i in range(n):
        sub_left = left0 + (width0 / n) * i
        sub_bottom = bottom0
        sub_width = width0 / n
        sub_height = height0
        ax_sub = fig.add_axes([sub_left, sub_bottom, sub_width, sub_height])
        sub_axes.append(ax_sub)

    for ax, slot, cid in zip(sub_axes, slots, cids):
        kws = load_keywords(slot, cid)
        text = " ".join(kws)
        if not text:
            ax.text(0.5, 0.5, "No keywords", ha="center", va="center")
            ax.axis("off")
            continue

        wc = WordCloud(
            width=800,
            height=600,
            background_color="white",
            colormap="tab20"
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(slot, fontsize=12)
        ax.axis("off")


# -----------------------------------------------------
# Composite Figure
# -----------------------------------------------------
def generate_composite(chain_id, chains_df):
    sub = chains_df[chains_df["chain_id"] == chain_id].sort_values("year")
    slots = sub["slot"].tolist()
    cids = sub["cluster"].tolist()

    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 2.5, 3.5])

    # (1) Summary
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(f"Topic {chain_id} — LLM Summary", fontsize=18)
    ax1.axis("off")

    y = 0.9
    for s, c in zip(slots, cids):
        lines = slot_summary(s, c)
        ax1.text(0.02, y, f"{s} (cluster {c}):", fontsize=12, weight="bold")
        y -= 0.06
        for line in lines:
            ax1.text(0.04, y, f"- {line}", fontsize=11, wrap=True)
            y -= 0.05
        y -= 0.02

    # (2) Keyword Timeline
    ax2 = fig.add_subplot(gs[1])
    draw_keyword_timeline(ax2, slots, cids)

    # (3) WordClouds
    ax3 = fig.add_subplot(gs[2])
    draw_wordclouds(fig, ax3, slots, cids)

    out = OUT_DIR / f"topic_{chain_id}_composite.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved composite figure: {out}")


# -----------------------------------------------------
# Separate Figures for report
# -----------------------------------------------------
def save_summary_only(chain_id, chains_df):
    sub = chains_df[chains_df["chain_id"] == chain_id].sort_values("year")
    slots = sub["slot"].tolist()
    cids = sub["cluster"].tolist()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_title(f"Topic {chain_id} — LLM Summary", fontsize=18)
    ax.axis("off")

    y = 0.9
    for s, c in zip(slots, cids):
        lines = slot_summary(s, c)
        ax.text(0.02, y, f"{s} (cluster {c}):", fontsize=12, weight="bold")
        y -= 0.06
        for line in lines:
            ax.text(0.04, y, f"- {line}", fontsize=11, wrap=True)
            y -= 0.05
        y -= 0.02

    out = OUT_DIR / f"topic_{chain_id}_summary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved summary figure: {out}")


def save_timeline_only(chain_id, chains_df):
    sub = chains_df[chains_df["chain_id"] == chain_id].sort_values("year")
    slots = sub["slot"].tolist()
    cids = sub["cluster"].tolist()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    draw_keyword_timeline(ax, slots, cids)

    out = OUT_DIR / f"topic_{chain_id}_timeline.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved timeline figure: {out}")


def save_wordclouds_only(chain_id, chains_df):
    sub = chains_df[chains_df["chain_id"] == chain_id].sort_values("year")
    slots = sub["slot"].tolist()
    cids = sub["cluster"].tolist()

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)
    draw_wordclouds(fig, ax, slots, cids)

    out = OUT_DIR / f"topic_{chain_id}_wordclouds.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved wordcloud figure: {out}")


# -----------------------------------------------------
# Topic Chain Construction
# -----------------------------------------------------
def parse_year_from_slot(slot):
    s, e = slot.split("_")
    return (int(s) + int(e)) / 2.0


def build_topic_chains(vol_df, matching, min_length=2):
    # backward map
    backward = {}
    for key, match in matching.items():
        sA, sB = key.split("->")
        for cidA, info in match.items():
            if info["match"] is None:
                continue
            cidB = int(info["match"])
            backward[(sB, cidB)] = (sA, int(cidA))

    # all clusters
    all_pairs = {
        (row["slot"], int(row["cluster"]))
        for _, row in vol_df.iterrows()
        if int(row["cluster"]) != -1
    }

    chain_starts = [(slot, cid) for (slot, cid) in all_pairs if (slot, cid) not in backward]

    # forward map
    forward = {}
    for key, match in matching.items():
        sA, sB = key.split("->")
        for cidA, info in match.items():
            if info["match"] is None:
                continue
            cidB = int(info["match"])
            forward[(sA, int(cidA))] = (sB, cidB)

    rows = []
    chain_id = 0
    vol_lookup = {
        (row["slot"], int(row["cluster"])): int(row["volume"])
        for _, row in vol_df.iterrows()
    }

    for s0, c0 in chain_starts:
        slot, cid = s0, c0
        chain = []
        while True:
            chain.append(
                {
                    "chain_id": chain_id,
                    "slot": slot,
                    "year": parse_year_from_slot(slot),
                    "cluster": cid,
                    "volume": vol_lookup.get((slot, cid), 0),
                }
            )
            nxt = forward.get((slot, cid))
            if nxt is None:
                break
            slot, cid = nxt

        if len(chain) >= min_length:
            rows.extend(chain)
            chain_id += 1

    return pd.DataFrame(rows)


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    vol_df = pd.read_csv("results/topic_trends/cluster_volume.csv")
    with open("results/topic_trends/cluster_matching.json", "r", encoding="utf-8") as f:
        matching = json.load(f)

    chains_df = build_topic_chains(vol_df, matching, min_length=2)
    if chains_df.empty:
        print("No topic chains available.")
        return

    agg = chains_df.groupby("chain_id")["volume"].sum().reset_index()
    target = int(agg.sort_values("volume", ascending=False).iloc[0]["chain_id"])
    print(f"[+] Target chain_id: {target}")


    generate_composite(target, chains_df)

    save_summary_only(target, chains_df)
    save_timeline_only(target, chains_df)
    save_wordclouds_only(target, chains_df)


if __name__ == "__main__":
    main()
