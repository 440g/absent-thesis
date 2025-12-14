import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


SEM_PATH = Path("results/sbert_diachronic_word_change.csv")
VOL_PATH = Path("results/topic_trends/cluster_volume.csv")
MATCH_PATH = Path("results/topic_trends/cluster_matching.json")
KW_DIR = Path("results/topic_keywords")
FIG_DIR = Path("figures_enhanced")
FIG_DIR.mkdir(exist_ok=True)

DROP_SLOTS = {"1986_1988","1989_1991"}

sns.set_theme(style="whitegrid")


def plot_semantic_drift(top_n=20):
    df = pd.read_csv(SEM_PATH)
    df = df[df["combined_score"].notnull()]
    df = df.sort_values("combined_score", ascending=False).head(top_n)

    colors = sns.color_palette("flare", n_colors=len(df))

    plt.figure(figsize=(11, 7))
    bars = plt.barh(df["word"][::-1], df["combined_score"][::-1], color=colors[::-1])
    plt.title(f"Top {top_n} Most Semantically Drifting Words", fontsize=16)
    plt.xlabel("Semantic Drift Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "semantic_drift.png", dpi=350)
    plt.close()



def load_data():
    vol = pd.read_csv(VOL_PATH)
    vol = vol[~vol["slot"].isin(DROP_SLOTS)]
    with open(MATCH_PATH, "r") as f:
        m = json.load(f)
    # drop obsolete edges
    m2 = {k:v for k,v in m.items() if not any(s in DROP_SLOTS for s in k.split("->"))}
    return vol, m2


def parse_year(slot):
    s,e = slot.split("_")
    return (int(s)+int(e))/2


def build_chains(vol_df, matching):
    forward = {}
    backward = {}

    for key,v in matching.items():
        sA, sB = key.split("->")
        for cidA, info in v.items():
            if info["match"] is None:
                continue
            cidB = info["match"]
            forward[(sA,int(cidA))] = (sB,int(cidB))
            backward[(sB,int(cidB))] = (sA,int(cidA))

    all_pairs = {(row.slot,int(row.cluster)) for _,row in vol_df.iterrows() if row.cluster!=-1}

    starts = [(s,c) for (s,c) in all_pairs if (s,c) not in backward]

    vol_lookup = {(row.slot,int(row.cluster)):int(row.volume) for _,row in vol_df.iterrows()}

    chains=[]
    chain_id=0
    for (s0,c0) in starts:
        s,c = s0,c0
        chain=[]
        while True:
            chain.append({
                "chain_id":chain_id,
                "slot":s,
                "year":parse_year(s),
                "cluster":c,
                "volume":vol_lookup.get((s,c),0)
            })
            nxt=forward.get((s,c))
            if nxt is None: break
            s,c = nxt
        if len(chain)>=2:
            chains.extend(chain)
            chain_id+=1
    return pd.DataFrame(chains)



def plot_topic_trends(chains_df, top_k=5):
    agg = chains_df.groupby("chain_id")["volume"].sum().reset_index()
    agg = agg.sort_values("volume",ascending=False).head(top_k)
    top_ids = agg["chain_id"].tolist()

    plt.figure(figsize=(12,7))

    palette = sns.color_palette("tab10", n_colors=len(top_ids))

    for i, cid in enumerate(top_ids):
        sub = chains_df[chains_df["chain_id"]==cid].sort_values("year")
        y = sub["volume"].values
        y_smooth = gaussian_filter1d(y, sigma=1)   # smoothing

        plt.plot(
            sub["year"],
            y_smooth,
            label=f"Topic {cid}",
            linewidth=2.4,
            marker="o",
            color=palette[i]
        )

    plt.title("Topic Volume Trends (Smoothed)", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Volume (# papers)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "topic_trends_smoothed.png", dpi=350)
    plt.close()



def load_keywords(slot):
    p = KW_DIR / f"{slot}_keywords.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_topic_keywords(chains_df, chain_id):
    sub = chains_df[chains_df["chain_id"]==chain_id].sort_values("year")
    if sub.empty:
        return

    plt.figure(figsize=(12,0.7*len(sub)+2))
    plt.title(f"Topic {chain_id}: Keyword Evolution", fontsize=16)
    plt.axis("off")

    for i, row in enumerate(sub.iloc):
        kw = load_keywords(row.slot)
        cid = str(int(row.cluster))
        kws = kw.get(cid, [])[:10]
        txt = ", ".join(kws)

        plt.text(
            0.02,
            1-(i/(len(sub)+1)),
            f"{row.slot}  →  {txt}",
            fontsize=11,
            fontweight="medium"
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"topic_{chain_id}_keywords.png", dpi=350)
    plt.close()



def main():
    print("✓ Plotting semantic drift...")
    plot_semantic_drift()

    print("✓ Loading topic data...")
    vol, matching = load_data()

    print("✓ Building topic chains...")
    chains = build_chains(vol, matching)

    if chains.empty:
        print("[!] No chains found")
        return

    print("✓ Plotting smoothed topic trends...")
    plot_topic_trends(chains)

    agg = chains.groupby("chain_id")["volume"].sum().reset_index()
    top_topic = int(agg.sort_values("volume",ascending=False).iloc[0]["chain_id"])

    print(f"✓ Plotting keyword evolution for Topic {top_topic}...")
    plot_topic_keywords(chains, top_topic)

    print("✓ All enhanced visualizations saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
