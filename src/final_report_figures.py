import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from wordcloud import WordCloud
from matplotlib.sankey import Sankey

# Paths
SLOT_DIR = Path("dataset/slots")
META_PATH = Path("dataset/faiss/slot_metadata.pkl")
VOL_PATH = Path("results/topic_trends/cluster_volume.csv")
MATCH_PATH = Path("results/topic_trends/cluster_matching.json")
DRIFT_PATH = Path("results/sbert_semantic_shift_words.csv")
KW_DIR = Path("results/topic_keywords")
OUT_DIR = Path("figures/final_report")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(META_PATH, "rb") as f:
    META = pickle.load(f)

def load_slot(slot):
    return pd.read_parquet(SLOT_DIR / f"slot_{slot}.parquet")


def load_embeddings(slot):
    return np.load(META[slot]["embeddings_path"], mmap_mode="r").astype("float32")


def parse_year(slot):
    s, e = slot.split("_")
    return (int(s) + int(e)) / 2


# -------------------------------------------------------------
#  UMAP Cluster Map
# -------------------------------------------------------------
def plot_umap(slot):
    df = load_slot(slot)
    embs = load_embeddings(slot)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embed2d = reducer.fit_transform(embs)

    df_plot = pd.DataFrame({
        "x": embed2d[:, 0],
        "y": embed2d[:, 1],
        "cluster": df["cluster"].tolist()
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_plot,
        x="x", y="y",
        hue="cluster",
        palette="tab20",
        s=8,
        alpha=0.7,
        legend=False
    )
    plt.title(f"UMAP Cluster Map — Slot {slot}", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"umap_slot_{slot}.png", dpi=300)
    plt.close()


# -------------------------------------------------------------
# Drift Plot 
# -------------------------------------------------------------
def plot_drift_trajectory(top_n=20):
    df = pd.read_csv(DRIFT_PATH)
    df = df.sort_values("combined_score", ascending=False).head(top_n)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=df,
        y="word",
        x="combined_score",
        palette="viridis"
    )
    plt.title(f"Top {top_n} Words by Semantic Drift (Combined Score)")
    plt.xlabel("Combined Drift Score")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "drift_barplot.png", dpi=300)
    plt.close()



# -------------------------------------------------------------
#  Keyword Overlap Heatmap
# -------------------------------------------------------------
def plot_keyword_overlap(slot):
    path = KW_DIR / f"{slot}_keywords.json"
    if not path.exists():
        print(f"No keyword file for {slot}")
        return

    with open(path, "r", encoding="utf-8") as f:
        kw = json.load(f)

    clusters = list(kw.keys())
    mat = np.zeros((len(clusters), len(clusters)))

    for i, ci in enumerate(clusters):
        set_i = set(kw[ci])
        for j, cj in enumerate(clusters):
            set_j = set(kw[cj])
            if len(set_i | set_j) == 0:
                mat[i, j] = 0
            else:
                mat[i, j] = len(set_i & set_j) / len(set_i | set_j)

    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=False, cmap="Blues")
    plt.title(f"Keyword Overlap Heatmap — Slot {slot}")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"keyword_overlap_{slot}.png", dpi=300)
    plt.close()


# -------------------------------------------------------------
#  Topic Chain Sankey Diagram
# -------------------------------------------------------------
def plot_sankey():
    vol_df = pd.read_csv(VOL_PATH)
    with open(MATCH_PATH, "r", encoding="utf-8") as f:
        match = json.load(f)

    flows = []
    labels = []
    for key, m in match.items():
        sA, sB = key.split("->")
        for cidA, info in m.items():
            if info["match"] is None:
                continue
            cA = f"{sA}:{cidA}"
            cB = f"{sB}:{info['match']}"
            flows.append((cA, cB))

    if not flows:
        return

    plt.figure(figsize=(10, 6))
    sankey = Sankey(unit=None)

    sankey.add(flows=[1] * len(flows), labels=[f"{a}→{b}" for a, b in flows])
    sankey.finish()

    plt.title("Topic Chain Sankey Diagram")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "topic_sankey.png", dpi=300)
    plt.close()


# -------------------------------------------------------------
# Semantic Drift Scatter Plot
# -------------------------------------------------------------
def plot_drift_scatter(top_n=200):
    df = pd.read_csv(DRIFT_PATH)

    has_cosine = "cosine_drift" in df.columns
    has_jaccard = "jaccard_drift" in df.columns

    if has_cosine and has_jaccard:
        x = df["cosine_drift"]
        y = df["jaccard_drift"]
        xlabel = "Cosine Drift"
        ylabel = "Jaccard Drift"
    elif has_cosine:
        x = df["cosine_drift"]
        y = df["combined_score"]
        xlabel = "Cosine Drift"
        ylabel = "Combined Drift"
    else:
        x = df["combined_score"]
        y = df["global_freq"]
        xlabel = "Combined Drift"
        ylabel = "Global Frequency"

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Drift Scatter Plot")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "drift_scatter.png", dpi=300)
    plt.close()



# -------------------------------------------------------------
# FAISS Neighbor Density Plot
# -------------------------------------------------------------
def plot_neighbor_density():
    df = pd.read_csv(DRIFT_PATH)
    freq = df["frequency"]

    plt.figure(figsize=(7, 5))
    sns.histplot(freq, bins=30, kde=True)
    plt.title("FAISS Neighbor Density")
    plt.xlabel("Neighbor Count")
    plt.ylabel("Frequency (Words)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "faiss_neighbor_density.png", dpi=300)
    plt.close()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    print("[1] UMAP")
    sample_slot = sorted(META.keys())[len(META) // 2]  
    plot_umap(sample_slot)

    print("[2] Drift Trajectory")
    plot_drift_trajectory()

    print("[3] Keyword Overlap")
    plot_keyword_overlap(sample_slot)

    print("[4] Sankey")
    plot_sankey()

    print("[5] Drift Scatter")
    plot_drift_scatter()

    print("[6] Neighbor Density")
    plot_neighbor_density()

    print(f"[✓] All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
