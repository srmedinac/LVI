"""Generate attention violin plots and UMAP of patch embeddings."""

import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.manifold import TSNE

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.config import DEVICE, CHECKPOINT_DIR, RESULTS_DIR, EMORY_FEATS_PATH
from pipeline.models import SurvivalModel
from pipeline.train import set_seed

FILTERED_FEATS = "trident_outputs/emory_mibc/20x_512px_0px_overlap/filtered_conch_v15_LVI"
SAVE_PATH = os.path.join(RESULTS_DIR)

CUTPOINT_35PCT = -2.7268  # 35th percentile


def load_model():
    model = SurvivalModel().to(DEVICE)
    ckpt = os.path.join(CHECKPOINT_DIR, "joint_model_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def get_patient_attention(model, patient_id):
    """Get per-patch ABMIL attention for a patient."""
    h5_path = os.path.join(EMORY_FEATS_PATH, f"{patient_id}.h5")
    if not os.path.exists(h5_path):
        return None, None, None, None

    with h5py.File(h5_path, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32)

    with torch.no_grad():
        feat_tensor = features.unsqueeze(0).to(DEVICE)
        risk, lvi_logit, main_attn, _, _ = model(
            {"features": feat_tensor},
            return_raw_attention=True,
            return_patch_predictions=True,
        )

    attn_raw = main_attn.squeeze(0).squeeze(0)  # (n_heads, n_patches)
    attn_avg = attn_raw.mean(dim=0)
    abmil_attn = torch.softmax(attn_avg, dim=0).cpu().numpy()

    return (
        abmil_attn,
        features.numpy(),
        risk.item(),
        torch.sigmoid(lvi_logit).item(),
    )


def plot_attention_violins():
    """Violin plots of attention distributions by LVI status and risk group."""
    print("=== Attention Violin Plots ===")

    set_seed()
    model = load_model()

    # Load clinical + predictions
    clinical = pd.read_csv("clinical_merged_emory_final.csv")
    preds = pd.read_csv(os.path.join(SAVE_PATH, "joint_test_predictions.csv"))
    test = clinical[clinical["dataset"] == "test"].reset_index(drop=True)
    test["risk_score_new"] = preds["risk_score"].values
    test["lvi_prob_new"] = preds["lvi_prob"].values
    test["risk_group"] = np.where(
        test["risk_score_new"] > CUTPOINT_35PCT, "High Risk", "Low Risk"
    )
    test["lvi_label"] = np.where(test["LVI"] == "Present", "LVI+", "LVI-")

    # Collect per-patient mean/max attention
    records = []
    for _, row in test.iterrows():
        pid = row["patient_id"]
        attn, _, risk, lvi_prob = get_patient_attention(model, pid)
        if attn is None:
            continue
        records.append({
            "patient_id": pid,
            "mean_attn": np.mean(attn),
            "max_attn": np.max(attn),
            "attn_entropy": -np.sum(attn * np.log(attn + 1e-10)),
            "n_patches": len(attn),
            "risk_group": row["risk_group"],
            "lvi_label": row["lvi_label"],
            "risk_score": row["risk_score_new"],
        })

    df = pd.DataFrame(records)
    print(f"  Collected attention for {len(df)} patients")

    # --- Figure: 2x2 violins ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors_risk = {"Low Risk": "#00BFC4", "High Risk": "#F8766D"}
    colors_lvi = {"LVI-": "#00BFC4", "LVI+": "#F8766D"}

    # Top-left: Max attention by risk group
    for i, group in enumerate(["Low Risk", "High Risk"]):
        vals = df[df["risk_group"] == group]["max_attn"]
        vp = axes[0, 0].violinplot([vals], positions=[i], showmeans=True, showmedians=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(colors_risk[group])
            pc.set_alpha(0.7)
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(["Low Risk", "High Risk"], fontsize=13)
    axes[0, 0].set_ylabel("Max Attention Weight", fontsize=13)
    axes[0, 0].set_title("Max Attention by Risk Group", fontsize=14, fontweight="bold")

    # Top-right: Max attention by LVI status
    for i, group in enumerate(["LVI-", "LVI+"]):
        vals = df[df["lvi_label"] == group]["max_attn"]
        vp = axes[0, 1].violinplot([vals], positions=[i], showmeans=True, showmedians=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(colors_lvi[group])
            pc.set_alpha(0.7)
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["LVI-", "LVI+"], fontsize=13)
    axes[0, 1].set_ylabel("Max Attention Weight", fontsize=13)
    axes[0, 1].set_title("Max Attention by LVI Status", fontsize=14, fontweight="bold")

    # Bottom-left: Attention entropy by risk group
    for i, group in enumerate(["Low Risk", "High Risk"]):
        vals = df[df["risk_group"] == group]["attn_entropy"]
        vp = axes[1, 0].violinplot([vals], positions=[i], showmeans=True, showmedians=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(colors_risk[group])
            pc.set_alpha(0.7)
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Low Risk", "High Risk"], fontsize=13)
    axes[1, 0].set_ylabel("Attention Entropy", fontsize=13)
    axes[1, 0].set_title("Attention Entropy by Risk Group", fontsize=14, fontweight="bold")

    # Bottom-right: Number of vessel patches by risk/LVI
    for i, group in enumerate(["Low Risk", "High Risk"]):
        vals = df[df["risk_group"] == group]["n_patches"]
        vp = axes[1, 1].violinplot([vals], positions=[i], showmeans=True, showmedians=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(colors_risk[group])
            pc.set_alpha(0.7)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["Low Risk", "High Risk"], fontsize=13)
    axes[1, 1].set_ylabel("Number of Vessel Patches", fontsize=13)
    axes[1, 1].set_title("Vessel Patch Count by Risk Group", fontsize=14, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(SAVE_PATH, "attention_violins.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Print stats
    from scipy.stats import mannwhitneyu
    for metric in ["max_attn", "attn_entropy", "n_patches"]:
        high = df[df["risk_group"] == "High Risk"][metric]
        low = df[df["risk_group"] == "Low Risk"][metric]
        stat, p = mannwhitneyu(high, low, alternative="two-sided")
        print(f"  {metric} High vs Low Risk: p={p:.4f} (median {high.median():.4f} vs {low.median():.4f})")

    for metric in ["max_attn", "n_patches"]:
        pos = df[df["lvi_label"] == "LVI+"][metric]
        neg = df[df["lvi_label"] == "LVI-"][metric]
        stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
        print(f"  {metric} LVI+ vs LVI-: p={p:.4f} (median {pos.median():.4f} vs {neg.median():.4f})")


def plot_umap():
    """UMAP/t-SNE of patch embeddings colored by attention."""
    print("\n=== Patch Embedding t-SNE ===")

    set_seed()
    model = load_model()

    clinical = pd.read_csv("clinical_merged_emory_final.csv")
    preds = pd.read_csv(os.path.join(SAVE_PATH, "joint_test_predictions.csv"))
    test = clinical[clinical["dataset"] == "test"].reset_index(drop=True)
    test["risk_score_new"] = preds["risk_score"].values
    test["risk_group"] = np.where(
        test["risk_score_new"] > CUTPOINT_35PCT, "High Risk", "Low Risk"
    )
    test["lvi_label"] = np.where(test["LVI"] == "Present", "LVI+", "LVI-")

    # Collect patches from a subset of patients (too many patches otherwise)
    all_features = []
    all_attentions = []
    all_risk_groups = []
    all_lvi_labels = []

    # Sample ~20 patients per group for manageable t-SNE
    np.random.seed(42)
    for group in ["Low Risk", "High Risk"]:
        group_pids = test[test["risk_group"] == group]["patient_id"].values
        selected = np.random.choice(
            group_pids, size=min(20, len(group_pids)), replace=False
        )
        for pid in selected:
            attn, feats, _, _ = get_patient_attention(model, pid)
            if attn is None:
                continue
            lvi = test[test["patient_id"] == pid]["lvi_label"].values[0]
            all_features.append(feats)
            all_attentions.append(attn)
            all_risk_groups.extend([group] * len(attn))
            all_lvi_labels.extend([lvi] * len(attn))

    all_features = np.concatenate(all_features, axis=0)
    all_attentions = np.concatenate(all_attentions, axis=0)
    all_risk_groups = np.array(all_risk_groups)
    all_lvi_labels = np.array(all_lvi_labels)

    print(f"  Total patches: {len(all_features)}")
    print(f"  Risk groups: {dict(zip(*np.unique(all_risk_groups, return_counts=True)))}")
    print(f"  LVI labels: {dict(zip(*np.unique(all_lvi_labels, return_counts=True)))}")

    # Subsample if too many
    max_patches = 5000
    if len(all_features) > max_patches:
        idx = np.random.choice(len(all_features), max_patches, replace=False)
        all_features = all_features[idx]
        all_attentions = all_attentions[idx]
        all_risk_groups = all_risk_groups[idx]
        all_lvi_labels = all_lvi_labels[idx]
        print(f"  Subsampled to {max_patches} patches")

    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embedding = tsne.fit_transform(all_features)

    # --- Figure: 1x3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: colored by risk group
    for group, color in [("Low Risk", "#00BFC4"), ("High Risk", "#F8766D")]:
        mask = all_risk_groups == group
        axes[0].scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, s=3, alpha=0.4, label=group, rasterized=True,
        )
    axes[0].legend(fontsize=12, markerscale=5)
    axes[0].set_title("Colored by Risk Group", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("t-SNE 1", fontsize=12)
    axes[0].set_ylabel("t-SNE 2", fontsize=12)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: colored by LVI status
    for label, color in [("LVI-", "#00BFC4"), ("LVI+", "#F8766D")]:
        mask = all_lvi_labels == label
        axes[1].scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, s=3, alpha=0.4, label=label, rasterized=True,
        )
    axes[1].legend(fontsize=12, markerscale=5)
    axes[1].set_title("Colored by LVI Status", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("t-SNE 1", fontsize=12)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Panel 3: colored by attention weight
    # Log-transform attention for better visibility
    attn_log = np.log10(all_attentions + 1e-6)
    sc = axes[2].scatter(
        embedding[:, 0], embedding[:, 1],
        c=attn_log, cmap="inferno", s=3, alpha=0.5, rasterized=True,
    )
    cbar = plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("log10(Attention)", fontsize=11)
    axes[2].set_title("Colored by ABMIL Attention", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("t-SNE 1", fontsize=12)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.suptitle(
        f"t-SNE of Vessel-Adjacent Patch Embeddings (n={len(all_features)} patches)",
        fontsize=16, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    out_path = os.path.join(SAVE_PATH, "tsne_patches.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    plot_attention_violins()
    plot_umap()
    print("\nDone!")
