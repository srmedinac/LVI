"""Correlate AI risk scores with angiogenesis gene expression (TCGA).

Run AFTER TCGA inference is complete.

Usage:
    python results_2026/results/rna_angiogenesis.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

SAVE_PATH = "results_2026/results"
RNA_PATH = "/home/smedin7/Documents/BLCA-PCA-SEAL/tcga_blca_bulk_tpm.csv"

# Hallmark angiogenesis genes (MSigDB)
ANGIOGENESIS_GENES = [
    "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "FLT4",
    "PECAM1", "CD34", "ENG", "ANGPT1", "ANGPT2", "TEK",
    "HIF1A", "PDGFB", "PDGFRB", "NRP1", "NRP2",
    "THBS1", "THBS2", "PGF", "CXCL8", "CXCR2",
    "MMP2", "MMP9", "COL4A1", "COL4A2", "ITGAV", "ITGB3",
]

# Endothelial / vascular markers
VASCULAR_MARKERS = ["PECAM1", "CD34", "ENG", "KDR", "FLT1", "VWF", "CDH5", "ERG"]


def load_data():
    """Load TCGA predictions + RNA + merge."""
    # Load TCGA predictions (will exist after inference)
    pred_file = os.path.join(SAVE_PATH, "tcga_predictions.csv")
    if not os.path.exists(pred_file):
        print(f"ERROR: {pred_file} not found. Run TCGA inference first.")
        sys.exit(1)

    preds = pd.read_csv(pred_file)
    print(f"TCGA predictions: {len(preds)} patients")

    # Load RNA
    rna = pd.read_csv(RNA_PATH)
    print(f"Bulk RNA: {len(rna)} patients, {len(rna.columns)-1} genes")

    # Merge
    merged = preds.merge(rna, on="patient_id", how="inner")
    print(f"Merged: {len(merged)} patients")

    return merged


def compute_angiogenesis_score(df, genes):
    """Compute mean z-scored expression across angiogenesis genes."""
    available = [g for g in genes if g in df.columns]
    missing = [g for g in genes if g not in df.columns]
    if missing:
        print(f"  Missing genes: {missing}")

    # Z-score each gene, then average
    z_scores = df[available].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return z_scores.mean(axis=1), available


def plot_angiogenesis_analysis(df, cutpoint_pct=0.35):
    """Generate angiogenesis correlation figures."""

    # Risk stratification
    cutpoint = np.quantile(df["risk_score"], cutpoint_pct)
    df["risk_group"] = np.where(df["risk_score"] > cutpoint, "High Risk", "Low Risk")

    # Compute angiogenesis score
    df["angio_score"], angio_genes_used = compute_angiogenesis_score(df, ANGIOGENESIS_GENES)
    print(f"  Using {len(angio_genes_used)} angiogenesis genes")

    # --- Figure 1: Angiogenesis score by risk group (boxplot) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    high_angio = df[df["risk_group"] == "High Risk"]["angio_score"]
    low_angio = df[df["risk_group"] == "Low Risk"]["angio_score"]
    stat, p = mannwhitneyu(high_angio, low_angio, alternative="two-sided")

    bp = axes[0].boxplot(
        [low_angio, high_angio],
        labels=["Low Risk", "High Risk"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor("#00BFC4")
    bp["boxes"][1].set_facecolor("#F8766D")
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    axes[0].set_ylabel("Angiogenesis Score (z-scored)", fontsize=13)
    axes[0].set_title("Angiogenesis Score by AI Risk Group", fontsize=14, fontweight="bold")
    axes[0].text(
        0.5, 0.95, f"p = {p:.4f}" if p >= 0.0001 else f"p < 0.0001",
        transform=axes[0].transAxes, ha="center", va="top", fontsize=13,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # --- Figure 2: Scatter risk score vs angiogenesis ---
    rho, p_spear = spearmanr(df["risk_score"], df["angio_score"])
    colors = ["#F8766D" if g == "High Risk" else "#00BFC4" for g in df["risk_group"]]
    axes[1].scatter(df["risk_score"], df["angio_score"], c=colors, s=15, alpha=0.5)
    axes[1].set_xlabel("AI Risk Score", fontsize=13)
    axes[1].set_ylabel("Angiogenesis Score", fontsize=13)
    axes[1].set_title("Risk Score vs Angiogenesis", fontsize=14, fontweight="bold")
    axes[1].text(
        0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {p_spear:.4f}",
        transform=axes[1].transAxes, ha="left", va="top", fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # --- Figure 3: Individual vascular marker boxplots ---
    markers_available = [g for g in VASCULAR_MARKERS if g in df.columns]
    p_values = []
    fold_changes = []
    for gene in markers_available:
        high_vals = df[df["risk_group"] == "High Risk"][gene]
        low_vals = df[df["risk_group"] == "Low Risk"][gene]
        _, p_val = mannwhitneyu(high_vals, low_vals, alternative="two-sided")
        fc = high_vals.median() / (low_vals.median() + 1e-8)
        p_values.append(p_val)
        fold_changes.append(fc)

    # Bar plot of -log10(p) for each marker
    neg_log_p = [-np.log10(p) for p in p_values]
    bar_colors = ["#F8766D" if p < 0.05 else "#999999" for p in p_values]
    axes[2].barh(markers_available, neg_log_p, color=bar_colors, alpha=0.8)
    axes[2].axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1, label="p=0.05")
    axes[2].set_xlabel("-log10(p-value)", fontsize=13)
    axes[2].set_title("Vascular Markers: High vs Low Risk", fontsize=14, fontweight="bold")
    axes[2].legend(fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(SAVE_PATH, "rna_angiogenesis_tcga.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Print summary
    print(f"\n  Angiogenesis score: High Risk median={high_angio.median():.3f} vs Low Risk median={low_angio.median():.3f}, p={p:.4f}")
    print(f"  Spearman correlation: rho={rho:.3f}, p={p_spear:.4f}")
    print(f"\n  Individual vascular markers (High vs Low Risk):")
    for gene, pv, fc in zip(markers_available, p_values, fold_changes):
        sig = "*" if pv < 0.05 else ""
        print(f"    {gene:10s}: p={pv:.4f} FC={fc:.2f} {sig}")


if __name__ == "__main__":
    df = load_data()
    plot_angiogenesis_analysis(df)
    print("\nDone!")
