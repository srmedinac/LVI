"""Expanded vascular RNA analysis: angiogenesis + lymphangiogenesis.

Correlates AI risk scores with angiogenesis and lymphangiogenesis gene
expression from TCGA bulk RNA-seq.

Usage:
    python results_2026/results/rna_vascular_analysis.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

SAVE_PATH = "results_2026/results"
RNA_PATH = "/home/smedin7/Documents/BLCA-PCA-SEAL/tcga_blca_bulk_tpm.csv"
CUTPOINT = -2.7603  # Training set 35th percentile

# ── Gene sets ────────────────────────────────────────────────────────────────

LYMPHANGIOGENESIS_GENES = {
    "VEGFC": "VEGF-C (lymphatic growth factor)",
    "VEGFD": "VEGF-D (lymphatic growth factor)",
    "FLT4": "VEGFR-3 (lymphatic receptor)",
    "PROX1": "Prox1 (lymphatic TF)",
    "LYVE1": "LYVE-1 (lymphatic marker)",
    "PDPN": "Podoplanin (lymphatic marker)",
    "SOX18": "SOX18 (lymphatic TF)",
    "FOXC2": "FOXC2 (lymphatic valve TF)",
    "CCBE1": "CCBE1 (VEGF-C processing)",
    "ADAMTS3": "ADAMTS3 (VEGF-C processing)",
    "NRP2": "Neuropilin-2 (lymphatic co-receptor)",
}

ANGIOGENESIS_GENES = {
    "VEGFA": "VEGF-A (angiogenic growth factor)",
    "VEGFB": "VEGF-B",
    "KDR": "VEGFR-2 (angiogenic receptor)",
    "FLT1": "VEGFR-1",
    "PGF": "PlGF (placental growth factor)",
    "PECAM1": "CD31 (endothelial marker)",
    "CD34": "CD34 (endothelial progenitor)",
    "ENG": "Endoglin (TGF-β co-receptor)",
    "VWF": "von Willebrand Factor",
    "CDH5": "VE-Cadherin (endothelial junction)",
    "ERG": "ERG (endothelial TF)",
    "ANGPT1": "Angiopoietin-1 (vessel stability)",
    "ANGPT2": "Angiopoietin-2 (vessel destabilization)",
    "TEK": "Tie2 (angiopoietin receptor)",
    "HIF1A": "HIF-1α (hypoxia response)",
    "PDGFB": "PDGF-B (pericyte recruitment)",
    "PDGFRB": "PDGFRβ (pericyte marker)",
    "NRP1": "Neuropilin-1 (VEGF co-receptor)",
    "DLL4": "DLL4 (Notch ligand, tip cells)",
    "NOTCH1": "Notch1 (stalk cell specification)",
}

# Combined for the gene-level heatmap
ALL_GENES = {}
ALL_GENES.update({k: ("Lymphangiogenesis", v) for k, v in LYMPHANGIOGENESIS_GENES.items()})
ALL_GENES.update({k: ("Angiogenesis", v) for k, v in ANGIOGENESIS_GENES.items()})


def load_data():
    preds = pd.read_csv(os.path.join(SAVE_PATH, "tcga_predictions.csv"))
    rna = pd.read_csv(RNA_PATH)
    merged = preds.merge(rna, on="patient_id", how="inner")
    merged["risk_group"] = np.where(
        merged["risk_score"] > CUTPOINT, "High Risk", "Low Risk"
    )
    n_high = (merged["risk_group"] == "High Risk").sum()
    n_low = (merged["risk_group"] == "Low Risk").sum()
    print(f"Merged: {len(merged)} patients (High Risk: {n_high}, Low Risk: {n_low})")
    return merged


def compute_score(df, genes):
    available = [g for g in genes if g in df.columns]
    z = df[available].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return z.mean(axis=1), available


def gene_level_tests(df, gene_dict):
    """Mann-Whitney test for each gene, High vs Low Risk. Returns sorted DataFrame."""
    results = []
    for gene, (pathway, desc) in gene_dict.items():
        if gene not in df.columns:
            continue
        high = df.loc[df["risk_group"] == "High Risk", gene]
        low = df.loc[df["risk_group"] == "Low Risk", gene]
        stat, p = mannwhitneyu(high, low, alternative="two-sided")
        # Log2 fold change of medians
        log2fc = np.log2((high.median() + 1) / (low.median() + 1))
        rho, p_rho = spearmanr(df["risk_score"], df[gene])
        results.append({
            "gene": gene,
            "pathway": pathway,
            "description": desc,
            "median_high": high.median(),
            "median_low": low.median(),
            "log2fc": log2fc,
            "p_value": p,
            "spearman_rho": rho,
            "spearman_p": p_rho,
        })
    res = pd.DataFrame(results)
    # FDR correction
    _, fdr, _, _ = multipletests(res["p_value"], method="fdr_bh")
    res["fdr"] = fdr
    res = res.sort_values("p_value")
    return res


def _style_boxplot(ax, data_low, data_high, ylabel, title, label_a="Low Risk", label_b="High Risk"):
    """Shared boxplot styling for presentation quality."""
    bp = ax.boxplot(
        [data_low, data_high],
        tick_labels=[label_a, label_b],
        patch_artist=True, widths=0.55,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    bp["boxes"][0].set_facecolor("#00BFC4"); bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor("#F8766D"); bp["boxes"][1].set_alpha(0.75)
    for box in bp["boxes"]:
        box.set_linewidth(1.3)

    # Overlay strip points
    for i, (d, color) in enumerate([(data_low, "#00BFC4"), (data_high, "#F8766D")]):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, s=8, alpha=0.3,
                   color=color, edgecolors="none", zorder=3)

    _, p = mannwhitneyu(data_high, data_low, alternative="two-sided")
    p_str = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.tick_params(labelsize=12)
    ax.text(0.5, 0.96, p_str, transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return p


def plot_figure(df, gene_results):
    """Generate presentation-quality gene-level figure."""
    from matplotlib.patches import Patch

    # ── Compute scores (stored for print_summary) ────────────────────────────
    angio_score, _ = compute_score(df, list(ANGIOGENESIS_GENES.keys()))
    lymph_score, _ = compute_score(df, list(LYMPHANGIOGENESIS_GENES.keys()))
    df["angio_score"] = angio_score
    df["lymph_score"] = lymph_score

    high_mask = df["risk_group"] == "High Risk"
    low_mask = df["risk_group"] == "Low Risk"

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    gs = gridspec.GridSpec(1, 2, wspace=0.45, width_ratios=[2, 1.2])

    # ── Panel A: Gene-level log2FC bar chart (all 31 genes) ──────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    lymph_res = gene_results[gene_results["pathway"] == "Lymphangiogenesis"].copy()
    angio_res = gene_results[gene_results["pathway"] == "Angiogenesis"].copy()
    ordered = pd.concat([lymph_res, angio_res])

    genes = ordered["gene"].values
    log2fc = ordered["log2fc"].values
    pvals = ordered["p_value"].values
    fdr = ordered["fdr"].values
    pathways = ordered["pathway"].values

    y_pos = np.arange(len(genes))
    colors = ["#56B4E9" if p == "Lymphangiogenesis" else "#E69F00" for p in pathways]
    ax_a.barh(y_pos, log2fc, color=colors, alpha=0.85, height=0.72,
              edgecolor="white", linewidth=0.5)

    for i, (fc, p, q) in enumerate(zip(log2fc, pvals, fdr)):
        if q < 0.05:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        else:
            marker = ""
        if marker:
            x_pos = fc + 0.015 if fc >= 0 else fc - 0.015
            ha = "left" if fc >= 0 else "right"
            ax_a.text(x_pos, i, marker, ha=ha, va="center", fontsize=13,
                      fontweight="bold", color="#333333")

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(genes, fontsize=12, fontfamily="monospace")
    ax_a.set_xlabel("Log2 Fold Change (High Risk / Low Risk)", fontsize=14, fontweight="bold")
    ax_a.set_title("A. Gene-Level Expression: High vs Low Risk",
                    fontsize=16, fontweight="bold", pad=12)
    ax_a.axvline(0, color="black", linewidth=1)
    ax_a.invert_yaxis()
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.tick_params(labelsize=12)

    # Pathway separator
    n_lymph = len(lymph_res)
    if n_lymph > 0 and len(angio_res) > 0:
        ax_a.axhline(n_lymph - 0.5, color="gray", linewidth=1.2, linestyle="--", alpha=0.6)

    ax_a.legend(
        [Patch(facecolor="#56B4E9", alpha=0.85), Patch(facecolor="#E69F00", alpha=0.85)],
        ["Lymphangiogenesis", "Angiogenesis"],
        fontsize=12, loc="lower right", framealpha=0.9,
    )
    ax_a.text(0.99, 0.01, "* p < 0.05    ** FDR < 0.05",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=11, fontstyle="italic", color="gray")

    # ── Panel B: Top significant genes boxplots ──────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    top_genes = gene_results.head(6)
    box_data_low = []
    box_data_high = []
    labels = []

    for row in top_genes.itertuples():
        gene = row.gene
        vals = np.log2(df[gene] + 1)
        mu, sd = vals.mean(), vals.std()
        box_data_low.append((vals[low_mask] - mu) / (sd + 1e-8))
        box_data_high.append((vals[high_mask] - mu) / (sd + 1e-8))
        labels.append(gene)

    x = np.arange(len(labels))
    width = 0.35

    bp_low = ax_b.boxplot(
        box_data_low, positions=x - width / 2, widths=width * 0.8,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
    )
    bp_high = ax_b.boxplot(
        box_data_high, positions=x + width / 2, widths=width * 0.8,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
    )
    for box in bp_low["boxes"]:
        box.set_facecolor("#00BFC4"); box.set_alpha(0.75); box.set_linewidth(1.3)
    for box in bp_high["boxes"]:
        box.set_facecolor("#F8766D"); box.set_alpha(0.75); box.set_linewidth(1.3)

    fdr_lookup = dict(zip(gene_results["gene"], gene_results["fdr"]))
    for i, row in enumerate(top_genes.itertuples()):
        star = "**" if row.fdr < 0.05 else "*" if row.p_value < 0.05 else ""
        if star:
            ymax = max(box_data_low[i].quantile(0.75), box_data_high[i].quantile(0.75))
            ax_b.text(i, ymax + 0.35, star, ha="center", va="bottom",
                      fontsize=13, fontweight="bold", color="#333333")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=12, fontweight="bold", rotation=45, ha="right")
    ax_b.set_ylabel("Expression (z-scored)", fontsize=14, fontweight="bold")
    ax_b.set_title("B. Top Differentially Expressed\n     Genes", fontsize=16, fontweight="bold", pad=12)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(labelsize=12)

    ax_b.legend(
        [Patch(facecolor="#00BFC4", alpha=0.75), Patch(facecolor="#F8766D", alpha=0.75)],
        ["Low Risk", "High Risk"], fontsize=11, loc="upper right", framealpha=0.9,
    )

    out_path = os.path.join(SAVE_PATH, "rna_vascular_analysis_tcga.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


def plot_vegfc_figure(df, gene_results):
    """Standalone VEGFC-focused figure for presentation."""
    from matplotlib.patches import Patch

    high_mask = df["risk_group"] == "High Risk"
    low_mask = df["risk_group"] == "Low Risk"

    fig = plt.figure(figsize=(18, 7), facecolor="white")
    gs = gridspec.GridSpec(1, 3, wspace=0.4)

    # ── Panel A: VEGFC expression by risk group (boxplot + strip) ────────────
    ax_a = fig.add_subplot(gs[0, 0])
    vegfc_low = np.log2(df.loc[low_mask, "VEGFC"] + 1)
    vegfc_high = np.log2(df.loc[high_mask, "VEGFC"] + 1)
    _style_boxplot(ax_a, vegfc_low, vegfc_high,
                   "VEGFC Expression\n(log2 TPM+1)", "A. VEGFC by AI Risk Group")

    # ── Panel B: VEGFC vs risk score scatter ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    vegfc_log = np.log2(df["VEGFC"] + 1)
    rho, p_rho = spearmanr(df["risk_score"], vegfc_log)

    colors = ["#F8766D" if h else "#00BFC4" for h in high_mask]
    ax_b.scatter(df["risk_score"], vegfc_log, c=colors, s=20, alpha=0.5, edgecolors="none")

    # Regression line
    z = np.polyfit(df["risk_score"], vegfc_log, 1)
    x_line = np.linspace(df["risk_score"].min(), df["risk_score"].max(), 100)
    ax_b.plot(x_line, np.polyval(z, x_line), color="#333333", linewidth=2, linestyle="--", alpha=0.7)

    ax_b.set_xlabel("AI Risk Score", fontsize=13, fontweight="bold")
    ax_b.set_ylabel("VEGFC Expression\n(log2 TPM+1)", fontsize=13, fontweight="bold")
    ax_b.set_title("B. VEGFC vs AI Risk Score", fontsize=15, fontweight="bold", pad=10)
    ax_b.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {p_rho:.4f}",
              transform=ax_b.transAxes, ha="left", va="top", fontsize=12,
              fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(labelsize=11)

    # ── Panel C: Key lymphangiogenesis genes boxplots ────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    key_genes = ["VEGFC", "PDPN", "ADAMTS3", "NRP2", "VEGFD", "LYVE1"]
    box_data_low = []
    box_data_high = []
    gene_pvals = []

    for gene in key_genes:
        vals = np.log2(df[gene] + 1)
        mu, sd = vals.mean(), vals.std()
        z_low = (vals[low_mask] - mu) / (sd + 1e-8)
        z_high = (vals[high_mask] - mu) / (sd + 1e-8)
        box_data_low.append(z_low)
        box_data_high.append(z_high)
        _, p = mannwhitneyu(vals[high_mask], vals[low_mask], alternative="two-sided")
        gene_pvals.append(p)

    # Look up FDR from gene_results
    fdr_lookup = dict(zip(gene_results["gene"], gene_results["fdr"]))

    x = np.arange(len(key_genes))
    width = 0.35

    bp_low = ax_c.boxplot(
        box_data_low, positions=x - width / 2, widths=width * 0.8,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    bp_high = ax_c.boxplot(
        box_data_high, positions=x + width / 2, widths=width * 0.8,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for box in bp_low["boxes"]:
        box.set_facecolor("#00BFC4"); box.set_alpha(0.75); box.set_linewidth(1.3)
    for box in bp_high["boxes"]:
        box.set_facecolor("#F8766D"); box.set_alpha(0.75); box.set_linewidth(1.3)

    for i, (gene, p) in enumerate(zip(key_genes, gene_pvals)):
        fdr_val = fdr_lookup.get(gene, 1.0)
        star = "**" if fdr_val < 0.05 else "*" if p < 0.05 else ""
        if star:
            ymax = max(box_data_low[i].quantile(0.75), box_data_high[i].quantile(0.75))
            ax_c.text(i, ymax + 0.35, star, ha="center", va="bottom",
                      fontsize=13, fontweight="bold", color="#333333")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(key_genes, fontsize=12, fontweight="bold")
    ax_c.set_ylabel("Expression (z-scored)", fontsize=13, fontweight="bold")
    ax_c.set_title("C. Lymphangiogenesis Genes", fontsize=15, fontweight="bold", pad=10)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.tick_params(labelsize=11)

    ax_c.legend(
        [Patch(facecolor="#00BFC4", alpha=0.75), Patch(facecolor="#F8766D", alpha=0.75)],
        ["Low Risk", "High Risk"], fontsize=11, loc="upper right", framealpha=0.9,
    )
    ax_c.text(0.99, 0.01, "* p < 0.05    ** FDR < 0.05",
              transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=10, fontstyle="italic", color="gray")

    out_path = os.path.join(SAVE_PATH, "rna_vegfc_lymphangiogenesis.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"VEGFC figure saved: {out_path}")


def print_summary(df, gene_results):
    """Print detailed summary table."""
    print("\n" + "=" * 80)
    print("VASCULAR RNA ANALYSIS SUMMARY")
    print("=" * 80)

    # Pathway scores
    for name, score_col in [("Angiogenesis", "angio_score"), ("Lymphangiogenesis", "lymph_score")]:
        high = df.loc[df["risk_group"] == "High Risk", score_col]
        low = df.loc[df["risk_group"] == "Low Risk", score_col]
        _, p = mannwhitneyu(high, low)
        rho, p_rho = spearmanr(df["risk_score"], df[score_col])
        print(f"\n{name} Score:")
        print(f"  High Risk median: {high.median():.4f}  |  Low Risk median: {low.median():.4f}  |  p = {p:.4f}")
        print(f"  Spearman with risk score: rho = {rho:.3f}, p = {p_rho:.4f}")

    # Gene-level table
    print(f"\n{'Gene':<12} {'Pathway':<20} {'log2FC':>8} {'p-value':>10} {'FDR':>10} {'Spearman ρ':>12}")
    print("-" * 76)
    for row in gene_results.itertuples():
        sig = "**" if row.fdr < 0.05 else " *" if row.p_value < 0.05 else "  "
        print(f"{row.gene:<12} {row.pathway:<20} {row.log2fc:>8.4f} {row.p_value:>10.4f} {row.fdr:>10.4f} {row.spearman_rho:>11.3f} {sig}")

    # Count significant
    n_sig = (gene_results["p_value"] < 0.05).sum()
    n_fdr = (gene_results["fdr"] < 0.05).sum()
    print(f"\nSignificant: {n_sig}/{len(gene_results)} at p<0.05, {n_fdr}/{len(gene_results)} at FDR<0.05")


if __name__ == "__main__":
    df = load_data()
    gene_results = gene_level_tests(df, ALL_GENES)

    # Save gene results table
    gene_results.to_csv(os.path.join(SAVE_PATH, "rna_gene_level_results.csv"), index=False)
    print(f"Gene-level results saved: {SAVE_PATH}/rna_gene_level_results.csv")

    plot_figure(df, gene_results)
    plot_vegfc_figure(df, gene_results)
    print_summary(df, gene_results)
    print("\nDone!")
