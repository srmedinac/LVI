"""TCGA external validation inference.

Usage:
    python -m pipeline.inference_tcga --clinical /path/to/tcga_clinical.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from tqdm import tqdm

from pipeline.config import DEVICE, SEED, CHECKPOINT_DIR, RESULTS_DIR, TCGA_FEATS_PATH, TCGA_FILTERED_FEATS_PATH
from pipeline.models import MODELS, CHECKPOINT_NAMES
from pipeline.train import set_seed


def load_tcga_clinical(csv_path):
    """Load TCGA clinical data. Expects columns: patient_id, os_event, os_time.

    If the CSV has TCGA barcode columns, we extract the patient-level ID
    (first 12 characters, e.g. TCGA-XX-XXXX).
    """
    df = pd.read_csv(csv_path)

    # Find the slide/barcode column if patient_id not present
    if "patient_id" not in df.columns:
        for col in ("bcr_patient_barcode", "submitter_id", "case_id"):
            if col in df.columns:
                df["patient_id"] = df[col].str[:12]
                break

    return df


def find_tcga_h5_files(df, feats_path=TCGA_FILTERED_FEATS_PATH):
    """Match patient IDs to available H5 feature files.

    TCGA H5 files may be named by slide ID (longer barcode) or patient ID.
    We check for both and filter to patients with available features.
    """
    available = set()
    if os.path.isdir(feats_path):
        for f in os.listdir(feats_path):
            if f.endswith(".h5"):
                available.add(f.replace(".h5", ""))

    matched = []
    for _, row in df.iterrows():
        pid = row["patient_id"]
        # Try exact match first, then prefix match (slide-level files)
        if pid in available:
            matched.append(row)
        else:
            prefix_matches = [a for a in available if a.startswith(pid)]
            if prefix_matches:
                # Use the first DX slide if available
                dx_matches = [m for m in prefix_matches if "-DX" in m]
                slide_id = dx_matches[0] if dx_matches else prefix_matches[0]
                row_copy = row.copy()
                row_copy["slide_id"] = slide_id
                matched.append(row_copy)

    result = pd.DataFrame(matched)
    print(f"Matched {len(result)}/{len(df)} patients to H5 files")
    return result


def run_tcga_inference(tcga_df, feats_path=TCGA_FILTERED_FEATS_PATH):
    """Run all models on TCGA data and return results DataFrame."""
    set_seed()

    all_rows = []

    for model_name in ("Joint", "Survival-Only", "LVI-Only"):
        ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAMES[model_name])
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found for {model_name}, skipping.")
            continue

        model_cls = MODELS[model_name]
        model = model_cls().to(DEVICE)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        )
        model.eval()

        print(f"\nRunning {model_name} on TCGA ({len(tcga_df)} patients)...")

        with torch.no_grad():
            for _, row in tqdm(tcga_df.iterrows(), total=len(tcga_df)):
                # Determine which file to load
                slide_key = row.get("slide_id", row["patient_id"])
                h5_path = os.path.join(feats_path, slide_key + ".h5")

                import h5py
                with h5py.File(h5_path, "r") as f:
                    features = torch.from_numpy(f["features"][:]).unsqueeze(0).to(DEVICE)

                risk_score, lvi_logit = model({"features": features})

                all_rows.append({
                    "patient_id": row["patient_id"],
                    "model": model_name,
                    "risk_score": risk_score.item(),
                    "lvi_prob": torch.sigmoid(lvi_logit).item(),
                    "os_event": row.get("os_event", np.nan),
                    "os_time": row.get("os_time", np.nan),
                })

    results = pd.DataFrame(all_rows)
    return results


def plot_tcga_km_curves(results_df, risk_threshold=None, output_dir=RESULTS_DIR):
    """Plot KM curves for TCGA external validation per model."""
    os.makedirs(output_dir, exist_ok=True)

    for model_name, group in results_df.groupby("model"):
        risks = group["risk_score"].values
        events = group["os_event"].values.astype(bool)
        times = group["os_time"].values / 365.25

        threshold = risk_threshold if risk_threshold is not None else np.median(risks)
        high_risk = risks > threshold

        fig, ax = plt.subplots(figsize=(8, 6))
        kmf_low = KaplanMeierFitter()
        kmf_high = KaplanMeierFitter()

        kmf_low.fit(times[~high_risk], events[~high_risk], label="Low Risk")
        kmf_high.fit(times[high_risk], events[high_risk], label="High Risk")

        kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#00BFC4")
        kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#F8766D")

        lr = logrank_test(
            times[~high_risk], times[high_risk],
            events[~high_risk], events[high_risk],
        )

        c_idx = concordance_index(times, -risks, events)
        ax.set_title(
            f"TCGA - {model_name} (C-index={c_idx:.3f}, p={lr.p_value:.4f})"
        )
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Overall Survival")
        ax.legend()

        fname = f"km_tcga_{model_name.lower().replace('-', '_')}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="TCGA external validation")
    parser.add_argument("--clinical", required=True, help="Path to TCGA clinical CSV")
    parser.add_argument(
        "--feats", default=TCGA_FILTERED_FEATS_PATH,
        help="Path to TCGA H5 features (default: filtered; use TCGA_FEATS_PATH for unfiltered)"
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="Risk score threshold for KM (default: median)")
    args = parser.parse_args()

    tcga_df = load_tcga_clinical(args.clinical)
    tcga_df = find_tcga_h5_files(tcga_df, args.feats)

    results = run_tcga_inference(tcga_df, args.feats)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "tcga_inference_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # KM curves
    valid = results.dropna(subset=["os_event", "os_time"])
    if len(valid) > 0:
        plot_tcga_km_curves(valid, risk_threshold=args.threshold)


if __name__ == "__main__":
    main()
