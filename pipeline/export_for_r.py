"""Export prediction results for R survival analysis.

Usage:
    python -m pipeline.export_for_r
    python -m pipeline.export_for_r --emory_clinical /path/to/Emory_Bladder_case_cleaned.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pipeline.config import (
    DEVICE, SEED, CHECKPOINT_DIR, RESULTS_DIR, CLINICAL_CSV,
)
from pipeline.dataset import load_clinical_data, create_dataloaders
from pipeline.models import MODELS, CHECKPOINT_NAMES
from pipeline.train import set_seed


def export_km_data(output_dir=RESULTS_DIR):
    """Run all models on Emory data and export per-patient results for R.

    Produces survival_data_for_R.csv with columns:
        patient_id, dataset, risk_score, lvi_prob, lvi_true, event, time, risk_group
    """
    set_seed()
    df = load_clinical_data()
    train_loader, val_loader, test_loader = create_dataloaders(df)

    # Use the Joint model for the primary export
    model_name = "Joint"
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAMES[model_name])
    model_cls = MODELS[model_name]
    model = model_cls().to(DEVICE)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    rows = []
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    for split_name, loader in loaders.items():
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        idx = 0

        with torch.no_grad():
            for features, events, times, lvi in tqdm(loader, desc=split_name):
                features_d = {"features": features.to(DEVICE)}
                risk_scores, lvi_logits = model(features_d)

                for i in range(features.shape[0]):
                    rows.append({
                        "patient_id": split_df.iloc[idx]["patient_id"],
                        "dataset": split_name,
                        "risk_score": risk_scores[i].item(),
                        "lvi_prob": torch.sigmoid(lvi_logits[i]).item(),
                        "lvi_true": int(lvi[i].item()),
                        "event": int(events[i].item()),
                        "time": times[i].item(),
                    })
                    idx += 1

    result = pd.DataFrame(rows)

    # Compute risk group based on training set median
    train_median = result[result["dataset"] == "train"]["risk_score"].median()
    result["risk_group"] = (result["risk_score"] > train_median).astype(int)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "survival_data_for_R.csv")
    result.to_csv(out_path, index=False)
    print(f"Exported {len(result)} rows to {out_path}")
    print(f"Risk threshold (train median): {train_median:.4f}")

    return result, train_median


def export_clinical_merged(
    emory_clinical_csv,
    predictions_df=None,
    output_dir=RESULTS_DIR,
):
    """Merge Emory clinical data with model predictions for MVA.

    Args:
        emory_clinical_csv: Path to Emory_Bladder_case_cleaned.csv with
                            columns: OpS PPID, pT, pN, Age, Sex, Race, LVI, NAC
        predictions_df: DataFrame from export_km_data. If None, reads from results/.
    """
    clinical = pd.read_csv(emory_clinical_csv)

    if predictions_df is None:
        pred_path = os.path.join(output_dir, "survival_data_for_R.csv")
        predictions_df = pd.read_csv(pred_path)

    cols = ["OpS PPID", "pT", "pN", "Age", "Sex", "Race", "LVI", "NAC"]
    available = [c for c in cols if c in clinical.columns]
    clinical_subset = clinical[available]

    merged = predictions_df.merge(
        clinical_subset,
        left_on="patient_id",
        right_on="OpS PPID",
        how="left",
    )
    if "OpS PPID" in merged.columns:
        merged.drop("OpS PPID", axis=1, inplace=True)

    out_path = os.path.join(output_dir, "clinical_merged_emory_final.csv")
    merged.to_csv(out_path, index=False)
    print(f"Merged clinical data saved to {out_path}")


def export_tcga_merged(
    tcga_clinical_csv,
    tcga_results_csv=None,
    output_dir=RESULTS_DIR,
):
    """Merge TCGA clinical data with inference results."""
    clinical = pd.read_csv(tcga_clinical_csv)

    if tcga_results_csv is None:
        tcga_results_csv = os.path.join(output_dir, "tcga_inference_results.csv")
    predictions = pd.read_csv(tcga_results_csv)

    cols = [
        "bcr_patient_barcode",
        "age_at_initial_pathologic_diagnosis",
        "gender",
        "race",
        "ajcc_pathologic_tumor_stage",
    ]
    available = [c for c in cols if c in clinical.columns]
    clinical_subset = clinical[available]

    merged = predictions.merge(
        clinical_subset,
        left_on="patient_id",
        right_on="bcr_patient_barcode",
        how="left",
    )
    if "bcr_patient_barcode" in merged.columns:
        merged.drop("bcr_patient_barcode", axis=1, inplace=True)

    out_path = os.path.join(output_dir, "clinical_merged_tcga_final.csv")
    merged.to_csv(out_path, index=False)
    print(f"Merged TCGA data saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export data for R analysis")
    parser.add_argument("--emory_clinical", help="Path to Emory_Bladder_case_cleaned.csv")
    parser.add_argument("--tcga_clinical", help="Path to TCGA_clinical_data_CSV.csv")
    parser.add_argument("--tcga_results", help="Path to tcga_inference_results.csv")
    args = parser.parse_args()

    # Always export KM data
    predictions, threshold = export_km_data()

    if args.emory_clinical:
        export_clinical_merged(args.emory_clinical, predictions)

    if args.tcga_clinical:
        export_tcga_merged(args.tcga_clinical, args.tcga_results)


if __name__ == "__main__":
    main()
