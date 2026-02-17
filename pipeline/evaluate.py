"""Evaluation: metrics, KM curves, ROC plots, and ablation comparison.

Usage:
    python -m pipeline.evaluate              # Evaluate all 3 models on train + test
    python -m pipeline.evaluate --model Joint
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
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from pipeline.config import DEVICE, SEED, CHECKPOINT_DIR, RESULTS_DIR
from pipeline.dataset import load_clinical_data, create_dataloaders
from pipeline.models import MODELS, CHECKPOINT_NAMES
from pipeline.train import set_seed


def evaluate_model(model, loader, model_name, split_name="test"):
    """Run inference and compute survival + LVI metrics."""
    model.eval()
    all_risk, all_lvi_logit, all_event, all_time, all_lvi_true = [], [], [], [], []

    with torch.no_grad():
        for features, events, times, lvi in tqdm(loader, desc=f"{model_name} ({split_name})"):
            features = {"features": features.to(DEVICE)}
            risk_scores, lvi_logits = model(features)

            all_risk.append(risk_scores.cpu().numpy())
            all_lvi_logit.append(lvi_logits.cpu().numpy())
            all_event.append(events.numpy())
            all_time.append(times.numpy())
            all_lvi_true.append(lvi.numpy())

    risk_scores = np.concatenate(all_risk)
    lvi_logits = np.concatenate(all_lvi_logit)
    events = np.concatenate(all_event)
    times = np.concatenate(all_time)
    lvi_true = np.concatenate(all_lvi_true)
    lvi_probs = 1.0 / (1.0 + np.exp(-lvi_logits))  # sigmoid

    # Metrics
    c_index = concordance_index(times, -risk_scores, events)
    try:
        lvi_auc = roc_auc_score(lvi_true, lvi_probs)
    except ValueError:
        lvi_auc = float("nan")

    print(f"\n{model_name} [{split_name}]:")
    print(f"  C-index:  {c_index:.4f}")
    print(f"  LVI AUC:  {lvi_auc:.4f}")

    return {
        "model": model_name,
        "split": split_name,
        "c_index": c_index,
        "lvi_auc": lvi_auc,
        "risk_scores": risk_scores,
        "lvi_probs": lvi_probs,
        "events": events,
        "times": times,
        "lvi_true": lvi_true,
    }


def load_model(model_name):
    """Load a trained model from checkpoint."""
    model_cls = MODELS[model_name]
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAMES[model_name])
    model = model_cls().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def plot_km_curves(results, output_dir=RESULTS_DIR):
    """Generate Kaplan-Meier plots for each model/split combination."""
    os.makedirs(output_dir, exist_ok=True)

    for res in results:
        model_name = res["model"]
        split = res["split"]
        risks = res["risk_scores"]
        events = res["events"].astype(bool)
        times = res["times"] / 365.25  # Convert to years

        median_risk = np.median(risks)
        high_risk = risks > median_risk

        fig, ax = plt.subplots(figsize=(8, 6))
        kmf_low = KaplanMeierFitter()
        kmf_high = KaplanMeierFitter()

        kmf_low.fit(times[~high_risk], events[~high_risk], label="Low Risk")
        kmf_high.fit(times[high_risk], events[high_risk], label="High Risk")

        kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#00BFC4")
        kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#F8766D")

        # Log-rank test
        lr = logrank_test(
            times[~high_risk], times[high_risk],
            events[~high_risk], events[high_risk],
        )

        ax.set_title(f"{model_name} - {split.capitalize()} (p={lr.p_value:.4f})")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Overall Survival")
        ax.legend()

        fname = f"km_{model_name.lower().replace('-', '_')}_{split}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_roc_curves(results, output_dir=RESULTS_DIR):
    """Generate ROC plots for LVI classification."""
    os.makedirs(output_dir, exist_ok=True)

    for res in results:
        if np.isnan(res["lvi_auc"]):
            continue

        fpr, tpr, _ = roc_curve(res["lvi_true"], res["lvi_probs"])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="#F8766D", lw=2,
                label=f'AUC = {res["lvi_auc"]:.3f}')
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f'{res["model"]} - {res["split"].capitalize()} LVI ROC')
        ax.legend()

        fname = f'roc_{res["model"].lower().replace("-", "_")}_{res["split"]}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def generate_summary_table(results):
    """Create a comparison DataFrame from evaluation results."""
    rows = []
    for res in results:
        rows.append({
            "Model": res["model"],
            "Split": res["split"],
            "C-index": round(res["c_index"], 4),
            "LVI AUC": round(res["lvi_auc"], 4),
            "N": len(res["events"]),
            "Events": int(res["events"].sum()),
        })
    return pd.DataFrame(rows)


def run_ablation_evaluation():
    """Load all 3 checkpoints and evaluate on train + test."""
    set_seed()
    df = load_clinical_data()
    train_loader, val_loader, test_loader = create_dataloaders(df)

    all_results = []

    for model_name in ("Joint", "Survival-Only", "LVI-Only"):
        ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAMES[model_name])
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {model_name}, skipping.")
            continue

        model = load_model(model_name)

        for split_name, loader in [("train", train_loader), ("test", test_loader)]:
            res = evaluate_model(model, loader, model_name, split_name)
            all_results.append(res)

    # Summary
    summary = generate_summary_table(all_results)
    print("\n" + "=" * 60)
    print("ABLATION COMPARISON")
    print("=" * 60)
    print(summary.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "ablation_summary.csv"), index=False)

    # Plots
    print("\nGenerating plots...")
    plot_km_curves(all_results)
    plot_roc_curves(all_results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LVI models")
    parser.add_argument(
        "--model",
        choices=["Joint", "Survival-Only", "LVI-Only", "all"],
        default="all",
    )
    args = parser.parse_args()

    set_seed()
    df = load_clinical_data()
    train_loader, val_loader, test_loader = create_dataloaders(df)

    if args.model == "all":
        run_ablation_evaluation()
    else:
        model = load_model(args.model)
        for split_name, loader in [("train", train_loader), ("test", test_loader)]:
            evaluate_model(model, loader, args.model, split_name)


if __name__ == "__main__":
    main()
