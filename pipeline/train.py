"""Training loops for Joint, Survival-Only, and LVI-Only models.

Usage:
    python -m pipeline.train                  # Train all 3 models (ablation)
    python -m pipeline.train --model Joint    # Train only the joint model
    python -m pipeline.train --epochs 2       # Quick smoke test
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lifelines.utils import concordance_index
from torchsurv.loss.cox import neg_partial_log_likelihood

from pipeline.config import (
    DEVICE, SEED, CHECKPOINT_DIR,
    LR, WEIGHT_DECAY, GRAD_CLIP,
    JOINT_EPOCHS, JOINT_PATIENCE,
    COX_WEIGHT, LVI_WEIGHT, SPARSITY_WEIGHT,
    CONSISTENCY_WEIGHT, CONSISTENCY_SCALE,
    ABLATION_EPOCHS,
    SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
)
from pipeline.models import SurvivalModel, SurvivalOnlyModel, LVIOnlyModel
from pipeline.dataset import load_clinical_data, create_dataloaders


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_joint_model(train_loader, val_loader, epochs=JOINT_EPOCHS):
    """Train the full joint model with multi-task loss."""
    print("=" * 50)
    print("TRAINING JOINT MODEL")
    print("=" * 50)

    model = SurvivalModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
    )
    bce_loss = nn.BCEWithLogitsLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        total_cox = 0.0
        total_lvi = 0.0
        total_consistency = 0.0
        total_sparsity = 0.0

        for features, events, times, lvi in train_loader:
            features = {"features": features.to(DEVICE)}
            events = events.bool().to(DEVICE)
            times = times.to(DEVICE)
            lvi = lvi.to(DEVICE)

            optimizer.zero_grad()
            risk_scores, lvi_logits = model(features)

            cox_loss = neg_partial_log_likelihood(risk_scores, events, times)
            lvi_loss = bce_loss(lvi_logits, lvi)

            # Sparsity loss (entropy of patch LVI probs)
            _, _, patch_lvi_logits, patch_lvi_probs = model(
                features, return_patch_predictions=True
            )
            entropy = -torch.sum(
                patch_lvi_probs * torch.log(patch_lvi_probs + 1e-8), dim=1
            )
            sparsity_loss = torch.mean(entropy)

            # Consistency loss between risk score and LVI prediction
            lvi_probs = torch.sigmoid(lvi_logits)
            consistency_loss = F.l1_loss(
                torch.tanh(risk_scores / 3.0), 2 * lvi_probs - 1
            )

            loss = (
                COX_WEIGHT * cox_loss
                + LVI_WEIGHT * lvi_loss
                + SPARSITY_WEIGHT * sparsity_loss
                + CONSISTENCY_SCALE * CONSISTENCY_WEIGHT * consistency_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_cox += cox_loss.item()
            total_lvi += lvi_loss.item()
            total_consistency += consistency_loss.item()
            total_sparsity += sparsity_loss.item()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_risks, val_events, val_times = [], [], []

        with torch.no_grad():
            for features, events, times, lvi in val_loader:
                features = {"features": features.to(DEVICE)}
                events = events.bool().to(DEVICE)
                times = times.to(DEVICE)
                lvi = lvi.to(DEVICE)

                risk_scores, lvi_logits = model(features)
                val_risks.extend(risk_scores.cpu().numpy())
                val_events.extend(events.cpu().numpy())
                val_times.extend(times.cpu().numpy())

                cox_loss = neg_partial_log_likelihood(risk_scores, events, times)
                lvi_loss_val = bce_loss(lvi_logits, lvi)

                _, _, _, patch_probs = model(
                    features, return_patch_predictions=True
                )
                entropy = -torch.sum(
                    patch_probs * torch.log(patch_probs + 1e-8), dim=1
                )
                sparsity = torch.mean(entropy)

                lvi_probs = torch.sigmoid(lvi_logits)
                consistency = F.l1_loss(
                    torch.tanh(risk_scores / 3.0), 2 * lvi_probs - 1
                )

                val_loss += (
                    COX_WEIGHT * cox_loss
                    + LVI_WEIGHT * lvi_loss_val
                    + SPARSITY_WEIGHT * sparsity
                    + CONSISTENCY_SCALE * CONSISTENCY_WEIGHT * consistency
                ).item()

        val_c_index = concordance_index(
            np.array(val_times), -np.array(val_risks), np.array(val_events)
        )
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}, "
                f"C-index={val_c_index:.3f}"
            )

        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "joint_model_best.pth"),
            )
        else:
            patience_counter += 1

        if patience_counter >= JOINT_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best val loss: {best_loss:.4f}")
    return model


def train_survival_only_model(train_loader, epochs=ABLATION_EPOCHS):
    """Train model only on survival task (Cox loss)."""
    print("=" * 50)
    print("TRAINING SURVIVAL-ONLY MODEL")
    print("=" * 50)

    model = SurvivalOnlyModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for features, events, times, lvi in train_loader:
            features = {"features": features.to(DEVICE)}
            events = events.bool().to(DEVICE)
            times = times.to(DEVICE)

            optimizer.zero_grad()
            risk_scores, _ = model(features)

            cox_loss = neg_partial_log_likelihood(risk_scores, events, times)
            cox_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            total_loss += cox_loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Survival Loss = {total_loss/len(train_loader):.4f}")

    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "survival_only_model.pth"),
    )
    return model


def train_lvi_only_model(train_loader, epochs=ABLATION_EPOCHS):
    """Train model only on LVI task (BCE loss)."""
    print("=" * 50)
    print("TRAINING LVI-ONLY MODEL")
    print("=" * 50)

    model = LVIOnlyModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce_loss = nn.BCEWithLogitsLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for features, events, times, lvi in train_loader:
            features = {"features": features.to(DEVICE)}
            lvi = lvi.to(DEVICE)

            optimizer.zero_grad()
            _, lvi_logits = model(features)

            lvi_loss = bce_loss(lvi_logits, lvi)
            lvi_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            total_loss += lvi_loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: LVI Loss = {total_loss/len(train_loader):.4f}")

    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "lvi_only_model.pth"),
    )
    return model


def run_ablation(epochs_override=None):
    """Train all 3 models sequentially."""
    set_seed()

    df = load_clinical_data()
    train_loader, val_loader, test_loader = create_dataloaders(df)

    print(f"\nData splits:")
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split]
        print(
            f"  {split.capitalize()}: {len(subset)} samples, "
            f"OS event rate {subset['os'].mean():.3f} ({int(subset['os'].sum())}/{len(subset)})"
        )
    print()

    joint_epochs = epochs_override or JOINT_EPOCHS
    ablation_epochs = epochs_override or ABLATION_EPOCHS

    train_joint_model(train_loader, val_loader, epochs=joint_epochs)
    train_survival_only_model(train_loader, epochs=ablation_epochs)
    train_lvi_only_model(train_loader, epochs=ablation_epochs)

    print("\nAll models trained. Checkpoints saved to:", CHECKPOINT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Train LVI prediction models")
    parser.add_argument(
        "--model",
        choices=["Joint", "Survival-Only", "LVI-Only", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs (for smoke testing)",
    )
    args = parser.parse_args()

    set_seed()
    df = load_clinical_data()
    train_loader, val_loader, _ = create_dataloaders(df)

    print(f"\nData splits:")
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split]
        print(f"  {split.capitalize()}: {len(subset)} samples")
    print()

    if args.model == "all":
        run_ablation(epochs_override=args.epochs)
    elif args.model == "Joint":
        train_joint_model(
            train_loader, val_loader,
            epochs=args.epochs or JOINT_EPOCHS,
        )
    elif args.model == "Survival-Only":
        train_survival_only_model(
            train_loader,
            epochs=args.epochs or ABLATION_EPOCHS,
        )
    elif args.model == "LVI-Only":
        train_lvi_only_model(
            train_loader,
            epochs=args.epochs or ABLATION_EPOCHS,
        )


if __name__ == "__main__":
    main()
