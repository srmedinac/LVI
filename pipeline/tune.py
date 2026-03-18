"""Hyperparameter tuning for the Joint model.

Sweeps over regularization, capacity, loss weights, and learning rate.
Selects best config by validation C-index, then retrains and evaluates on test.

Usage:
    python -m pipeline.tune
"""

import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from torchsurv.loss.cox import neg_partial_log_likelihood

from pipeline.config import DEVICE, SEED, CHECKPOINT_DIR, RESULTS_DIR
from pipeline.dataset import load_clinical_data, create_dataloaders, LVDataset
from pipeline.models import SurvivalModel
from pipeline.train import set_seed
from torch.utils.data import DataLoader


# ── Search space ─────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "lr": [5e-5, 1e-4, 2e-4],
    "weight_decay": [1e-3, 5e-3, 1e-2],
    "dropout": [0.3, 0.5, 0.7],
    "hidden_dim": [128, 256],
    "num_features": [256, 512],
    "cox_weight": [1.0],
    "lvi_weight": [0.5, 1.0],
    "consistency_weight": [0.0, 0.3],
    "sparsity_weight": [0.0],
}
# 3*3*3*2*2*1*2*2*1 = 432 combos

MAX_EPOCHS = 80
PATIENCE = 15


def train_one_config(config, train_loader, val_loader, verbose=False):
    """Train a joint model with given config, return val metrics."""
    model = SurvivalModel(
        dropout=config["dropout"],
        hidden_dim=config["hidden_dim"],
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_ci = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        for features, events, times, lvi in train_loader:
            features = {"features": features.to(DEVICE)}
            events = events.bool().to(DEVICE)
            times = times.to(DEVICE)
            lvi = lvi.to(DEVICE)

            optimizer.zero_grad()

            risk_scores, lvi_logits, patch_lvi_logits, patch_lvi_probs = model(
                features, return_patch_predictions=True
            )

            cox_loss = neg_partial_log_likelihood(risk_scores, events, times)
            lvi_loss = bce_loss(lvi_logits, lvi)

            loss = config["cox_weight"] * cox_loss + config["lvi_weight"] * lvi_loss

            if config["sparsity_weight"] > 0:
                entropy = -torch.sum(
                    patch_lvi_probs * torch.log(patch_lvi_probs + 1e-8), dim=1
                )
                loss += config["sparsity_weight"] * torch.mean(entropy)

            if config["consistency_weight"] > 0:
                lvi_probs = torch.sigmoid(lvi_logits)
                consistency = F.l1_loss(
                    torch.tanh(risk_scores / 3.0), 2 * lvi_probs - 1
                )
                loss += 0.05 * config["consistency_weight"] * consistency

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_risks, val_events, val_times, val_lvi_true, val_lvi_logits = (
            [], [], [], [], [],
        )
        with torch.no_grad():
            for features, events, times, lvi in val_loader:
                features = {"features": features.to(DEVICE)}
                risk_scores, lvi_logits = model(features)
                val_risks.extend(risk_scores.cpu().numpy())
                val_events.extend(events.numpy())
                val_times.extend(times.numpy())
                val_lvi_true.extend(lvi.numpy())
                val_lvi_logits.extend(lvi_logits.cpu().numpy())

        val_risks = np.array(val_risks)
        val_events = np.array(val_events)
        val_times = np.array(val_times)

        try:
            val_ci = concordance_index(val_times, -val_risks, val_events)
        except Exception:
            val_ci = 0.5

        if val_ci > best_val_ci:
            best_val_ci = val_ci
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # Compute final val LVI AUC with best model
    val_lvi_auc = float("nan")
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        model.eval()
        val_lvi_true_all, val_lvi_logits_all = [], []
        with torch.no_grad():
            for features, events, times, lvi in val_loader:
                features = {"features": features.to(DEVICE)}
                _, lvi_logits = model(features)
                val_lvi_true_all.extend(lvi.numpy())
                val_lvi_logits_all.extend(lvi_logits.cpu().numpy())
        try:
            val_lvi_probs = 1.0 / (1.0 + np.exp(-np.array(val_lvi_logits_all)))
            val_lvi_auc = roc_auc_score(np.array(val_lvi_true_all), val_lvi_probs)
        except ValueError:
            pass

    return best_val_ci, val_lvi_auc, best_state, epoch + 1


def evaluate_on_test(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    all_risk, all_lvi_logit, all_event, all_time, all_lvi_true = [], [], [], [], []

    with torch.no_grad():
        for features, events, times, lvi in test_loader:
            features = {"features": features.to(DEVICE)}
            risk_scores, lvi_logits = model(features)
            all_risk.append(risk_scores.cpu().numpy())
            all_lvi_logit.append(lvi_logits.cpu().numpy())
            all_event.append(events.numpy())
            all_time.append(times.numpy())
            all_lvi_true.append(lvi.numpy())

    risks = np.concatenate(all_risk)
    events = np.concatenate(all_event)
    times = np.concatenate(all_time)
    lvi_true = np.concatenate(all_lvi_true)
    lvi_probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_lvi_logit)))

    c_index = concordance_index(times, -risks, events)
    try:
        lvi_auc = roc_auc_score(lvi_true, lvi_probs)
    except ValueError:
        lvi_auc = float("nan")

    return c_index, lvi_auc


def main():
    set_seed()
    df = load_clinical_data()

    # Generate all combos
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = list(itertools.product(*values))
    print(f"Total configurations: {len(combos)}")

    best_ci = 0.0
    best_config = None
    best_state = None
    results = []

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))

        # Recreate dataloaders with this num_features
        set_seed()
        train_loader, val_loader, test_loader = create_dataloaders(
            df, num_features=config["num_features"]
        )

        val_ci, val_auc, state, n_epochs = train_one_config(
            config, train_loader, val_loader
        )

        results.append((val_ci, val_auc, config, n_epochs))

        marker = " ***" if val_ci > best_ci else ""
        print(
            f"[{i+1}/{len(combos)}] Val C-index={val_ci:.4f} AUC={val_auc:.4f} "
            f"ep={n_epochs} | lr={config['lr']:.0e} wd={config['weight_decay']:.0e} "
            f"do={config['dropout']} hd={config['hidden_dim']} "
            f"nf={config['num_features']} lw={config['lvi_weight']} "
            f"cw={config['consistency_weight']} sw={config['sparsity_weight']}"
            f"{marker}"
        )

        if val_ci > best_ci:
            best_ci = val_ci
            best_config = config
            best_state = state

    # ── Top 10 results ───────────────────────────────────────────────────
    results.sort(key=lambda x: x[0], reverse=True)
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGS BY VAL C-INDEX")
    print("=" * 70)
    for rank, (ci, auc, cfg, ep) in enumerate(results[:10], 1):
        print(
            f"  #{rank}: C-index={ci:.4f} AUC={auc:.4f} ep={ep} | "
            f"lr={cfg['lr']:.0e} wd={cfg['weight_decay']:.0e} "
            f"do={cfg['dropout']} hd={cfg['hidden_dim']} "
            f"nf={cfg['num_features']} lw={cfg['lvi_weight']} "
            f"cw={cfg['consistency_weight']} sw={cfg['sparsity_weight']}"
        )

    # ── Evaluate best on test ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"BEST CONFIG: {best_config}")
    print(f"Val C-index: {best_ci:.4f}")

    model = SurvivalModel(
        dropout=best_config["dropout"],
        hidden_dim=best_config["hidden_dim"],
    ).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    set_seed()
    _, _, test_loader = create_dataloaders(
        df, num_features=best_config["num_features"]
    )
    test_ci, test_auc = evaluate_on_test(model, test_loader)
    print(f"Test C-index: {test_ci:.4f}")
    print(f"Test LVI AUC: {test_auc:.4f}")

    # Save best model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "joint_model_tuned.pth"),
    )
    print(f"Saved to {CHECKPOINT_DIR}/joint_model_tuned.pth")


if __name__ == "__main__":
    main()
