"""Model architectures for LVI prediction and survival analysis.

Three models for ablation study:
  - SurvivalModel (Joint): MHA → patch LVI → weighted features → ABMIL → Cox + BCE
  - SurvivalOnlyModel: ABMIL → Cox (LVI head untrained)
  - LVIOnlyModel: MHA → patch LVI → weighted features → ABMIL → BCE (survival head untrained)
"""

import torch
import torch.nn as nn
from trident.slide_encoder_models import ABMILSlideEncoder

from pipeline.config import (
    INPUT_FEATURE_DIM, N_HEADS, HEAD_DIM, MHA_NUM_HEADS,
    HIDDEN_DIM, DROPOUT, GATED,
)


class SurvivalModel(nn.Module):
    """Joint multi-task model: survival + LVI prediction."""

    def __init__(
        self,
        input_feature_dim=INPUT_FEATURE_DIM,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        dropout=DROPOUT,
        gated=GATED,
        hidden_dim=HIDDEN_DIM,
    ):
        super().__init__()

        self.lvi_patch_attention = nn.MultiheadAttention(
            embed_dim=input_feature_dim,
            num_heads=MHA_NUM_HEADS,
            dropout=dropout,
            batch_first=True,
        )

        self.patch_lvi_head = nn.Sequential(
            nn.Linear(input_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated,
        )

        self.shared_hidden = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        self.lvi_aggregator = nn.Sequential(
            nn.Linear(input_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Learnable fusion gate: sigmoid(0.2) ≈ 0.55 (init close to original 0.6)
        self.fusion_gate = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, return_raw_attention=False, return_patch_predictions=False):
        patch_features = x["features"]

        lvi_attended, lvi_attn_weights = self.lvi_patch_attention(
            patch_features, patch_features, patch_features
        )

        patch_lvi_logits = self.patch_lvi_head(lvi_attended)
        patch_lvi_probs = torch.sigmoid(patch_lvi_logits.squeeze(-1))

        lvi_weighted = patch_features * patch_lvi_probs.unsqueeze(-1)
        g = torch.sigmoid(self.fusion_gate)
        combined = {"features": g * lvi_weighted + (1 - g) * patch_features}

        if return_raw_attention:
            slide_features, main_attn = self.feature_encoder(
                combined, return_raw_attention=True
            )
        else:
            slide_features = self.feature_encoder(combined)

        shared = self.shared_hidden(slide_features)
        risk_score = self.survival_head(shared).squeeze(1) * 3.0

        lvi_context = torch.mean(lvi_weighted, dim=1)
        lvi_logit = self.lvi_aggregator(lvi_context).squeeze(1)

        if return_patch_predictions:
            if return_raw_attention:
                return risk_score, lvi_logit, main_attn, patch_lvi_logits, lvi_attn_weights
            return risk_score, lvi_logit, patch_lvi_logits, patch_lvi_probs

        if return_raw_attention:
            return risk_score, lvi_logit, main_attn

        return risk_score, lvi_logit


class SurvivalOnlyModel(nn.Module):
    """Survival-only model (no LVI components in training)."""

    def __init__(
        self,
        input_feature_dim=INPUT_FEATURE_DIM,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        dropout=DROPOUT,
        gated=GATED,
        hidden_dim=HIDDEN_DIM,
    ):
        super().__init__()

        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated,
        )

        self.shared_hidden = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # LVI head present for evaluation only (not trained)
        self.lvi_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            slide_features, attention = self.feature_encoder(
                x, return_raw_attention=True
            )
        else:
            slide_features = self.feature_encoder(x)

        shared = self.shared_hidden(slide_features)
        risk_score = self.survival_head(shared).squeeze(1) * 3.0
        lvi_logit = self.lvi_head(shared).squeeze(1)

        if return_raw_attention:
            return risk_score, lvi_logit, attention

        return risk_score, lvi_logit


class LVIOnlyModel(nn.Module):
    """LVI-only model (survival head present but not trained)."""

    def __init__(
        self,
        input_feature_dim=INPUT_FEATURE_DIM,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        dropout=DROPOUT,
        gated=GATED,
        hidden_dim=HIDDEN_DIM,
    ):
        super().__init__()

        self.lvi_patch_attention = nn.MultiheadAttention(
            embed_dim=input_feature_dim,
            num_heads=MHA_NUM_HEADS,
            dropout=dropout,
            batch_first=True,
        )

        self.patch_lvi_head = nn.Sequential(
            nn.Linear(input_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated,
        )

        self.shared_hidden = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lvi_aggregator = nn.Sequential(
            nn.Linear(input_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Learnable fusion gate: sigmoid(0.2) ≈ 0.55 (init close to original 0.6)
        self.fusion_gate = nn.Parameter(torch.tensor(0.2))

        # Survival head present for evaluation only (not trained)
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x, return_raw_attention=False):
        patch_features = x["features"]

        lvi_attended, _ = self.lvi_patch_attention(
            patch_features, patch_features, patch_features
        )
        patch_lvi_logits = self.patch_lvi_head(lvi_attended)
        patch_lvi_probs = torch.sigmoid(patch_lvi_logits.squeeze(-1))

        lvi_weighted = patch_features * patch_lvi_probs.unsqueeze(-1)
        g = torch.sigmoid(self.fusion_gate)
        combined = {"features": g * lvi_weighted + (1 - g) * patch_features}

        if return_raw_attention:
            slide_features, attention = self.feature_encoder(
                combined, return_raw_attention=True
            )
        else:
            slide_features = self.feature_encoder(combined)

        shared = self.shared_hidden(slide_features)

        lvi_context = torch.mean(lvi_weighted, dim=1)
        lvi_logit = self.lvi_aggregator(lvi_context).squeeze(1)
        risk_score = self.survival_head(shared).squeeze(1) * 3.0

        if return_raw_attention:
            return risk_score, lvi_logit, attention

        return risk_score, lvi_logit


# Model registry for easy lookup
MODELS = {
    "Joint": SurvivalModel,
    "Survival-Only": SurvivalOnlyModel,
    "LVI-Only": LVIOnlyModel,
}

CHECKPOINT_NAMES = {
    "Joint": "joint_model_best.pth",
    "Survival-Only": "survival_only_model.pth",
    "LVI-Only": "lvi_only_model.pth",
}
