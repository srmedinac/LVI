"""Central configuration for the LVI prediction pipeline."""

import os
import torch

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ────────────────────────────────────────────────────────────────
CLINICAL_CSV = os.path.join(PROJECT_ROOT, "clinical_data_emory_mibc_lvi_os.csv")

EMORY_FEATS_PATH = os.path.join(
    PROJECT_ROOT,
    "trident_outputs/emory_mibc/20x_512px_0px_overlap/"
    "late_fusion_patient_level_conch_v15_LVI",
)

TCGA_FEATS_PATH = os.path.join(
    PROJECT_ROOT,
    "trident_outputs/mibc_TCGA/20x_512px_0px_overlap/features_conch_v15",
)

TCGA_FILTERED_FEATS_PATH = os.path.join(
    PROJECT_ROOT,
    "trident_outputs/mibc_TCGA/20x_512px_0px_overlap/filtered_conch_v15_LVI",
)

TCGA_PATCHES_PATH = os.path.join(
    PROJECT_ROOT,
    "trident_outputs/mibc_TCGA/20x_512px_0px_overlap/patches",
)

TCGA_YOLO_PREDICTIONS_PATH = os.path.join(
    PROJECT_ROOT,
    "trident_outputs/mibc_TCGA/20x_512px_0px_overlap/yolo_predictions",
)

YOLO_WEIGHTS = os.path.join(
    PROJECT_ROOT, "yolo_lvi/yolo11n_run2/detect/train2/weights/best.pt"
)

# ── YOLO tile settings ───────────────────────────────────────────────────────
YOLO_TILE_SIZE = 2048        # Tile size used for YOLO prediction

# ── Output paths ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "pipeline/checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Model architecture ───────────────────────────────────────────────────────
INPUT_FEATURE_DIM = 768      # CONCH v1.5 embedding dimension
N_HEADS = 4                  # ABMIL attention heads
HEAD_DIM = 512               # ABMIL head dimension
MHA_NUM_HEADS = 8            # Multi-head attention heads (LVI patch attention)
HIDDEN_DIM = 256             # Shared hidden layer dimension
DROPOUT = 0.2
GATED = True

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_FEATURES = 512           # Patches sampled per slide during training
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Joint model
JOINT_EPOCHS = 60
JOINT_PATIENCE = 10
COX_WEIGHT = 1.0
LVI_WEIGHT = 1.0
SPARSITY_WEIGHT = 0.1
CONSISTENCY_WEIGHT = 0.3
CONSISTENCY_SCALE = 0.05     # Multiplied with CONSISTENCY_WEIGHT

# Ablation models
ABLATION_EPOCHS = 30

# Scheduler
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# ── 5-year censoring ─────────────────────────────────────────────────────────
CENSORING_DAYS = 1825

# ── Validation split ──────────────────────────────────────────────────────────
VAL_FRACTION = 0.2

# ── Heatmap generation ────────────────────────────────────────────────────────
HEATMAP_TARGET_MAG = 20
HEATMAP_PATCH_SIZE = 512
HEATMAP_OVERLAP = 0
HEATMAP_ENCODER = "conch_v15"
HEATMAP_SEGMENTATION = "hest"
