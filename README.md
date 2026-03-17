# LVI Prediction from H&E Whole-Slide Images in Bladder Cancer

Predicting lymphovascular invasion (LVI) status and overall survival (OS) from H&E-stained whole-slide images (WSIs) in muscle-invasive bladder cancer (MIBC) using a multi-task deep learning pipeline.

## Approach

```
WSI → YOLO11n vessel detection → bbox-level feature filtering → ABMIL multi-task learning
```

1. **YOLO11n** detects vessel-like structures on 2048×2048 tiles extracted on-the-fly from WSIs
2. **Feature filtering** keeps only CONCH v1.5 patch embeddings near YOLO detections (bbox center-in-box matching + global NMS)
3. **Multi-task ABMIL** jointly predicts LVI (binary) and OS (Cox regression) with a learnable fusion gate and patch-level LVI attention

## Pipeline Structure

```
pipeline/
├── config.py                 # Paths, hyperparameters, device settings
├── dataset.py                # Data loading with confidence-weighted sampling
├── models.py                 # SurvivalModel (joint), SurvivalOnlyModel, LVIOnlyModel
├── train.py                  # Multi-task training (Cox + BCE + sparsity + consistency)
├── evaluate.py               # C-index, AUC, Kaplan-Meier, ROC curves
├── inference_tcga.py         # External validation on TCGA-BLCA
├── export_for_r.py           # Export predictions for R survival analysis
├── heatmaps.py               # ABMIL attention heatmap visualization
└── preprocess/
    ├── yolo_predict.py       # On-the-fly YOLO inference on WSI tiles
    ├── filter_h5.py          # Filter CONCH features by YOLO detections
    ├── concat_patient_h5.py  # Merge slide-level features per patient
    ├── prepare_annotations.py # Convert LVI masks → COCO → YOLO format
    └── preprocess_tcga.py    # End-to-end TCGA preprocessing
```

## Model Architecture

The **joint model** (`SurvivalModel`) combines:

- **Patch-level LVI attention**: 8-head multi-head attention on 768-dim CONCH v1.5 features, with a per-patch LVI prediction head
- **Learnable fusion gate**: `nn.Parameter` initialized at 0.2 (sigmoid ≈ 0.55) that blends LVI-weighted and original feature representations
- **Gated ABMIL**: 4-head, 512-dim attention-based MIL for slide-level aggregation
- **Dual output heads**: survival risk score (tanh × 3.0 for Cox) and binary LVI classification

Two ablation models (survival-only, LVI-only) isolate each task's contribution.

### Training Losses

| Loss | Weight | Description |
|------|--------|-------------|
| Cox partial log-likelihood | 1.0 | Survival prediction via `torchsurv` |
| Binary cross-entropy | 1.0 | Patient-level LVI classification |
| Sparsity (entropy) | 0.1 | Encourages sparse patch-level LVI attention |
| Consistency (L1) | 0.015 | Aligns normalized risk score with LVI probability |

## Workflow

### 1. Preprocessing

**YOLO vessel detection** (on-the-fly, no tile images saved to disk):
```bash
python -m pipeline.preprocess.yolo_predict \
    --wsi_dir /path/to/wsi_images \
    --patches_h5_dir /path/to/trident_patches_h5 \
    --output_dir yolo_predictions/emory \
    --batch_size 64
```

**Filter features** by YOLO detections (global NMS + center-in-box):
```bash
python -m pipeline.preprocess.filter_h5 \
    --features_dir /path/to/conch_features \
    --patches_h5_dir /path/to/trident_patches_h5 \
    --yolo_dir yolo_predictions/emory \
    --output_dir /path/to/filtered_features
```

**Concatenate per patient** (merge multi-slide patients):
```bash
python -m pipeline.preprocess.concat_patient_h5 \
    --input_dir /path/to/filtered_features \
    --output_dir /path/to/patient_level_features \
    --clinical_csv clinical_data_emory_mibc_lvi_os.csv
```

### 2. Training

```bash
# Joint multi-task model (recommended)
python -m pipeline.train --model joint

# Full ablation study (joint + survival-only + LVI-only)
python -m pipeline.train --model ablation
```

### 3. Evaluation

```bash
python -m pipeline.evaluate
```

Produces: C-index, LVI AUC, Kaplan-Meier curves, ROC curves, and ablation comparison table in `results/`.

### 4. TCGA External Validation

```bash
# Preprocess TCGA (YOLO + filter in one step)
python -m pipeline.preprocess.preprocess_tcga

# Run inference
python -m pipeline.inference_tcga
```

### 5. Export for R Analysis

```bash
python -m pipeline.export_for_r
```

Exports `survival_data_for_R.csv` and `clinical_merged_{emory,tcga}_final.csv` for multivariate Cox regression and forest plots in R.

### 6. Attention Heatmaps

```bash
python -m pipeline.heatmaps --slide_path /path/to/slide.ndpi
```

## Data

| Cohort | Patients | Slides | LVI+ | OS Events |
|--------|----------|--------|------|-----------|
| Emory (train) | 81 | — | — | — |
| Emory (test) | 82 | — | 54 total | 73 total |
| TCGA-BLCA | 53 | — | N/A | — |

- **Features**: 768-dim CONCH v1.5 embeddings extracted at 20× magnification, 512px patches
- **Sampling**: 512 patches per patient during training (multinomial by YOLO confidence)
- **Censoring**: 5-year (1825 days)

## Dependencies

Requires the `trident` conda environment:

- PyTorch (with CUDA)
- Ultralytics (YOLO11n)
- h5py, openslide-python
- torchsurv, lifelines
- scikit-learn, pandas, numpy, matplotlib
- trident (slide encoders, tissue segmentation)

## Setup

```bash
conda activate trident
```

All paths are configured in `pipeline/config.py`.
