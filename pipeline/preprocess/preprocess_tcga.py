"""End-to-end TCGA preprocessing: YOLO predict → bbox-level filter → filtered H5s.

Runs vessel detection on TCGA patches and filters features to match the
Emory training pipeline (vessel-filtered features only).

Usage:
    # Full pipeline (predict + filter)
    python -m pipeline.preprocess.preprocess_tcga

    # Filter only (predictions already exist)
    python -m pipeline.preprocess.preprocess_tcga --skip_predict

    # Custom paths
    python -m pipeline.preprocess.preprocess_tcga \
        --patches_dir /path/to/patches \
        --h5_dir /path/to/features \
        --output_dir /path/to/filtered
"""

import argparse
import os

from pipeline.config import (
    TCGA_FEATS_PATH, TCGA_FILTERED_FEATS_PATH,
    TCGA_PATCHES_PATH, TCGA_YOLO_PREDICTIONS_PATH,
    YOLO_WEIGHTS,
)
from pipeline.preprocess.filter_h5 import filter_h5_by_yolo
from pipeline.preprocess.yolo_predict import predict_wsi_patches


def preprocess_tcga(
    patches_dir=TCGA_PATCHES_PATH,
    h5_dir=TCGA_FEATS_PATH,
    yolo_dir=TCGA_YOLO_PREDICTIONS_PATH,
    output_dir=TCGA_FILTERED_FEATS_PATH,
    model_path=YOLO_WEIGHTS,
    skip_predict=False,
    conf=0.4,
    iou=0.5,
    nms_iou=0.5,
    patch_size=512,
    device="cuda",
):
    """Run full TCGA preprocessing pipeline.

    Steps:
        1. Run YOLO vessel detection on TCGA patches (unless skip_predict=True)
        2. Filter H5 features by bbox-level detection overlap
        3. Output filtered H5s ready for inference
    """
    # Step 1: YOLO prediction
    if not skip_predict:
        if not os.path.isdir(patches_dir):
            raise FileNotFoundError(
                f"TCGA patches directory not found: {patches_dir}\n"
                "Either transfer YOLO predictions and use --skip_predict, "
                "or ensure patches are available."
            )
        print("=" * 60)
        print("Step 1: Running YOLO vessel detection on TCGA patches")
        print("=" * 60)
        predict_wsi_patches(
            model_path=model_path,
            patches_dir=patches_dir,
            output_dir=yolo_dir,
            conf=conf,
            iou=iou,
            device=device,
        )
    else:
        print("Skipping YOLO prediction (--skip_predict)")

    # Verify YOLO predictions exist
    if not os.path.isdir(yolo_dir):
        raise FileNotFoundError(
            f"YOLO predictions directory not found: {yolo_dir}\n"
            "Run without --skip_predict or transfer predictions first."
        )

    # Step 2: Filter H5 features
    print("\n" + "=" * 60)
    print("Step 2: Filtering TCGA features by vessel detections")
    print("=" * 60)
    filter_h5_by_yolo(
        yolo_dir=yolo_dir,
        h5_dir=h5_dir,
        output_dir=output_dir,
        nms_iou=nms_iou,
        patch_size=patch_size,
    )

    # Summary
    n_output = len([f for f in os.listdir(output_dir) if f.endswith(".h5")])
    n_input = len([f for f in os.listdir(h5_dir) if f.endswith(".h5")])
    print(f"\nDone: {n_output}/{n_input} TCGA slides filtered → {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="TCGA preprocessing: YOLO predict + filter H5 features"
    )
    parser.add_argument(
        "--patches_dir", default=TCGA_PATCHES_PATH,
        help="TCGA WSI patches directory"
    )
    parser.add_argument(
        "--h5_dir", default=TCGA_FEATS_PATH,
        help="TCGA unfiltered H5 features directory"
    )
    parser.add_argument(
        "--yolo_dir", default=TCGA_YOLO_PREDICTIONS_PATH,
        help="YOLO predictions output directory"
    )
    parser.add_argument(
        "--output_dir", default=TCGA_FILTERED_FEATS_PATH,
        help="Filtered H5 output directory"
    )
    parser.add_argument("--model", default=YOLO_WEIGHTS, help="YOLO weights path")
    parser.add_argument("--skip_predict", action="store_true",
                        help="Skip YOLO prediction (use existing predictions)")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    preprocess_tcga(
        patches_dir=args.patches_dir,
        h5_dir=args.h5_dir,
        yolo_dir=args.yolo_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        skip_predict=args.skip_predict,
        conf=args.conf,
        iou=args.iou,
        nms_iou=args.nms_iou,
        patch_size=args.patch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
