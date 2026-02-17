"""Run YOLO11n vessel detection on WSI patches.

Usage:
    python -m pipeline.preprocess.yolo_predict \
        --patches_dir /path/to/wsi_patches \
        --output_dir /path/to/predictions
"""

import argparse
import os

from tqdm import tqdm
from ultralytics import YOLO

from pipeline.config import YOLO_WEIGHTS


def predict_wsi_patches(
    model_path,
    patches_dir,
    output_dir,
    conf=0.4,
    iou=0.5,
    imgsz=1024,
    device="cuda",
):
    """Run YOLO inference on all WSI patch directories.

    Expected directory structure:
        patches_dir/
            WSI_NAME/
                WSI_NAME/
                    tiles/
                        *.png
    """
    model = YOLO(model_path)

    wsis = [
        d for d in os.listdir(patches_dir)
        if os.path.isdir(os.path.join(patches_dir, d))
    ]
    print(f"Found {len(wsis)} WSIs for prediction")

    for wsi in tqdm(wsis):
        output_folder = os.path.join(output_dir, wsi)
        if os.path.exists(output_folder):
            print(f"Skipping {wsi} — already processed")
            continue

        # Handle nested directory structure (WSI_NAME/WSI_NAME/tiles/)
        nested = os.path.join(patches_dir, wsi, wsi, "tiles")
        flat = os.path.join(patches_dir, wsi, "tiles")

        if os.path.isdir(nested):
            patch_dir = nested
        elif os.path.isdir(flat):
            patch_dir = flat
        else:
            print(f"Warning: No tiles directory found for {wsi}, skipping")
            continue

        print(f"Processing: {patch_dir}")
        results = model.predict(
            source=patch_dir,
            save=False,
            save_txt=True,
            save_conf=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            project=output_dir,
            name=wsi,
            stream=True,
            verbose=False,
        )
        # Consume the generator to run inference
        for _ in results:
            pass


def main():
    parser = argparse.ArgumentParser(description="YOLO vessel detection on WSI patches")
    parser.add_argument("--patches_dir", required=True, help="Directory of WSI patches")
    parser.add_argument("--output_dir", required=True, help="Output predictions directory")
    parser.add_argument("--model", default=YOLO_WEIGHTS, help="YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    predict_wsi_patches(
        model_path=args.model,
        patches_dir=args.patches_dir,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
