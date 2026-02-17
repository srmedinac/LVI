"""Run YOLO11n vessel detection on WSI tiles read on-the-fly.

Reads patch coordinates from trident H5 files, groups 512px patches into
2048x2048 tiles, reads each tile from the WSI via openslide, and runs YOLO
in memory. Saves only detection labels (no tile images on disk).

Usage:
    python -m pipeline.preprocess.yolo_predict \
        --wsi_dir /path/to/wsi_images \
        --patches_h5_dir /path/to/patches_h5 \
        --output_dir /path/to/predictions
"""

import argparse
import os

import h5py
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from pipeline.config import YOLO_WEIGHTS, YOLO_TILE_SIZE


def coords_to_tiles(patch_coords, tile_size=YOLO_TILE_SIZE):
    """Group 512px patch coordinates into tile-sized regions.

    Args:
        patch_coords: (N, 2) array of patch top-left (x, y) in WSI pixels.
        tile_size: Tile size for YOLO (default 2048).

    Returns:
        List of (tile_x, tile_y) tuples representing unique tile origins.
    """
    # Snap each patch to its parent tile
    tile_origins = set()
    for x, y in patch_coords:
        tile_x = int(x // tile_size) * tile_size
        tile_y = int(y // tile_size) * tile_size
        tile_origins.add((tile_x, tile_y))
    return sorted(tile_origins)


def predict_wsi_onthefly(
    model,
    wsi_path,
    patches_h5_path,
    output_dir,
    tile_size=YOLO_TILE_SIZE,
    conf=0.4,
    iou=0.5,
    imgsz=1024,
    device="cuda",
):
    """Run YOLO on tiles read on-the-fly from a single WSI.

    Args:
        model: Loaded YOLO model.
        wsi_path: Path to WSI file (.ndpi, .svs, etc.).
        patches_h5_path: H5 file with 'coords' dataset (patch coordinates).
        output_dir: Output directory for this WSI's labels.
        tile_size: Tile size to extract (default 2048).
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        imgsz: YOLO input image size.
        device: Device for inference.
    """
    # Read patch coordinates from H5
    with h5py.File(patches_h5_path, "r") as f:
        if "coords" not in f:
            print(f"  Warning: No coords in {patches_h5_path}, skipping")
            return 0
        coords = f["coords"][:]

    # Group into tiles
    tiles = coords_to_tiles(coords, tile_size)
    if not tiles:
        return 0

    # Open WSI
    slide = openslide.OpenSlide(wsi_path)
    slide_w, slide_h = slide.dimensions

    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    det_count = 0

    for tile_x, tile_y in tiles:
        # Clamp to slide bounds
        read_w = min(tile_size, slide_w - tile_x)
        read_h = min(tile_size, slide_h - tile_y)
        if read_w <= 0 or read_h <= 0:
            continue

        # Read region from WSI (level 0)
        region = slide.read_region((tile_x, tile_y), 0, (read_w, read_h))
        tile_img = region.convert("RGB")

        # Pad to full tile_size if needed (edge tiles)
        if read_w < tile_size or read_h < tile_size:
            padded = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
            padded.paste(tile_img, (0, 0))
            tile_img = padded

        # Run YOLO
        results = model.predict(
            source=tile_img,
            save=False,
            save_txt=False,
            save_conf=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )

        # Save detections as YOLO txt
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            label_name = f"x{tile_x}_y{tile_y}_w{tile_size}_h{tile_size}.txt"
            label_path = os.path.join(labels_dir, label_name)

            with open(label_path, "w") as lf:
                for box in boxes:
                    cls = int(box.cls.item())
                    # xywhn: normalized center x, center y, width, height
                    xywhn = box.xywhn[0]
                    cx, cy, w, h = xywhn[0].item(), xywhn[1].item(), xywhn[2].item(), xywhn[3].item()
                    conf_val = box.conf.item()
                    lf.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf_val:.6f}\n")
                    det_count += 1

    slide.close()
    return det_count


def predict_all_wsis(
    model_path,
    wsi_dir,
    patches_h5_dir,
    output_dir,
    tile_size=YOLO_TILE_SIZE,
    conf=0.4,
    iou=0.5,
    imgsz=1024,
    device="cuda",
):
    """Run YOLO on all WSIs, reading tiles on-the-fly.

    Args:
        model_path: Path to YOLO weights.
        wsi_dir: Directory of WSI files.
        patches_h5_dir: Directory of trident patches H5 files (with coords).
        output_dir: Output predictions root directory.
        tile_size: Tile size for extraction.
        conf: Confidence threshold.
        iou: IoU threshold.
        imgsz: YOLO input image size.
        device: Device for inference.
    """
    model = YOLO(model_path)

    # Build WSI lookup: stem → full path
    wsi_extensions = (".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".scn")
    wsi_lookup = {}
    for f in os.listdir(wsi_dir):
        if any(f.lower().endswith(ext) for ext in wsi_extensions):
            stem = os.path.splitext(f)[0]
            wsi_lookup[stem] = os.path.join(wsi_dir, f)

    # Get all patches H5 files
    h5_files = sorted([
        f for f in os.listdir(patches_h5_dir)
        if f.endswith("_patches.h5") or f.endswith(".h5")
    ])

    print(f"Found {len(h5_files)} H5 files, {len(wsi_lookup)} WSIs")

    total_dets = 0
    processed = 0
    skipped = 0

    for h5_file in tqdm(h5_files, desc="YOLO prediction"):
        # Match H5 to WSI: H5 name is WSI_NAME_patches.h5 or WSI_NAME.h5
        h5_stem = h5_file.replace("_patches.h5", "").replace(".h5", "")

        wsi_output_dir = os.path.join(output_dir, h5_stem)

        # Skip if already processed
        if os.path.exists(os.path.join(wsi_output_dir, "labels")):
            skipped += 1
            continue

        # Find matching WSI
        wsi_path = wsi_lookup.get(h5_stem)
        if wsi_path is None:
            # Try fuzzy match (H5 stem might be prefix of WSI name)
            matches = [k for k in wsi_lookup if k.startswith(h5_stem) or h5_stem.startswith(k)]
            if matches:
                wsi_path = wsi_lookup[matches[0]]

        if wsi_path is None:
            print(f"\n  Warning: No WSI found for {h5_stem}, skipping")
            skipped += 1
            continue

        h5_path = os.path.join(patches_h5_dir, h5_file)

        n_dets = predict_wsi_onthefly(
            model=model,
            wsi_path=wsi_path,
            patches_h5_path=h5_path,
            output_dir=wsi_output_dir,
            tile_size=tile_size,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
        total_dets += n_dets
        processed += 1

    print(f"\nDone: {processed} WSIs processed, {skipped} skipped, {total_dets} total detections")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO vessel detection on WSI tiles (on-the-fly, no tile extraction)"
    )
    parser.add_argument("--wsi_dir", required=True, help="Directory of WSI files")
    parser.add_argument("--patches_h5_dir", required=True,
                        help="Directory of trident patches H5 files (with coords)")
    parser.add_argument("--output_dir", required=True, help="Output predictions directory")
    parser.add_argument("--model", default=YOLO_WEIGHTS, help="YOLO weights path")
    parser.add_argument("--tile_size", type=int, default=YOLO_TILE_SIZE,
                        help="Tile size for YOLO (default: 2048)")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    predict_all_wsis(
        model_path=args.model,
        wsi_dir=args.wsi_dir,
        patches_h5_dir=args.patches_h5_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
