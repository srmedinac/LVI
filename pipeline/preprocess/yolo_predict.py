"""Run YOLO11n vessel detection on WSI tiles read on-the-fly.

Reads patch coordinates from trident H5 files, groups 512px patches into
2048x2048 tiles, reads each tile from the WSI via openslide, and runs YOLO
in batched mode. Uses threaded prefetching to overlap I/O with GPU inference.

Usage:
    python -m pipeline.preprocess.yolo_predict \
        --wsi_dir /path/to/wsi_images \
        --patches_h5_dir /path/to/patches_h5 \
        --output_dir /path/to/predictions
"""

import argparse
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread

import h5py
import numpy as np
import openslide
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from pipeline.config import YOLO_WEIGHTS, YOLO_TILE_SIZE

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 8


def coords_to_tiles(patch_coords, tile_size=YOLO_TILE_SIZE):
    """Group 512px patch coordinates into tile-sized regions."""
    tile_origins = set()
    for x, y in patch_coords:
        tile_x = int(x // tile_size) * tile_size
        tile_y = int(y // tile_size) * tile_size
        tile_origins.add((tile_x, tile_y))
    return sorted(tile_origins)


def read_tile(slide, tile_x, tile_y, tile_size):
    """Read a tile from the WSI, padding edge tiles to full size."""
    slide_w, slide_h = slide.dimensions
    read_w = min(tile_size, slide_w - tile_x)
    read_h = min(tile_size, slide_h - tile_y)
    if read_w <= 0 or read_h <= 0:
        return None

    region = slide.read_region((tile_x, tile_y), 0, (read_w, read_h))
    tile_img = region.convert("RGB")

    if read_w < tile_size or read_h < tile_size:
        padded = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        padded.paste(tile_img, (0, 0))
        tile_img = padded

    return tile_img


def save_batch_results(results, tile_infos, labels_dir, tile_size):
    """Save YOLO results for a batch of tiles."""
    det_count = 0
    for result, (tile_x, tile_y) in zip(results, tile_infos):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        label_name = f"x{tile_x}_y{tile_y}_w{tile_size}_h{tile_size}.txt"
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, "w") as lf:
            for box in boxes:
                cls = int(box.cls.item())
                xywhn = box.xywhn[0]
                cx = xywhn[0].item()
                cy = xywhn[1].item()
                w = xywhn[2].item()
                h = xywhn[3].item()
                conf_val = box.conf.item()
                lf.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf_val:.6f}\n")
                det_count += 1

    return det_count


def prefetch_tiles(wsi_path, tiles, tile_size, batch_size, queue, num_workers):
    """Read tiles in parallel threads (each with own slide handle) into queue."""
    import threading

    # Thread-local slide handles to avoid contention
    _local = threading.local()

    def _read_one(args):
        tile_x, tile_y = args
        if not hasattr(_local, 'slide'):
            _local.slide = openslide.OpenSlide(wsi_path)
        img = read_tile(_local.slide, tile_x, tile_y, tile_size)
        if img is not None:
            return (img, (tile_x, tile_y))
        return None

    batch_imgs = []
    batch_infos = []

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for result in pool.map(_read_one, tiles):
            if result is None:
                continue
            img, info = result
            batch_imgs.append(img)
            batch_infos.append(info)

            if len(batch_imgs) >= batch_size:
                queue.put((list(batch_imgs), list(batch_infos)))
                batch_imgs.clear()
                batch_infos.clear()

    # Remaining tiles
    if batch_imgs:
        queue.put((list(batch_imgs), list(batch_infos)))

    # Sentinel
    queue.put(None)


def predict_wsi_onthefly(
    model,
    wsi_path,
    patches_h5_path,
    output_dir,
    tile_size=YOLO_TILE_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    conf=0.4,
    iou=0.5,
    imgsz=1024,
    device="cuda",
    half=True,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """Run batched YOLO on tiles with threaded prefetching for I/O overlap."""
    with h5py.File(patches_h5_path, "r") as f:
        if "coords" not in f:
            return 0
        coords = f["coords"][:]

    tiles = coords_to_tiles(coords, tile_size)
    if not tiles:
        return 0

    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    slide_name = os.path.basename(wsi_path)[:40]
    n_tiles = len(tiles)

    # Start prefetch thread — each worker opens its own slide handle
    queue = Queue(maxsize=3)  # Buffer up to 3 batches ahead
    prefetch_thread = Thread(
        target=prefetch_tiles,
        args=(wsi_path, tiles, tile_size, batch_size, queue, num_workers),
        daemon=True,
    )
    prefetch_thread.start()

    det_count = 0
    tiles_done = 0
    pbar = tqdm(total=n_tiles, desc=f"  {slide_name}", leave=False, unit="tile")

    while True:
        item = queue.get()
        if item is None:
            break

        batch_imgs, batch_infos = item
        results = model.predict(
            source=batch_imgs,
            save=False,
            save_txt=False,
            save_conf=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            verbose=False,
        )
        det_count += save_batch_results(results, batch_infos, labels_dir, tile_size)
        tiles_done += len(batch_imgs)
        pbar.update(len(batch_imgs))
        pbar.set_postfix(dets=det_count)

    pbar.close()
    prefetch_thread.join()
    return det_count


def predict_all_wsis(
    model_path,
    wsi_dir,
    patches_h5_dir,
    output_dir,
    tile_size=YOLO_TILE_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    conf=0.4,
    iou=0.5,
    imgsz=1024,
    device="cuda",
    half=True,
    max_gpu_fraction=1.0,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """Run YOLO on all WSIs with threaded I/O prefetching."""
    model = YOLO(model_path)

    if device == "cuda" and max_gpu_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(max_gpu_fraction)
        print(f"GPU memory limited to {max_gpu_fraction:.0%} of total")

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

    # Pre-filter: separate already-processed from pending
    pending = []
    skipped = 0
    for h5_file in h5_files:
        h5_stem = h5_file.replace("_patches.h5", "").replace(".h5", "")
        wsi_output_dir = os.path.join(output_dir, h5_stem)
        if os.path.exists(os.path.join(wsi_output_dir, "labels")):
            skipped += 1
        else:
            pending.append(h5_file)

    print(f"Found {len(h5_files)} H5 files, {len(wsi_lookup)} WSIs")
    print(f"Skipping {skipped} already processed, {len(pending)} remaining")
    print(f"batch_size={batch_size}, num_workers={num_workers}")

    total_dets = 0
    processed = 0
    failed = 0

    for h5_file in tqdm(pending, desc="YOLO prediction"):
        h5_stem = h5_file.replace("_patches.h5", "").replace(".h5", "")
        wsi_output_dir = os.path.join(output_dir, h5_stem)

        # Find matching WSI
        wsi_path = wsi_lookup.get(h5_stem)
        if wsi_path is None:
            matches = [k for k in wsi_lookup if k.startswith(h5_stem) or h5_stem.startswith(k)]
            if matches:
                wsi_path = wsi_lookup[matches[0]]

        if wsi_path is None:
            tqdm.write(f"  Warning: No WSI found for {h5_stem}, skipping")
            skipped += 1
            continue

        h5_path = os.path.join(patches_h5_dir, h5_file)

        try:
            n_dets = predict_wsi_onthefly(
                model=model,
                wsi_path=wsi_path,
                patches_h5_path=h5_path,
                output_dir=wsi_output_dir,
                tile_size=tile_size,
                batch_size=batch_size,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                half=half,
                num_workers=num_workers,
            )
            total_dets += n_dets
            processed += 1
        except Exception as e:
            tqdm.write(f"  ERROR processing {h5_stem}: {e}")
            traceback.print_exc()
            failed += 1
            torch.cuda.empty_cache()

    print(f"\nDone: {processed} processed, {skipped} skipped, {failed} failed, {total_dets} total detections")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO vessel detection on WSI tiles (on-the-fly, no tile extraction)"
    )
    parser.add_argument("--wsi_dir", required=True, help="Directory of WSI files")
    parser.add_argument("--patches_h5_dir", required=True,
                        help="Directory of trident patches H5 files (with coords)")
    parser.add_argument("--output_dir", required=True, help="Output predictions directory")
    parser.add_argument("--model", default=YOLO_WEIGHTS, help="YOLO weights path")
    parser.add_argument("--tile_size", type=int, default=YOLO_TILE_SIZE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Tiles per YOLO forward pass (default: 16)")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_half", action="store_true", help="Disable FP16 inference")
    parser.add_argument("--max_gpu_fraction", type=float, default=1.0,
                        help="Max fraction of GPU memory to use (default: 1.0)")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="Threads for tile reading (default: 8)")
    args = parser.parse_args()

    predict_all_wsis(
        model_path=args.model,
        wsi_dir=args.wsi_dir,
        patches_h5_dir=args.patches_h5_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        half=not args.no_half,
        max_gpu_fraction=args.max_gpu_fraction,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
