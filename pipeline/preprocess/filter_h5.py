"""Filter H5 feature files by YOLO vessel detections (bbox-level).

For each WSI, parses actual detection bounding boxes from YOLO txt files,
applies global NMS across tiles, then keeps only 512px patches whose center
falls within a detection bbox. Stores the max overlapping confidence per patch.

Output H5 schema: features (N, 768), coords (N, 2), confidences (N,)

Usage:
    python -m pipeline.preprocess.filter_h5 \
        --yolo_dir /path/to/yolo_predictions \
        --h5_dir /path/to/features_conch_v15 \
        --output_dir /path/to/filtered_h5
"""

import argparse
import os

import h5py
import numpy as np


def parse_yolo_filename(filename):
    """Parse YOLO prediction filename to extract tile position.

    Example filename: 'x14336_y30720_w2048_h2048.txt'
    Returns dict with keys: x, y, w, h (all in WSI absolute pixels).
    """
    parts = filename.replace(".txt", "").split("_")
    coords = {}
    for part in parts:
        if part.startswith("x"):
            coords["x"] = int(part[1:])
        elif part.startswith("y"):
            coords["y"] = int(part[1:])
        elif part.startswith("w"):
            coords["w"] = int(part[1:])
        elif part.startswith("h"):
            coords["h"] = int(part[1:])
    return coords


def parse_yolo_detections(txt_path, tile_coords):
    """Parse YOLO detection lines into absolute WSI bounding boxes.

    Each line: class cx cy w h [conf]
    All values normalized to tile size. We convert to absolute WSI coords.

    Returns:
        boxes: (M, 4) array of [x1, y1, x2, y2] in WSI absolute pixels.
        confidences: (M,) array of confidence scores.
    """
    tile_x, tile_y = tile_coords["x"], tile_coords["y"]
    tile_w, tile_h = tile_coords["w"], tile_coords["h"]

    boxes = []
    confidences = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # class cx cy w h [conf]
            cx_norm = float(parts[1])
            cy_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 1.0

            # Convert normalized tile coords to absolute WSI coords
            cx_abs = tile_x + cx_norm * tile_w
            cy_abs = tile_y + cy_norm * tile_h
            w_abs = w_norm * tile_w
            h_abs = h_norm * tile_h

            x1 = cx_abs - w_abs / 2
            y1 = cy_abs - h_abs / 2
            x2 = cx_abs + w_abs / 2
            y2 = cy_abs + h_abs / 2

            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)

    if not boxes:
        return np.empty((0, 4)), np.empty((0,))

    return np.array(boxes), np.array(confidences)


def nms(boxes, confidences, iou_threshold=0.5):
    """Non-maximum suppression on bounding boxes.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
        confidences: (N,) confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        keep: indices of kept boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confidences.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def patch_center_in_boxes(patch_coords, boxes, confidences, patch_size=512):
    """Check if patch centers fall within any detection box and track confidence.

    Args:
        patch_coords: (N, 2) array of patch top-left (x, y) coordinates.
        boxes: (M, 4) array of detection [x1, y1, x2, y2].
        confidences: (M,) confidence scores for each box.
        patch_size: Size of each patch (default 512px).

    Returns:
        mask: (N,) boolean array — True if patch center is in any box.
        max_conf: (N,) max confidence of overlapping boxes (0.0 if no overlap).
    """
    if len(boxes) == 0:
        return np.zeros(len(patch_coords), dtype=bool), np.zeros(len(patch_coords), dtype=np.float32)

    # Compute patch centers: (N,)
    cx = patch_coords[:, 0] + patch_size / 2
    cy = patch_coords[:, 1] + patch_size / 2

    # Broadcast: (N, 1) vs (1, M) → (N, M)
    in_x = (cx[:, None] >= boxes[None, :, 0]) & (cx[:, None] <= boxes[None, :, 2])
    in_y = (cy[:, None] >= boxes[None, :, 1]) & (cy[:, None] <= boxes[None, :, 3])
    overlap = in_x & in_y  # (N, M)

    mask = overlap.any(axis=1)  # (N,)

    # Max confidence among overlapping boxes per patch
    conf_matrix = np.where(overlap, confidences[None, :], 0.0)  # (N, M)
    max_conf = conf_matrix.max(axis=1).astype(np.float32)  # (N,)

    return mask, max_conf


def filter_h5_by_yolo(yolo_dir, h5_dir, output_dir, nms_iou=0.5, patch_size=512):
    """Filter H5 features to keep only patches overlapping actual vessel detections.

    Args:
        yolo_dir: Root of YOLO predictions (WSI_NAME/labels/*.txt).
        h5_dir: Directory of original H5 feature files (WSI_NAME.h5).
        output_dir: Where to write filtered H5 files.
        nms_iou: IoU threshold for global NMS across tiles.
        patch_size: Feature patch size in pixels (default 512).
    """
    os.makedirs(output_dir, exist_ok=True)

    for wsi_name in sorted(os.listdir(yolo_dir)):
        labels_dir = os.path.join(yolo_dir, wsi_name, "labels")
        if not os.path.isdir(labels_dir):
            continue

        h5_path = os.path.join(h5_dir, f"{wsi_name}.h5")
        if not os.path.exists(h5_path):
            print(f"  Warning: H5 file not found for {wsi_name}, skipping")
            continue

        with h5py.File(h5_path, "r") as f:
            if "coords" not in f or "features" not in f:
                print(f"  Error: Missing coords/features in {h5_path}, skipping")
                continue
            all_coords = f["coords"][:]
            all_features = f["features"][:]

        # Collect all detections across tiles
        all_boxes = []
        all_confs = []
        txt_files = [fn for fn in os.listdir(labels_dir) if fn.endswith(".txt")]

        for txt_file in txt_files:
            tile_coords = parse_yolo_filename(txt_file)
            txt_path = os.path.join(labels_dir, txt_file)
            boxes, confs = parse_yolo_detections(txt_path, tile_coords)
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_confs.append(confs)

        if not all_boxes:
            print(f"  {wsi_name}: no detections found, skipping")
            continue

        # Global NMS across all tiles
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_confs = np.concatenate(all_confs, axis=0)

        keep_idx = nms(all_boxes, all_confs, iou_threshold=nms_iou)
        nms_boxes = all_boxes[keep_idx]
        nms_confs = all_confs[keep_idx]

        # Filter patches: keep those whose center falls in a detection box
        mask, max_conf = patch_center_in_boxes(
            all_coords, nms_boxes, nms_confs, patch_size=patch_size
        )

        if not mask.any():
            print(f"  {wsi_name}: no patches overlap detections ({len(nms_boxes)} boxes, {len(all_coords)} patches)")
            continue

        filtered_features = all_features[mask]
        filtered_coords = all_coords[mask]
        filtered_confs = max_conf[mask]

        out_path = os.path.join(output_dir, f"{wsi_name}.h5")
        with h5py.File(out_path, "w") as out_f:
            out_f.create_dataset("features", data=filtered_features)
            out_f.create_dataset("coords", data=filtered_coords)
            out_f.create_dataset("confidences", data=filtered_confs)

        print(
            f"  {wsi_name}: {mask.sum()}/{len(all_coords)} patches kept "
            f"({len(nms_boxes)} detections after NMS)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Filter H5 features by YOLO detections (bbox-level)"
    )
    parser.add_argument("--yolo_dir", required=True, help="YOLO predictions root directory")
    parser.add_argument("--h5_dir", required=True, help="Original H5 features directory")
    parser.add_argument("--output_dir", required=True, help="Output filtered H5 directory")
    parser.add_argument("--nms_iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--patch_size", type=int, default=512, help="Feature patch size in pixels")
    args = parser.parse_args()

    filter_h5_by_yolo(
        args.yolo_dir, args.h5_dir, args.output_dir,
        nms_iou=args.nms_iou, patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
