"""Convert LVI mask patches to COCO JSON and then to YOLO format.

Two-step pipeline:
    1. Binary mask patches → COCO JSON (patches2coco)
    2. COCO JSON + source images → YOLO format with data.yaml (coco2yolo)

Usage:
    # Full pipeline: masks → COCO → YOLO
    python -m pipeline.preprocess.prepare_annotations \
        --mask_dir /path/to/lvi_mask_patches \
        --images_dir /path/to/wsi_patches \
        --output_dir /path/to/yolo_dataset

    # COCO only
    python -m pipeline.preprocess.prepare_annotations \
        --mask_dir /path/to/masks --coco_only --coco_output /path/to/annotations.json
"""

import argparse
import glob
import json
import os
import shutil

import cv2
import yaml


# ── COCO conversion ──────────────────────────────────────────────────────────

CATEGORY_IDS = {"lvi": 1}


def masks_to_coco(mask_dir, output_json):
    """Convert binary mask patches to COCO JSON annotations.

    Args:
        mask_dir: Directory of binary LVI mask images (PNG).
        output_json: Output path for the COCO JSON file.
    """
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    for mask_path in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        filename = os.path.basename(mask_path)
        mask = cv2.imread(mask_path)
        if mask is None:
            print(f"Warning: Could not read {mask_path}, skipping")
            continue

        h, w, _ = mask.shape
        image_id += 1
        images.append({
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": filename,
        })

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            bbox = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            seg_points = contour.flatten().tolist()
            if len(seg_points) < 6:
                continue

            annotations.append({
                "iscrowd": 0,
                "id": annotation_id,
                "image_id": image_id,
                "category_id": CATEGORY_IDS["lvi"],
                "bbox": [float(b) for b in bbox],
                "area": float(area),
                "segmentation": [seg_points],
            })
            annotation_id += 1

    coco = {
        "info": {"description": "LVI COCO Dataset", "version": "1.0"},
        "licenses": [],
        "images": images,
        "categories": [
            {"id": v, "name": k, "supercategory": k}
            for k, v in CATEGORY_IDS.items()
        ],
        "annotations": annotations,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"COCO: {annotation_id} annotations across {len(images)} images → {output_json}")
    return output_json


# ── YOLO conversion ──────────────────────────────────────────────────────────

def coco_to_yolo(images_dir, coco_json, output_images_dir, output_labels_dir):
    """Convert COCO annotations + images to YOLO polygon format.

    Args:
        images_dir: Source directory with patch PNG images.
        coco_json: COCO JSON annotation file.
        output_images_dir: Destination for copied images.
        output_labels_dir: Destination for YOLO label txt files.
    """
    with open(coco_json) as f:
        data = json.load(f)

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Index annotations by image_id
    ann_by_img = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    img_lookup = {img["file_name"]: img for img in data["images"]}

    for filename in os.listdir(images_dir):
        if not filename.endswith(".png"):
            continue

        src = os.path.join(images_dir, filename)
        shutil.copy(src, os.path.join(output_images_dir, filename))

        img = img_lookup.get(filename)
        if img is None:
            continue

        img_anns = ann_by_img.get(img["id"], [])
        if not img_anns:
            continue

        label_path = os.path.join(
            output_labels_dir, os.path.splitext(filename)[0] + ".txt"
        )
        with open(label_path, "w") as lf:
            for ann in img_anns:
                cat_id = ann["category_id"] - 1  # YOLO is 0-indexed
                seg = ann.get("segmentation", [[]])
                if not seg or not seg[0]:
                    continue
                polygon = seg[0]
                normalized = []
                for i, coord in enumerate(polygon):
                    if i % 2 == 0:
                        normalized.append(f"{coord / img['width']:.6f}")
                    else:
                        normalized.append(f"{coord / img['height']:.6f}")
                lf.write(f"{cat_id} " + " ".join(normalized) + "\n")


def create_yolo_yaml(coco_json, output_yaml, train_path, val_path, test_path=None):
    """Create YOLO data.yaml from COCO categories."""
    with open(coco_json) as f:
        data = json.load(f)

    names = sorted(
        [cat["name"] for cat in data["categories"]],
        key=lambda n: next(c["id"] for c in data["categories"] if c["name"] == n),
    )

    yaml_data = {
        "names": names,
        "nc": len(names),
        "train": train_path,
        "val": val_path,
        "test": test_path or "",
    }

    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"YAML config → {output_yaml}")


def full_pipeline(
    mask_dir,
    images_train_dir,
    images_val_dir,
    train_json,
    val_json,
    output_dir,
):
    """Run the full annotation pipeline: masks → COCO → YOLO.

    Args:
        mask_dir: Directory with LVI binary mask PNGs (or separate train/val dirs).
        images_train_dir: WSI patch images for training.
        images_val_dir: WSI patch images for validation.
        train_json: Path to training COCO JSON (will be created if mask_dir given).
        val_json: Path to validation COCO JSON.
        output_dir: Root output for YOLO dataset.
    """
    # Convert to YOLO format
    coco_to_yolo(
        images_train_dir, train_json,
        os.path.join(output_dir, "train", "images"),
        os.path.join(output_dir, "train", "labels"),
    )
    coco_to_yolo(
        images_val_dir, val_json,
        os.path.join(output_dir, "valid", "images"),
        os.path.join(output_dir, "valid", "labels"),
    )

    create_yolo_yaml(
        train_json,
        os.path.join(output_dir, "data.yaml"),
        train_path=os.path.join(output_dir, "train"),
        val_path=os.path.join(output_dir, "valid"),
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO annotations from LVI masks")
    parser.add_argument("--mask_dir", help="Directory of binary LVI mask patches")
    parser.add_argument("--coco_only", action="store_true", help="Only generate COCO JSON")
    parser.add_argument("--coco_output", default="annotations.json")

    # For full pipeline
    parser.add_argument("--images_train", help="WSI patch images for training")
    parser.add_argument("--images_val", help="WSI patch images for validation")
    parser.add_argument("--train_json", help="Training COCO JSON")
    parser.add_argument("--val_json", help="Validation COCO JSON")
    parser.add_argument("--output_dir", help="Output YOLO dataset directory")

    args = parser.parse_args()

    if args.mask_dir and args.coco_only:
        masks_to_coco(args.mask_dir, args.coco_output)
    elif args.train_json and args.output_dir:
        full_pipeline(
            mask_dir=args.mask_dir,
            images_train_dir=args.images_train,
            images_val_dir=args.images_val,
            train_json=args.train_json,
            val_json=args.val_json,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
