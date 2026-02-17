"""Convert QuPath GeoJSON annotations to YOLO training data.

Reads QuPath-exported GeoJSON files containing LVI region polygons,
extracts aligned image patches + binary masks from WSIs, then converts
to COCO and YOLO format for YOLO retraining.

Pipeline: QuPath GeoJSON → extract patches/masks → COCO JSON → YOLO format

Usage:
    python -m pipeline.preprocess.qupath_to_yolo \
        --geojson_dir /path/to/qupath_exports \
        --wsi_dir /path/to/wsis \
        --output_dir /path/to/yolo_dataset \
        --patch_size 1024 \
        --val_fraction 0.2
"""

import argparse
import json
import os

import cv2
import numpy as np

from pipeline.preprocess.prepare_annotations import masks_to_coco, coco_to_yolo, create_yolo_yaml


def load_geojson_annotations(geojson_path):
    """Load QuPath GeoJSON and extract LVI annotation polygons.

    QuPath exports annotations as GeoJSON FeatureCollections. Each Feature
    has a geometry (Polygon/MultiPolygon) and properties (classification).

    Returns:
        List of dicts with keys: polygon (Nx2 array), classification (str).
    """
    with open(geojson_path) as f:
        data = json.load(f)

    annotations = []
    features = data.get("features", data if isinstance(data, list) else [])

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})

        # Get classification name
        classification = props.get("classification", {})
        if isinstance(classification, dict):
            class_name = classification.get("name", "unknown")
        else:
            class_name = str(classification) if classification else "unknown"

        geom_type = geom.get("type", "")
        coords_list = geom.get("coordinates", [])

        if geom_type == "Polygon":
            # First ring is exterior boundary
            if coords_list:
                polygon = np.array(coords_list[0], dtype=np.float64)
                annotations.append({"polygon": polygon, "classification": class_name})
        elif geom_type == "MultiPolygon":
            for poly_coords in coords_list:
                if poly_coords:
                    polygon = np.array(poly_coords[0], dtype=np.float64)
                    annotations.append({"polygon": polygon, "classification": class_name})

    return annotations


def extract_patches_from_annotations(
    wsi_path, annotations, patch_size=1024, context_margin=256,
):
    """Extract image patches and binary masks around annotation regions.

    For each annotation polygon, extracts a patch centered on the polygon's
    bounding box (with context margin). Generates a binary mask of the
    annotation within the patch.

    Args:
        wsi_path: Path to WSI file.
        annotations: List of annotation dicts from load_geojson_annotations.
        patch_size: Output patch size in pixels.
        context_margin: Extra context around the annotation bbox.

    Returns:
        List of (image_patch, mask_patch, patch_info) tuples.
    """
    try:
        import openslide
    except ImportError:
        raise ImportError("openslide-python is required for WSI reading: pip install openslide-python")

    slide = openslide.OpenSlide(wsi_path)
    patches = []

    for ann in annotations:
        polygon = ann["polygon"]

        # Bounding box of the polygon
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)

        # Center of annotation
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        # Extract region centered on annotation
        read_x = int(cx - patch_size / 2)
        read_y = int(cy - patch_size / 2)

        # Clamp to slide bounds
        read_x = max(0, min(read_x, slide.dimensions[0] - patch_size))
        read_y = max(0, min(read_y, slide.dimensions[1] - patch_size))

        region = slide.read_region((read_x, read_y), 0, (patch_size, patch_size))
        image = np.array(region.convert("RGB"))

        # Create binary mask: polygon coords relative to patch origin
        mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
        local_polygon = polygon - np.array([read_x, read_y])
        local_polygon = local_polygon.astype(np.int32)
        cv2.fillPoly(mask, [local_polygon], 255)

        patches.append((image, mask, {
            "x": read_x, "y": read_y,
            "classification": ann["classification"],
        }))

    slide.close()
    return patches


def qupath_to_yolo(
    geojson_dir,
    wsi_dir,
    output_dir,
    patch_size=1024,
    val_fraction=0.2,
    lvi_classes=None,
):
    """Full pipeline: QuPath GeoJSON → YOLO training data.

    Args:
        geojson_dir: Directory of QuPath GeoJSON exports (one per WSI).
        wsi_dir: Directory of WSI files.
        output_dir: Output YOLO dataset directory.
        patch_size: Patch size for extraction.
        val_fraction: Fraction of patches for validation.
        lvi_classes: List of class names to include (default: ["LVI", "lvi"]).
    """
    if lvi_classes is None:
        lvi_classes = ["LVI", "lvi", "Lymphovascular invasion", "lymphovascular invasion"]

    # Temp directories for intermediate outputs
    patches_dir = os.path.join(output_dir, "_temp_patches")
    masks_dir = os.path.join(output_dir, "_temp_masks")
    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Step 1: Extract patches and masks from each WSI
    print("Step 1: Extracting patches and masks from WSIs")
    patch_count = 0
    geojson_files = [f for f in os.listdir(geojson_dir) if f.endswith(".geojson")]

    for geojson_file in sorted(geojson_files):
        wsi_name = os.path.splitext(geojson_file)[0]
        geojson_path = os.path.join(geojson_dir, geojson_file)

        # Find matching WSI
        wsi_path = None
        for ext in (".svs", ".ndpi", ".tiff", ".tif", ".mrxs"):
            candidate = os.path.join(wsi_dir, wsi_name + ext)
            if os.path.exists(candidate):
                wsi_path = candidate
                break

        if wsi_path is None:
            print(f"  Warning: No WSI found for {wsi_name}, skipping")
            continue

        annotations = load_geojson_annotations(geojson_path)

        # Filter to LVI annotations only
        lvi_anns = [a for a in annotations if a["classification"] in lvi_classes]
        if not lvi_anns:
            print(f"  {wsi_name}: no LVI annotations found ({len(annotations)} total)")
            continue

        print(f"  {wsi_name}: {len(lvi_anns)} LVI annotations")

        patches = extract_patches_from_annotations(wsi_path, lvi_anns, patch_size)

        for img, mask, info in patches:
            fname = f"{wsi_name}_x{info['x']}_y{info['y']}.png"
            cv2.imwrite(os.path.join(patches_dir, fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(masks_dir, fname), mask)
            patch_count += 1

    if patch_count == 0:
        print("No patches extracted. Check GeoJSON files and WSI paths.")
        return

    print(f"\nExtracted {patch_count} patches total")

    # Step 2: Split into train/val
    all_fnames = sorted(os.listdir(patches_dir))
    np.random.seed(42)
    np.random.shuffle(all_fnames)
    n_val = max(1, int(len(all_fnames) * val_fraction))
    val_fnames = set(all_fnames[:n_val])
    train_fnames = set(all_fnames[n_val:])

    # Create split directories for masks
    train_mask_dir = os.path.join(output_dir, "_temp_masks_train")
    val_mask_dir = os.path.join(output_dir, "_temp_masks_val")
    train_img_dir = os.path.join(output_dir, "_temp_imgs_train")
    val_img_dir = os.path.join(output_dir, "_temp_imgs_val")
    for d in (train_mask_dir, val_mask_dir, train_img_dir, val_img_dir):
        os.makedirs(d, exist_ok=True)

    import shutil
    for fname in all_fnames:
        if fname in val_fnames:
            shutil.copy(os.path.join(masks_dir, fname), os.path.join(val_mask_dir, fname))
            shutil.copy(os.path.join(patches_dir, fname), os.path.join(val_img_dir, fname))
        else:
            shutil.copy(os.path.join(masks_dir, fname), os.path.join(train_mask_dir, fname))
            shutil.copy(os.path.join(patches_dir, fname), os.path.join(train_img_dir, fname))

    # Step 3: Masks → COCO JSON
    print("\nStep 2: Converting masks to COCO JSON")
    train_json = os.path.join(output_dir, "train_coco.json")
    val_json = os.path.join(output_dir, "val_coco.json")
    masks_to_coco(train_mask_dir, train_json)
    masks_to_coco(val_mask_dir, val_json)

    # Step 4: COCO → YOLO format
    print("\nStep 3: Converting COCO to YOLO format")
    coco_to_yolo(
        train_img_dir, train_json,
        os.path.join(output_dir, "train", "images"),
        os.path.join(output_dir, "train", "labels"),
    )
    coco_to_yolo(
        val_img_dir, val_json,
        os.path.join(output_dir, "valid", "images"),
        os.path.join(output_dir, "valid", "labels"),
    )

    # Step 5: Generate data.yaml
    create_yolo_yaml(
        train_json,
        os.path.join(output_dir, "data.yaml"),
        train_path=os.path.join(output_dir, "train"),
        val_path=os.path.join(output_dir, "valid"),
    )

    # Cleanup temp directories
    import shutil
    for d in (patches_dir, masks_dir, train_mask_dir, val_mask_dir, train_img_dir, val_img_dir):
        shutil.rmtree(d, ignore_errors=True)
    for f in (train_json, val_json):
        if os.path.exists(f):
            os.remove(f)

    print(f"\nYOLO dataset ready at {output_dir}")
    print(f"  Train: {len(train_fnames)} images")
    print(f"  Val: {len(val_fnames)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Convert QuPath GeoJSON annotations to YOLO training data"
    )
    parser.add_argument("--geojson_dir", required=True,
                        help="Directory of QuPath GeoJSON exports")
    parser.add_argument("--wsi_dir", required=True,
                        help="Directory of WSI files")
    parser.add_argument("--output_dir", required=True,
                        help="Output YOLO dataset directory")
    parser.add_argument("--patch_size", type=int, default=1024,
                        help="Patch size for extraction (default: 1024)")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of patches for validation")
    args = parser.parse_args()

    qupath_to_yolo(
        geojson_dir=args.geojson_dir,
        wsi_dir=args.wsi_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
