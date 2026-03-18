"""LVI attention heatmap visualization.

Creates blended heatmap overlays on WSI showing:
  - ABMIL attention across ALL tissue patches (vessel-adjacent patches highlighted)
  - YOLO vessel detection bounding boxes (optional)
  - Top-k most attended patches saved as crops

Usage:
    python -m pipeline.visualize --patient_id 577-3428
    python -m pipeline.visualize --slide_name "577-3428_1.1 Bladder - 2024-09-18 18.06.19"
    python -m pipeline.visualize --patient_id 577-3428 --save_topk 5
"""

import argparse
import glob
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

from pipeline.config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR,
    EMORY_WSI_DIR, EMORY_YOLO_PREDICTIONS_PATH,
    EMORY_SLIDE_FEATS_PATH,
)
from pipeline.models import SurvivalModel
from pipeline.preprocess.filter_h5 import (
    parse_yolo_filename, parse_yolo_detections, nms,
)
from pipeline.train import set_seed

FILTERED_FEATS = "trident_outputs/emory_mibc/20x_512px_0px_overlap/filtered_conch_v15_LVI"
PATCH_SIZE_20X = 512   # patch size at 20x magnification
PATCH_SIZE_LVL0 = 1024  # patch size at level 0 (40x native → 512 at 20x = 1024 at 40x)
VIS_LEVEL = 4     # 16x downsample — good balance of detail vs speed


def find_slide_path(slide_name, wsi_dir=EMORY_WSI_DIR):
    """Find the WSI file for a given slide name."""
    for ext in (".ndpi", ".svs", ".tiff", ".tif"):
        path = os.path.join(wsi_dir, slide_name + ext)
        if os.path.exists(path):
            return path
    matches = glob.glob(os.path.join(wsi_dir, slide_name + ".*"))
    return matches[0] if matches else None


def load_yolo_boxes(slide_name, yolo_dir=EMORY_YOLO_PREDICTIONS_PATH):
    """Load and NMS YOLO detection boxes for a slide."""
    labels_dir = os.path.join(yolo_dir, slide_name, "labels")
    if not os.path.isdir(labels_dir):
        return np.empty((0, 4)), np.empty((0,))

    all_boxes, all_confs = [], []
    for txt_file in os.listdir(labels_dir):
        if not txt_file.endswith(".txt"):
            continue
        tile_coords = parse_yolo_filename(txt_file)
        boxes, confs = parse_yolo_detections(
            os.path.join(labels_dir, txt_file), tile_coords
        )
        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_confs.append(confs)

    if not all_boxes:
        return np.empty((0, 4)), np.empty((0,))

    all_boxes = np.concatenate(all_boxes)
    all_confs = np.concatenate(all_confs)
    keep = nms(all_boxes, all_confs, iou_threshold=0.5)
    return all_boxes[keep], all_confs[keep]


def run_model_on_features(model, features):
    """Run Joint model on features, return ABMIL attention weights."""
    model.eval()
    with torch.no_grad():
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        risk_score, lvi_logit, main_attn, _, _ = model(
            {"features": feat_tensor},
            return_raw_attention=True,
            return_patch_predictions=True,
        )

    # main_attn: (1, 1, n_heads, n_patches) — average heads, softmax
    attn_raw = main_attn.squeeze(0).squeeze(0)  # (n_heads, n_patches)
    attn_avg = attn_raw.mean(dim=0)             # (n_patches,)
    abmil_attn = torch.softmax(attn_avg, dim=0).cpu().numpy()

    risk = risk_score.item()
    lvi_prob = torch.sigmoid(lvi_logit).item()
    return abmil_attn, risk, lvi_prob


def create_overlay(scores, coords, patch_size_level0, scale, region_size):
    """Create pixel-level heatmap overlay (trident-style, vectorized)."""
    pw = int(np.ceil(patch_size_level0 * scale[0]))
    ph = int(np.ceil(patch_size_level0 * scale[1]))
    h, w = region_size[1], region_size[0]  # region_size is (w, h)

    coords_scaled = np.ceil(coords * scale).astype(int)

    overlay = np.full((h, w), np.nan, dtype=np.float32)
    counter = np.zeros((h, w), dtype=np.uint16)

    # Vectorized: compute all patch bounds at once, then fill
    x_starts = coords_scaled[:, 0]
    y_starts = coords_scaled[:, 1]
    x_ends = np.minimum(x_starts + pw, w)
    y_ends = np.minimum(y_starts + ph, h)

    for idx in range(len(scores)):
        ys, ye = y_starts[idx], y_ends[idx]
        xs, xe = x_starts[idx], x_ends[idx]
        if ys >= ye or xs >= xe:
            continue
        mask = counter[ys:ye, xs:xe] == 0
        overlay[ys:ye, xs:xe] = np.where(
            mask, scores[idx],
            (overlay[ys:ye, xs:xe] * counter[ys:ye, xs:xe] + scores[idx]) / (counter[ys:ye, xs:xe] + 1)
        )
        counter[ys:ye, xs:xe] += 1

    return overlay


def generate_heatmap(
    slide_name,
    model,
    output_dir,
    vis_level=VIS_LEVEL,
    cmap="coolwarm",
    alpha=0.45,
    blur_sigma=5.0,
    show_yolo_boxes=False,
    save_topk=0,
    yolo_box_color=(0, 255, 0),
    yolo_box_thickness=3,
):
    """Generate a blended heatmap overlay for one slide.

    All tissue patches get a score:
      - Filtered (vessel-adjacent) patches: ABMIL attention from model
      - Non-filtered patches: fixed low value (background tissue)
    Scores are rank-normalized, blended onto the WSI, with optional YOLO boxes.
    """

    # 1. Find and open WSI
    slide_path = find_slide_path(slide_name)
    if slide_path is None:
        print(f"  WSI not found for {slide_name}")
        return None

    wsi = openslide.OpenSlide(slide_path)
    level_dims = wsi.level_dimensions
    level_downsamples = wsi.level_downsamples

    # Clamp vis_level
    vis_level = min(vis_level, wsi.level_count - 1)
    downsample = level_downsamples[vis_level]
    scale = np.array([1.0 / downsample, 1.0 / downsample])
    region_size = tuple((np.array(level_dims[0]) * scale).astype(int))

    # 2. Load ALL patch coords from the original (unfiltered) slide H5
    orig_h5 = os.path.join(EMORY_SLIDE_FEATS_PATH, f"{slide_name}.h5")
    if not os.path.exists(orig_h5):
        print(f"  Original H5 not found: {orig_h5}")
        wsi.close()
        return None

    with h5py.File(orig_h5, "r") as f:
        all_coords = f["coords"][:]  # (N_total, 2)

    # 3. Load filtered (vessel-adjacent) features + coords
    filt_h5 = os.path.join(FILTERED_FEATS, f"{slide_name}.h5")
    has_filtered = os.path.exists(filt_h5)

    if has_filtered:
        with h5py.File(filt_h5, "r") as f:
            filt_features = f["features"][:]
            filt_coords = f["coords"][:]

        # 4. Run model on filtered features
        abmil_attn, risk, lvi_prob = run_model_on_features(model, filt_features)

        # 5. Map filtered attention back to all-patch indices
        # Build lookup: (x, y) → attention score
        filt_coord_set = {}
        for i, (fc) in enumerate(filt_coords):
            filt_coord_set[(int(fc[0]), int(fc[1]))] = abmil_attn[i]

        # Assign scores: filtered patches get attention, others get 0
        all_scores = np.zeros(len(all_coords), dtype=float)
        for i, coord in enumerate(all_coords):
            key = (int(coord[0]), int(coord[1]))
            if key in filt_coord_set:
                all_scores[i] = filt_coord_set[key]

        n_filtered = len(filt_coords)
    else:
        all_scores = np.zeros(len(all_coords), dtype=float)
        risk, lvi_prob = 0.0, 0.0
        n_filtered = 0

    # 6. Score normalization
    # Rank-normalize vessel patches independently so there's contrast within them,
    # then place them in the upper score range [0.5, 1.0].
    # Non-vessel tissue patches get a flat low value.
    non_vessel_mask = all_scores == 0

    vessel_scores = all_scores[~non_vessel_mask]
    if len(vessel_scores) > 0:
        vessel_ranked = rankdata(vessel_scores, method="average") / len(vessel_scores)
        all_scores[~non_vessel_mask] = 0.5 + 0.5 * vessel_ranked

    all_scores[non_vessel_mask] = 0.15  # cool blue for background tissue

    ranked = all_scores

    # 7. Create pixel-level overlay
    # Determine patch size at level 0 from WSI magnification
    native_mag = int(wsi.properties.get("openslide.objective-power", 40))
    patch_size_lvl0 = PATCH_SIZE_20X * (native_mag // 20)

    overlay = create_overlay(ranked, all_coords, patch_size_lvl0, scale, region_size)

    # 8. Smooth the overlay to reduce patchiness
    # Replace NaN temporarily for filtering
    nan_mask = np.isnan(overlay)
    overlay_smooth = overlay.copy()
    overlay_smooth[nan_mask] = 0
    overlay_smooth = gaussian_filter(overlay_smooth, sigma=blur_sigma)
    # Restore NaN outside tissue
    overlay_smooth[nan_mask] = np.nan

    # 9. Apply colormap
    cmap_fn = plt.get_cmap(cmap)
    overlay_colored = np.zeros((*overlay_smooth.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay_smooth)
    colored_valid = (cmap_fn(overlay_smooth[valid_mask]) * 255).astype(np.uint8)[:, :3]
    overlay_colored[valid_mask] = colored_valid

    # 10. Read WSI region and blend
    img = wsi.read_region((0, 0), vis_level, level_dims[vis_level]).convert("RGB")
    img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)

    blended = cv2.addWeighted(img, 1.0 - alpha, overlay_colored, alpha, 0)

    # Restore original tissue where there's no heatmap data (NaN regions)
    blended[nan_mask] = img[nan_mask]

    # 11. Draw YOLO boxes
    yolo_boxes = np.empty((0, 4))
    if show_yolo_boxes:
        yolo_boxes, _ = load_yolo_boxes(slide_name)
        if len(yolo_boxes) > 0:
            for box in yolo_boxes:
                x1, y1, x2, y2 = (box * scale[0]).astype(int)
                cv2.rectangle(blended, (x1, y1), (x2, y2),
                              yolo_box_color, yolo_box_thickness)

    # 12. Add info text
    info_text = f"Risk: {risk:.2f} | LVI Prob: {lvi_prob:.3f} | Vessel Patches: {n_filtered}/{len(all_coords)} | YOLO Boxes: {len(yolo_boxes)}"
    cv2.putText(blended, info_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(blended, info_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    # 13. Save
    os.makedirs(output_dir, exist_ok=True)
    clean_name = slide_name.replace(" ", "_").replace(":", "-")
    out_path = os.path.join(output_dir, f"heatmap_{clean_name}.png")
    Image.fromarray(blended).save(out_path)

    print(f"  {slide_name}")
    print(f"    Patches: {n_filtered}/{len(all_coords)} vessel-adjacent")
    print(f"    Risk: {risk:.3f}, LVI Prob: {lvi_prob:.3f}")
    print(f"    YOLO boxes: {len(yolo_boxes)}")
    print(f"    Saved: {out_path}")

    # 14. Save top-k patches
    if save_topk > 0 and has_filtered:
        topk_dir = os.path.join(output_dir, f"topk_{clean_name}")
        os.makedirs(topk_dir, exist_ok=True)
        topk_indices = np.argsort(abmil_attn)[-save_topk:][::-1]
        for rank, idx in enumerate(topk_indices):
            x, y = int(filt_coords[idx][0]), int(filt_coords[idx][1])
            patch = wsi.read_region((x, y), 0, (patch_size_lvl0, patch_size_lvl0)).convert("RGB")
            patch.save(os.path.join(
                topk_dir,
                f"rank{rank+1}_attn{abmil_attn[idx]:.4f}_x{x}_y{y}.png"
            ))
        print(f"    Top-{save_topk} patches saved to: {topk_dir}")

    wsi.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate LVI attention heatmaps")
    parser.add_argument("--slide_name", help="Exact slide name (without extension)")
    parser.add_argument("--patient_id", help="Patient ID (generates for all slides)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--vis_level", type=int, default=VIS_LEVEL)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap blend alpha")
    parser.add_argument("--blur", type=float, default=5.0, help="Gaussian blur sigma")
    parser.add_argument("--save_topk", type=int, default=0, help="Save top-k attended patches")
    parser.add_argument("--yolo_boxes", action="store_true", help="Show YOLO detection boxes")
    args = parser.parse_args()

    set_seed()

    # Load Joint model
    model = SurvivalModel().to(DEVICE)
    ckpt = os.path.join(CHECKPOINT_DIR, "joint_model_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Loaded model from {ckpt}")

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "heatmaps")

    if args.slide_name:
        generate_heatmap(
            args.slide_name, model, output_dir,
            vis_level=args.vis_level, cmap=args.cmap,
            alpha=args.alpha, blur_sigma=args.blur,
            show_yolo_boxes=args.yolo_boxes,
            save_topk=args.save_topk,
        )
    elif args.patient_id:
        pattern = os.path.join(FILTERED_FEATS, f"{args.patient_id}*.h5")
        slides = glob.glob(pattern)
        if not slides:
            print(f"No filtered features found for patient {args.patient_id}")
            return
        for h5_path in sorted(slides):
            slide_name = os.path.basename(h5_path).replace(".h5", "")
            generate_heatmap(
                slide_name, model, output_dir,
                vis_level=args.vis_level, cmap=args.cmap,
                alpha=args.alpha, blur_sigma=args.blur,
                show_yolo_boxes=args.yolo_boxes,
                save_topk=args.save_topk,
            )
    else:
        parser.error("Provide --slide_name or --patient_id")


if __name__ == "__main__":
    main()
