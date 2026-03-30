"""Multi-panel LVI attention heatmap visualization.

Creates a publication/presentation-quality figure showing the model's
decision layers side by side:
  A. H&E tissue overview
  B. YOLO vessel detections (green bounding boxes)
  C. Patch-level LVI probability (from MHA + MLP head)
  D. ABMIL slide-level attention (rank-normalized, vessel patches only)

Plus an optional top-k strip showing the highest-attention patch crops.

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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
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
PATCH_SIZE_20X = 512
PATCH_SIZE_LVL0 = 1024
VIS_LEVEL = 4


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    """Run Joint model on features, return ABMIL attention + patch LVI probs."""
    model.eval()
    with torch.no_grad():
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        risk_score, lvi_logit, main_attn, patch_lvi_logits, _ = model(
            {"features": feat_tensor},
            return_raw_attention=True,
            return_patch_predictions=True,
        )

    # ABMIL attention: (1, 1, n_heads, n_patches) → average heads → softmax
    attn_raw = main_attn.squeeze(0).squeeze(0)  # (n_heads, n_patches)
    attn_avg = attn_raw.mean(dim=0)
    abmil_attn = torch.softmax(attn_avg, dim=0).cpu().numpy()

    # Patch LVI probabilities
    patch_lvi_probs = torch.sigmoid(patch_lvi_logits.squeeze(0).squeeze(-1)).cpu().numpy()

    risk = risk_score.item()
    lvi_prob = torch.sigmoid(lvi_logit).item()
    return abmil_attn, patch_lvi_probs, risk, lvi_prob


def create_overlay(scores, coords, patch_size_level0, scale, region_size):
    """Create pixel-level heatmap overlay."""
    pw = int(np.ceil(patch_size_level0 * scale[0]))
    ph = int(np.ceil(patch_size_level0 * scale[1]))
    h, w = region_size[1], region_size[0]

    coords_scaled = np.ceil(coords * scale).astype(int)

    overlay = np.full((h, w), np.nan, dtype=np.float32)
    counter = np.zeros((h, w), dtype=np.uint16)

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
            (overlay[ys:ye, xs:xe] * counter[ys:ye, xs:xe] + scores[idx])
            / (counter[ys:ye, xs:xe] + 1)
        )
        counter[ys:ye, xs:xe] += 1

    return overlay


def blend_overlay(wsi_img, overlay, cmap_name, alpha=0.5, blur_sigma=3.0,
                  vmin=0.0, vmax=1.0):
    """Apply colormap to overlay and blend with WSI image."""
    nan_mask = np.isnan(overlay)
    overlay_smooth = overlay.copy()
    overlay_smooth[nan_mask] = 0
    if blur_sigma > 0:
        overlay_smooth = gaussian_filter(overlay_smooth, sigma=blur_sigma)
    overlay_smooth[nan_mask] = np.nan

    # Normalize to [0, 1] for colormap
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    normalized = norm(overlay_smooth)

    cmap_fn = plt.get_cmap(cmap_name)
    overlay_colored = np.zeros((*overlay_smooth.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay_smooth)
    colored_valid = (cmap_fn(normalized[valid_mask]) * 255).astype(np.uint8)[:, :3]
    overlay_colored[valid_mask] = colored_valid

    blended = cv2.addWeighted(wsi_img, 1.0 - alpha, overlay_colored, alpha, 0)
    blended[nan_mask] = wsi_img[nan_mask]
    return blended


# ── Multi-panel heatmap ─────────────────────────────────────────────────────

def generate_heatmap(
    slide_name,
    model,
    output_dir,
    vis_level=VIS_LEVEL,
    alpha=0.5,
    blur_sigma=3.0,
    save_topk=0,
    save_panels=False,
):
    """Generate a multi-panel heatmap figure for one slide.

    Panels:
      A. H&E overview
      B. YOLO vessel detections
      C. Patch-level LVI probability
      D. ABMIL attention (vessel patches)
    """

    # 1. Find and open WSI
    slide_path = find_slide_path(slide_name)
    if slide_path is None:
        print(f"  WSI not found for {slide_name}")
        return None

    wsi = openslide.OpenSlide(slide_path)
    vis_level = min(vis_level, wsi.level_count - 1)
    downsample = wsi.level_downsamples[vis_level]
    scale = np.array([1.0 / downsample, 1.0 / downsample])
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))

    native_mag = int(wsi.properties.get("openslide.objective-power", 40))
    patch_size_lvl0 = PATCH_SIZE_20X * (native_mag // 20)

    # 2. Read WSI thumbnail
    img = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    wsi_img = np.array(img)

    # 3. Load ALL patch coords
    orig_h5 = os.path.join(EMORY_SLIDE_FEATS_PATH, f"{slide_name}.h5")
    if not os.path.exists(orig_h5):
        print(f"  Original H5 not found: {orig_h5}")
        wsi.close()
        return None

    with h5py.File(orig_h5, "r") as f:
        all_coords = f["coords"][:]

    # 4. Load filtered features + coords
    filt_h5 = os.path.join(FILTERED_FEATS, f"{slide_name}.h5")
    has_filtered = os.path.exists(filt_h5)

    if has_filtered:
        with h5py.File(filt_h5, "r") as f:
            filt_features = f["features"][:]
            filt_coords = f["coords"][:]

        abmil_attn, patch_lvi_probs, risk, lvi_prob = run_model_on_features(
            model, filt_features
        )
        n_filtered = len(filt_coords)

        # Build coord → score lookups
        filt_attn_map = {}
        filt_lvi_map = {}
        for i, fc in enumerate(filt_coords):
            key = (int(fc[0]), int(fc[1]))
            filt_attn_map[key] = abmil_attn[i]
            filt_lvi_map[key] = patch_lvi_probs[i]
    else:
        risk, lvi_prob = 0.0, 0.0
        n_filtered = 0
        filt_attn_map = {}
        filt_lvi_map = {}
        abmil_attn = np.array([])
        filt_coords = np.empty((0, 2))

    # 5. Load YOLO boxes
    yolo_boxes, yolo_confs = load_yolo_boxes(slide_name)

    # 6. Build score arrays for all patches
    # ABMIL attention: vessel patches get rank-normalized [0.5, 1.0], others get 0.15
    attn_scores = np.zeros(len(all_coords), dtype=float)
    for i, coord in enumerate(all_coords):
        key = (int(coord[0]), int(coord[1]))
        if key in filt_attn_map:
            attn_scores[i] = filt_attn_map[key]

    vessel_mask = attn_scores > 0
    if vessel_mask.sum() > 0:
        vessel_ranked = rankdata(attn_scores[vessel_mask], method="average") / vessel_mask.sum()
        attn_scores[vessel_mask] = 0.5 + 0.5 * vessel_ranked
    attn_scores[~vessel_mask] = 0.15

    # LVI probability: vessel patches get their prob, others get 0
    lvi_scores = np.zeros(len(all_coords), dtype=float)
    for i, coord in enumerate(all_coords):
        key = (int(coord[0]), int(coord[1]))
        if key in filt_lvi_map:
            lvi_scores[i] = filt_lvi_map[key]

    # 7. Create overlays
    attn_overlay = create_overlay(attn_scores, all_coords, patch_size_lvl0, scale, region_size)
    lvi_overlay = create_overlay(lvi_scores, all_coords, patch_size_lvl0, scale, region_size)

    # 8. Blend overlays
    attn_blended = blend_overlay(wsi_img.copy(), attn_overlay, "coolwarm",
                                 alpha=alpha, blur_sigma=blur_sigma)
    lvi_blended = blend_overlay(wsi_img.copy(), lvi_overlay, "Reds",
                                alpha=alpha, blur_sigma=blur_sigma)

    # 9. YOLO boxes panel — filled semi-transparent rectangles for visibility
    yolo_panel = wsi_img.copy()
    if len(yolo_boxes) > 0:
        yolo_overlay = wsi_img.copy()
        for box in yolo_boxes:
            x1, y1, x2, y2 = (box * scale[0]).astype(int)
            cv2.rectangle(yolo_overlay, (x1, y1), (x2, y2), (0, 220, 0), -1)
            cv2.rectangle(yolo_overlay, (x1, y1), (x2, y2), (0, 160, 0), 2)
        yolo_panel = cv2.addWeighted(wsi_img, 0.65, yolo_overlay, 0.35, 0)

    # 10. Top-k patch crops
    topk_crops = []
    if save_topk > 0 and has_filtered:
        topk_indices = np.argsort(abmil_attn)[-save_topk:][::-1]
        for rank, idx in enumerate(topk_indices):
            x, y = int(filt_coords[idx][0]), int(filt_coords[idx][1])
            patch = wsi.read_region((x, y), 0, (patch_size_lvl0, patch_size_lvl0)).convert("RGB")
            topk_crops.append((np.array(patch), abmil_attn[idx], patch_lvi_probs[idx]))

            # Also save individually
            topk_dir = os.path.join(output_dir, f"topk_{slide_name.replace(' ', '_')}")
            os.makedirs(topk_dir, exist_ok=True)
            patch.save(os.path.join(
                topk_dir,
                f"rank{rank+1}_attn{abmil_attn[idx]:.4f}_lvi{patch_lvi_probs[idx]:.3f}_x{x}_y{y}.png"
            ))

    wsi.close()

    # ── Save individual panels ───────────────────────────────────────────────
    if save_panels:
        os.makedirs(output_dir, exist_ok=True)
        clean_name = slide_name.replace(" ", "_").replace(":", "-")
        panels_dir = os.path.join(output_dir, f"panels_{clean_name}")
        os.makedirs(panels_dir, exist_ok=True)

        # Raw panels as high-res images
        panel_images = {
            "A_HE_overview": wsi_img,
            "B_YOLO_detections": yolo_panel,
            "C_LVI_probability": lvi_blended,
            "D_ABMIL_attention": attn_blended,
        }
        for name, img_arr in panel_images.items():
            Image.fromarray(img_arr).save(
                os.path.join(panels_dir, f"{name}.png"), quality=95,
            )

        # Colorbars as standalone images
        for cmap_name, label, fname in [
            ("Reds", "Patch LVI Probability", "colorbar_lvi"),
            ("coolwarm", "ABMIL Attention", "colorbar_attention"),
        ]:
            fig_cb, ax_cb = plt.subplots(figsize=(1.2, 4), facecolor="white")
            sm = plt.cm.ScalarMappable(
                cmap=plt.get_cmap(cmap_name), norm=Normalize(vmin=0, vmax=1),
            )
            sm.set_array([])
            cbar = fig_cb.colorbar(sm, cax=ax_cb)
            cbar.set_label(label, fontsize=12, fontweight="bold")
            cbar.ax.tick_params(labelsize=10)
            fig_cb.savefig(
                os.path.join(panels_dir, f"{fname}.png"),
                dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1,
            )
            plt.close(fig_cb)

        # Top-k crops as individual high-res images
        for i, (crop, attn_val, lvi_val) in enumerate(topk_crops):
            Image.fromarray(crop).save(
                os.path.join(panels_dir, f"topk_{i+1}_attn{attn_val:.3f}_lvi{lvi_val:.2f}.png"),
                quality=95,
            )

        print(f"    Panels saved to: {panels_dir}")

    # ── Build the figure ─────────────────────────────────────────────────────

    has_topk = len(topk_crops) > 0
    height_ratios = [1, 0.25] if has_topk else [1]
    nrows = 2 if has_topk else 1

    fig = plt.figure(figsize=(24, 7 * nrows), facecolor="white", dpi=150)
    gs = gridspec.GridSpec(nrows, 4, figure=fig, height_ratios=height_ratios,
                           hspace=0.2, wspace=0.08)

    panels = [
        (wsi_img, "A. H&E Overview", None, None),
        (yolo_panel, f"B. YOLO Detections (n={len(yolo_boxes)})", None, None),
        (lvi_blended, "C. Patch LVI Probability", "Reds", (0, 1)),
        (attn_blended, "D. ABMIL Attention", "coolwarm", (0, 1)),
    ]

    panel_axes = []
    for col, (img_data, title, cmap_name, clim) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img_data)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")

        if cmap_name and clim:
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name),
                                       norm=Normalize(vmin=clim[0], vmax=clim[1]))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.5,
                                aspect=20)
            cbar.ax.tick_params(labelsize=7)
            cbar.outline.set_linewidth(0.5)

        panel_axes.append(ax)

    # Top-k strip
    if has_topk:
        n_crops = len(topk_crops)
        for i, (crop, attn_val, lvi_val) in enumerate(topk_crops):
            ax = fig.add_subplot(gs[1, :], frameon=False)
            ax.set_visible(False)

        # Use a sub-gridspec for the crops
        gs_crops = gridspec.GridSpecFromSubplotSpec(
            1, n_crops, subplot_spec=gs[1, :], wspace=0.08
        )
        for i, (crop, attn_val, lvi_val) in enumerate(topk_crops):
            ax = fig.add_subplot(gs_crops[0, i])
            ax.imshow(crop)
            border_color = "#e74c3c" if lvi_val > 0.5 else "#3498db"
            ax.set_title(f"#{i+1}  attn={attn_val:.3f}  LVI={lvi_val:.2f}",
                         fontsize=10, fontweight="bold", pad=6, color="#2c3e50")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_color(border_color)

    # Suptitle with patient info
    patient_id = slide_name.split("_")[0] if "_" in slide_name else slide_name
    risk_label = "High" if risk > -2.7603 else "Low"
    lvi_label = f"LVI Prob: {lvi_prob:.3f}"

    fig.suptitle(
        f"Patient {patient_id}   |   Risk: {risk:.2f} ({risk_label})   |   "
        f"{lvi_label}   |   Vessel Patches: {n_filtered}/{len(all_coords)}   |   "
        f"YOLO Boxes: {len(yolo_boxes)}",
        fontsize=14, fontweight="bold", y=1.02, color="#2c3e50",
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    clean_name = slide_name.replace(" ", "_").replace(":", "-")
    out_path = os.path.join(output_dir, f"heatmap_{clean_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close()

    print(f"  {slide_name}")
    print(f"    Patches: {n_filtered}/{len(all_coords)} vessel-adjacent")
    print(f"    Risk: {risk:.3f} ({risk_label}), LVI Prob: {lvi_prob:.3f}")
    print(f"    YOLO boxes: {len(yolo_boxes)}")
    print(f"    Saved: {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate multi-panel LVI attention heatmaps")
    parser.add_argument("--slide_name", help="Exact slide name (without extension)")
    parser.add_argument("--patient_id", help="Patient ID (generates for all slides)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--vis_level", type=int, default=VIS_LEVEL)
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap blend alpha")
    parser.add_argument("--blur", type=float, default=3.0, help="Gaussian blur sigma")
    parser.add_argument("--save_topk", type=int, default=5, help="Save top-k attended patches (0 to skip)")
    parser.add_argument("--save_panels", action="store_true", help="Save each panel as a standalone high-res image")
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
            vis_level=args.vis_level, alpha=args.alpha,
            blur_sigma=args.blur, save_topk=args.save_topk,
            save_panels=args.save_panels,
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
                vis_level=args.vis_level, alpha=args.alpha,
                blur_sigma=args.blur, save_topk=args.save_topk,
                save_panels=args.save_panels,
            )
    else:
        parser.error("Provide --slide_name or --patient_id")


if __name__ == "__main__":
    main()
