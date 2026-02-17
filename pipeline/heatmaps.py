"""Attention heatmap generation using trident.

Usage:
    python -m pipeline.heatmaps --slide /path/to/slide.svs --model Joint
    python -m pipeline.heatmaps --patient_ids P001 P002 --model Joint --slides_dir /path/to/slides
"""

import argparse
import os

import torch
from trident import OpenSlideWSI, visualize_heatmap
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory as patch_encoder_factory

from pipeline.config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR,
    HEATMAP_TARGET_MAG, HEATMAP_PATCH_SIZE, HEATMAP_OVERLAP,
    HEATMAP_ENCODER, HEATMAP_SEGMENTATION,
)
from pipeline.models import MODELS, CHECKPOINT_NAMES
from pipeline.train import set_seed


def generate_heatmap(model, slide_path, output_dir, slide_name=None):
    """Generate an attention heatmap for a single slide.

    Steps:
        1. Load WSI via OpenSlideWSI
        2. Segment tissue
        3. Extract patch coordinates
        4. Extract patch features (CONCH v1.5)
        5. Run model forward pass with return_raw_attention=True
        6. Visualize heatmap using trident
    """
    if slide_name is None:
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load WSI
    print(f"  Loading WSI: {slide_path}")
    slide = OpenSlideWSI(slide_path, lazy_init=False)

    # 2. Segment tissue
    print("  Segmenting tissue...")
    seg_model = segmentation_model_factory(HEATMAP_SEGMENTATION)
    slide.segment_tissue(seg_model, target_mag=HEATMAP_TARGET_MAG)

    # 3. Extract coordinates
    print("  Extracting patch coordinates...")
    coords = slide.extract_tissue_coords(
        target_mag=HEATMAP_TARGET_MAG,
        patch_size=HEATMAP_PATCH_SIZE,
        overlap=HEATMAP_OVERLAP,
    )

    # 4. Extract features
    print("  Extracting patch features...")
    encoder = patch_encoder_factory(HEATMAP_ENCODER)
    features = slide.extract_patch_features(encoder, coords)

    # 5. Model inference
    print("  Running model inference...")
    model.eval()
    with torch.no_grad():
        feat_tensor = features.unsqueeze(0).to(DEVICE)
        risk_score, lvi_logit, attention = model(
            {"features": feat_tensor}, return_raw_attention=True
        )

    attention_np = attention.squeeze().cpu().numpy()

    print(f"  Risk score: {risk_score.item():.4f}")
    print(f"  LVI prob:   {torch.sigmoid(lvi_logit).item():.4f}")

    # 6. Visualize
    print("  Generating heatmap...")
    output_path = os.path.join(output_dir, f"{slide_name}_heatmap.png")
    visualize_heatmap(
        wsi=slide,
        scores=attention_np,
        coords=coords,
        output_path=output_path,
    )

    print(f"  Saved: {output_path}")
    return {
        "slide": slide_name,
        "risk_score": risk_score.item(),
        "lvi_prob": torch.sigmoid(lvi_logit).item(),
        "heatmap_path": output_path,
    }


def batch_generate_heatmaps(
    model, slides_dir, patient_ids=None, output_dir=None, ext=".svs"
):
    """Generate heatmaps for multiple slides.

    Args:
        model: Loaded model in eval mode.
        slides_dir: Directory containing WSI files.
        patient_ids: Optional list of patient IDs to process.
                     If None, processes all slides in slides_dir.
        output_dir: Where to save heatmaps. Defaults to results/heatmaps/.
        ext: Slide file extension (default: .svs).
    """
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "heatmaps")

    # Find slides
    all_slides = [
        f for f in os.listdir(slides_dir) if f.endswith(ext)
    ]

    if patient_ids is not None:
        slides = [
            s for s in all_slides
            if any(pid in s for pid in patient_ids)
        ]
    else:
        slides = all_slides

    print(f"Generating heatmaps for {len(slides)} slides...")
    results = []

    for slide_file in slides:
        slide_path = os.path.join(slides_dir, slide_file)
        try:
            res = generate_heatmap(model, slide_path, output_dir)
            results.append(res)
        except Exception as e:
            print(f"  ERROR processing {slide_file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate attention heatmaps")
    parser.add_argument("--slide", help="Path to a single WSI file")
    parser.add_argument("--slides_dir", help="Directory of WSI files")
    parser.add_argument("--patient_ids", nargs="+", help="Patient IDs to process")
    parser.add_argument(
        "--model", default="Joint",
        choices=["Joint", "Survival-Only", "LVI-Only"],
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    set_seed()

    # Load model
    model_cls = MODELS[args.model]
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAMES[args.model])
    model = model_cls().to(DEVICE)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "heatmaps")

    if args.slide:
        generate_heatmap(model, args.slide, output_dir)
    elif args.slides_dir:
        batch_generate_heatmaps(
            model, args.slides_dir,
            patient_ids=args.patient_ids,
            output_dir=output_dir,
        )
    else:
        parser.error("Provide either --slide or --slides_dir")


if __name__ == "__main__":
    main()
