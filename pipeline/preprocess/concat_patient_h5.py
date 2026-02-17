"""Merge per-slide H5 feature files into per-patient H5 files.

Given a directory of slide-level H5s (named SLIDE_ID.h5) and a mapping from
patient_id to slide_ids, concatenates features, coords, and confidences across
all slides for each patient. Adds a slide_index column to coords.

Usage:
    python -m pipeline.preprocess.concat_patient_h5 \
        --input_dir /path/to/slide_features \
        --output_dir /path/to/patient_features \
        --clinical_csv /path/to/clinical.csv
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd


def find_slides_for_patient(patient_id, input_dir):
    """Find all H5 files in input_dir that belong to a patient.

    Matches by checking if the filename starts with the patient_id.
    """
    matches = []
    for fname in os.listdir(input_dir):
        if not fname.endswith(".h5"):
            continue
        # Check prefix match (patient_id is typically a prefix of slide_id)
        slide_id = fname.replace(".h5", "")
        if slide_id.startswith(patient_id) or slide_id == patient_id:
            matches.append(os.path.join(input_dir, fname))
    return sorted(matches)


def concat_patient_h5(input_dir, output_dir, clinical_csv=None, patient_ids=None):
    """Concatenate per-slide H5s into per-patient H5s.

    Args:
        input_dir: Directory of slide-level H5 files.
        output_dir: Directory to write patient-level H5 files.
        clinical_csv: CSV with patient_id column. Used to get the list of patients.
        patient_ids: Explicit list of patient IDs. Overrides clinical_csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    if patient_ids is None and clinical_csv is not None:
        df = pd.read_csv(clinical_csv)
        patient_ids = df["patient_id"].unique().tolist()
    elif patient_ids is None:
        # Infer patients from filenames by grouping on common prefixes
        print("No patient list provided. Processing all H5 files as-is.")
        return

    processed = 0
    skipped = 0

    for pid in patient_ids:
        out_path = os.path.join(output_dir, f"{pid}.h5")
        if os.path.exists(out_path):
            skipped += 1
            continue

        slide_paths = find_slides_for_patient(pid, input_dir)

        if not slide_paths:
            print(f"  Warning: No slides found for {pid}")
            skipped += 1
            continue

        all_features = []
        all_coords = []
        all_confidences = []
        has_confidences = False

        for slide_idx, sp in enumerate(slide_paths):
            with h5py.File(sp, "r") as f:
                if "features" not in f:
                    continue
                feats = f["features"][:]
                all_features.append(feats)
                n_patches = feats.shape[0]

                if "coords" in f:
                    coords = f["coords"][:]  # (N, 2)
                    # Add slide index as 3rd column
                    slide_col = np.full((n_patches, 1), slide_idx, dtype=coords.dtype)
                    coords_with_idx = np.hstack([coords, slide_col])
                    all_coords.append(coords_with_idx)

                if "confidences" in f:
                    has_confidences = True
                    all_confidences.append(f["confidences"][:])
                else:
                    # Fill with 1.0 (uniform) if this slide lacks confidences
                    all_confidences.append(np.ones(n_patches, dtype=np.float32))

        if not all_features:
            print(f"  Warning: No features found for {pid}")
            continue

        features = np.concatenate(all_features, axis=0)

        with h5py.File(out_path, "w") as out_f:
            out_f.create_dataset("features", data=features)
            if all_coords:
                coords = np.concatenate(all_coords, axis=0)
                out_f.create_dataset("coords", data=coords)
            if has_confidences:
                confidences = np.concatenate(all_confidences, axis=0)
                out_f.create_dataset("confidences", data=confidences)

        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed} patients...")

    print(f"Done: {processed} created, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(description="Merge per-slide H5s into per-patient H5s")
    parser.add_argument("--input_dir", required=True, help="Slide-level H5 directory")
    parser.add_argument("--output_dir", required=True, help="Patient-level H5 output directory")
    parser.add_argument("--clinical_csv", help="CSV with patient_id column")
    args = parser.parse_args()

    concat_patient_h5(args.input_dir, args.output_dir, args.clinical_csv)


if __name__ == "__main__":
    main()
