"""Dataset classes and data loading utilities."""

import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from pipeline.config import (
    EMORY_FEATS_PATH, TCGA_FEATS_PATH, CLINICAL_CSV,
    BATCH_SIZE, NUM_FEATURES, SEED, VAL_FRACTION, CENSORING_DAYS,
)


def apply_5year_censoring(df, cutoff_days=CENSORING_DAYS):
    """Apply 5-year censoring: cap time at cutoff, set event=0 if censored."""
    df = df.copy()
    mask = df["time"] > cutoff_days
    df.loc[mask, "os"] = 0
    df.loc[mask, "time"] = cutoff_days
    return df


def load_clinical_data(csv_path=CLINICAL_CSV, censor=True):
    """Load clinical CSV and create train/val/test splits.

    The CSV has a 'split' column with 'train' and 'test'.
    We further split 'train' into 'train' and 'val'.
    """
    df = pd.read_csv(csv_path)

    if censor:
        df = apply_5year_censoring(df)

    train_df = df[df["split"] == "train"].copy()

    train_indices, val_indices = train_test_split(
        train_df.index,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=train_df["os"],
    )
    df.loc[val_indices, "split"] = "val"

    return df


class LVDataset(Dataset):
    """Emory dataset: loads patient-level H5 features."""

    def __init__(self, feats_path, df, split, num_features=NUM_FEATURES):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        h5_path = os.path.join(self.feats_path, row["patient_id"] + ".h5")

        with h5py.File(h5_path, "r") as f:
            features = torch.from_numpy(f["features"][:])
            # Load confidences if available (for weighted sampling)
            if "confidences" in f:
                confidences = torch.from_numpy(f["confidences"][:])
            else:
                confidences = None

        # Subsample patches for train/val
        if self.split in ("train", "val"):
            n = features.shape[0]
            if n >= self.num_features:
                if self.split == "train":
                    if confidences is not None:
                        # Confidence-weighted sampling (higher confidence → more likely)
                        weights = confidences.float()
                        weights = weights / weights.sum()
                        indices = torch.multinomial(weights, self.num_features, replacement=False)
                    else:
                        indices = torch.randperm(n)[:self.num_features]
                else:
                    indices = torch.arange(self.num_features)
            else:
                gen = torch.Generator().manual_seed(SEED)
                indices = torch.randint(n, (self.num_features,), generator=gen)
            features = features[indices]

        event = torch.tensor(row["os"], dtype=torch.float32)
        time = torch.tensor(row["time"], dtype=torch.float32)
        lvi = torch.tensor(row["lvi"], dtype=torch.float32)

        return features, event, time, lvi


class TCGADataset(Dataset):
    """TCGA dataset for external validation inference."""

    def __init__(self, df, feats_path=TCGA_FEATS_PATH):
        self.df = df.reset_index(drop=True)
        self.feats_path = feats_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        h5_path = os.path.join(self.feats_path, row["patient_id"] + ".h5")

        with h5py.File(h5_path, "r") as f:
            features = torch.from_numpy(f["features"][:])

        event = torch.tensor(row["os_event"], dtype=torch.float32)
        time = torch.tensor(row["os_time"], dtype=torch.float32)

        return features, event, time, row["patient_id"]


def _worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)


def create_dataloaders(df, feats_path=EMORY_FEATS_PATH):
    """Create train, val, and test DataLoaders for Emory data."""
    train_loader = DataLoader(
        LVDataset(feats_path, df, "train"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        LVDataset(feats_path, df, "val"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        LVDataset(feats_path, df, "test"),
        batch_size=1,
        shuffle=False,
        worker_init_fn=_worker_init_fn,
    )
    return train_loader, val_loader, test_loader
