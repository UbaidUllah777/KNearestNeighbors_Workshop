
# -*- coding: utf-8 -*-
"""
Data acquisition module for the KNN workshop.

Source: UCI Machine Learning Repository
Dataset: Heart Disease (ID 45)
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


class HeartDiseaseDataAcquisition:
    """Acquire, validate, and preserve the original UCI dataset."""

    def __init__(self, dataset_id=45, data_dir="data"):
        self.dataset_id = dataset_id
        self.data_dir = Path(data_dir)

        self.dataset = None
        self.features = None
        self.original_target = None
        self.binary_target = None

    def acquire(self):
        """Download and validate the Heart Disease dataset."""

        self.dataset = fetch_ucirepo(id=self.dataset_id)

        features = self.dataset.data.features.copy()
        targets = self.dataset.data.targets.copy()

        if features.empty:
            raise ValueError("The acquired feature dataset is empty.")

        if "num" not in targets.columns:
            raise ValueError("Expected target column 'num' not found.")

        if "num" in features.columns:
            raise ValueError("Target 'num' must not appear in features.")

        original_target = pd.to_numeric(
            targets["num"], errors="coerce"
        )

        if original_target.isna().any():
            raise ValueError(
                "Original target contains missing or invalid values."
            )

        if not original_target.isin([0, 1, 2, 3, 4]).all():
            raise ValueError(
                "Unexpected Heart Disease target labels detected."
            )

        if not features.index.equals(original_target.index):
            raise ValueError(
                "Feature and target indices are not aligned."
            )

        original_target = original_target.astype(int)
        original_target.name = "num"

        binary_target = (original_target > 0).astype(int)
        binary_target.name = "heart_disease"

        self.features = features
        self.original_target = original_target
        self.binary_target = binary_target

        print("UCI Heart Disease dataset acquired successfully.")
        print("Samples:", len(features))
        print("Features:", features.shape[1])
        print("Missing feature values:", features.isna().sum().sum())

        return (
            self.features.copy(),
            self.original_target.copy(),
            self.binary_target.copy()
        )

    def save_raw_csv(self):
        """Save the original features and original target to CSV."""

        if self.features is None or self.original_target is None:
            raise RuntimeError(
                "Call acquire() before saving the dataset."
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)

        raw_df = pd.concat(
            [self.features, self.original_target],
            axis=1
        )

        output_path = self.data_dir / "heart_disease_raw.csv"

        raw_df.to_csv(output_path, index=False)

        print("\nRaw dataset saved successfully.")
        print("File:", output_path)
        print("Rows:", raw_df.shape[0])
        print("Columns:", raw_df.shape[1])

        return output_path
