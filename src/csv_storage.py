
# -*- coding: utf-8 -*-
"""
CSV storage for the KNN Heart Disease workshop.

Preserves reproducible dataset snapshots without
introducing preprocessing data leakage.
"""

from pathlib import Path
import pandas as pd

import numpy as np
from scipy import sparse


class HeartDiseaseCSVStorage:
    """Save and verify Heart Disease dataset snapshots."""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_prepared(self, X, y):
        """
        Save original features with the binary target.

        No imputation, encoding, or scaling occurs here.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")

        if not isinstance(y, pd.Series):
            raise TypeError("y must be a Series.")

        if X.empty:
            raise ValueError("Features cannot be empty.")

        if not X.index.equals(y.index):
            raise ValueError(
                "Feature and target indices do not match."
            )

        if not X.index.is_unique:
            raise ValueError(
                "Dataset index must contain unique identifiers."
            )

        reserved = {"record_id", "num", "heart_disease"}

        if reserved.intersection(X.columns):
            raise ValueError(
                "Target or record identifier found among features."
            )

        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError(
                "Binary target must contain only 0 and 1."
            )

        # Create reproducible prepared dataset
        prepared_df = X.copy()

        # Retain stable source row identifiers
        prepared_df.insert(
            0,
            "record_id",
            X.index
        )

        # Add binary classification target
        prepared_df["heart_disease"] = y

        output_path = (
            self.data_dir / "heart_disease_prepared.csv"
        )

        prepared_df.to_csv(output_path, index=False)

        # Verify saved file by loading it again
        reloaded_df = pd.read_csv(output_path)

        if reloaded_df.shape != prepared_df.shape:
            raise ValueError(
                "Saved CSV dimensions do not match."
            )

        if reloaded_df.columns.tolist() != prepared_df.columns.tolist():
            raise ValueError(
                "Saved CSV columns do not match."
            )

        if reloaded_df["record_id"].duplicated().any():
            raise ValueError(
                "Duplicate record identifiers found."
            )

        if (
            reloaded_df[X.columns].isna().sum().sum()
            != X.isna().sum().sum()
        ):
            raise ValueError(
                "Missing-value counts changed unexpectedly."
            )

        if not reloaded_df["heart_disease"].equals(
            y.reset_index(drop=True).rename("heart_disease")
        ):
            raise ValueError(
                "Saved target does not match original target."
            )

        print("=== PREPARED DATASET SAVED ===")
        print("File:", output_path)
        print("Rows:", reloaded_df.shape[0])
        print("Columns:", reloaded_df.shape[1])

        print(
            "Missing feature values:",
            reloaded_df[X.columns].isna().sum().sum()
        )

        print("\nTarget distribution:")
        print(
            reloaded_df["heart_disease"]
            .value_counts()
            .sort_index()
        )

        print("\nCSV verification: PASSED")

        return output_path

    
    def save_cleaned_splits(
        self,
        fitted_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    ):
        """
        Save training and testing data after applying
        an already-fitted preprocessing pipeline.

        No fitting occurs in this method.
        """

        # Check that feature and target indices match
        if not X_train.index.equals(y_train.index):
            raise ValueError("Training indices are misaligned.")

        if not X_test.index.equals(y_test.index):
            raise ValueError("Testing indices are misaligned.")

        if not X_train.index.is_unique or not X_test.index.is_unique:
            raise ValueError("Duplicate record identifiers detected.")

        if set(X_train.index) & set(X_test.index):
            raise ValueError("Training and testing records overlap.")

        # Retrieve the fitted preprocessing component
        transformer = fitted_pipeline.named_steps["preprocessing"]

        if not hasattr(transformer, "transformers_"):
            raise ValueError("Preprocessing pipeline is not fitted.")

        feature_names = transformer.get_feature_names_out().tolist()

        def transform_split(X, y):
            # Apply already-fitted transformations
            transformed = transformer.transform(X)

            if sparse.issparse(transformed):
                transformed = transformed.toarray()

            values = np.asarray(transformed, dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    "Processed dataset contains invalid values."
                )

            cleaned = pd.DataFrame(
                values,
                columns=feature_names
            )

            cleaned.insert(
                0, "record_id", X.index.to_numpy()
            )

            cleaned["heart_disease"] = y.to_numpy()

            return cleaned

        # Process training and testing data separately
        train_cleaned = transform_split(X_train, y_train)
        test_cleaned = transform_split(X_test, y_test)

        # Save cleaned datasets
        train_path = self.data_dir / "heart_train_cleaned.csv"
        test_path = self.data_dir / "heart_test_cleaned.csv"

        train_cleaned.to_csv(train_path, index=False)
        test_cleaned.to_csv(test_path, index=False)

        # Reload and verify saved artifacts
        for path, original in [
            (train_path, train_cleaned),
            (test_path, test_cleaned),
        ]:
            reloaded = pd.read_csv(path)

            if reloaded.shape != original.shape:
                raise ValueError(f"Shape mismatch: {path}")

            if reloaded.columns.tolist() != original.columns.tolist():
                raise ValueError(f"Column mismatch: {path}")

            if not np.array_equal(
                reloaded["record_id"],
                original["record_id"],
            ):
                raise ValueError(f"Record ID mismatch: {path}")

            if not np.array_equal(
                reloaded["heart_disease"],
                original["heart_disease"],
            ):
                raise ValueError(f"Target mismatch: {path}")

            if not np.allclose(
                reloaded[feature_names].to_numpy(),
                original[feature_names].to_numpy(),
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError(f"Feature mismatch: {path}")

        print("=== CLEANED DATASETS SAVED ===")

        print("\nTraining dataset:")
        print("File:", train_path)
        print("Shape:", train_cleaned.shape)

        print("\nTesting dataset:")
        print("File:", test_path)
        print("Shape:", test_cleaned.shape)

        print("\nTransformed features:", len(feature_names))
        print("Missing values:", (
            train_cleaned.isna().sum().sum()
            + test_cleaned.isna().sum().sum()
        ))

        print("\nCSV verification: PASSED")

        return train_path, test_path
