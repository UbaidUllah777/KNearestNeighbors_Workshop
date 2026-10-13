
# -*- coding: utf-8 -*-
"""
Simple feature engineering for the UCI Heart Disease dataset.

Creates two easy-to-understand binary features:
1. Resting blood pressure >= 140
2. Cholesterol >= 240
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class HeartFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create simple threshold-based features for KNN.

    Original input features are preserved.
    The target variable is not used.
    """

    REQUIRED_COLUMNS = (
        "trestbps",
        "chol",
    )

    ENGINEERED_NUMERICAL_FEATURES = (
        "bp_140_or_higher",
        "chol_240_or_higher",
    )

    def _validate_input(self, X):
        """Validate the input dataset."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if X.empty:
            raise ValueError("Input data cannot be empty.")

        if X.columns.duplicated().any():
            raise ValueError(
                "Duplicate feature columns detected."
            )

        missing = set(self.REQUIRED_COLUMNS) - set(X.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        existing = (
            set(self.ENGINEERED_NUMERICAL_FEATURES)
            & set(X.columns)
        )

        if existing:
            raise ValueError(
                f"Engineered columns already exist: {sorted(existing)}"
            )

    def fit(self, X, y=None):
        """
        Learn the input column structure.

        No averages, thresholds, or other statistics
        are calculated from the dataset.
        """

        self._validate_input(X)

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object
        )

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """Add two binary features to the input dataset."""

        check_is_fitted(self, "feature_names_in_")

        self._validate_input(X)

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                "Input columns differ from the fitted structure."
            )

        result = X.copy()

        # Convert original measurements to numerical values
        blood_pressure = pd.to_numeric(
            result["trestbps"],
            errors="raise"
        )

        cholesterol = pd.to_numeric(
            result["chol"],
            errors="raise"
        )

        # Feature 1:
        # 1 if resting blood pressure is 140 or higher
        # 0 otherwise
        result["bp_140_or_higher"] = np.where(
            blood_pressure.isna(),
            np.nan,
            (blood_pressure >= 140).astype(float)
        )

        # Feature 2:
        # 1 if cholesterol is 240 or higher
        # 0 otherwise
        result["chol_240_or_higher"] = np.where(
            cholesterol.isna(),
            np.nan,
            (cholesterol >= 240).astype(float)
        )

        # Missing measurements remain missing,
        # ready for the pipeline's imputer.

        return result

    def get_feature_names_out(self, input_features=None):
        """Return original and new feature names."""

        check_is_fitted(self, "feature_names_in_")

        if input_features is not None:
            if list(input_features) != list(self.feature_names_in_):
                raise ValueError(
                    "Feature names do not match fitted input."
                )

        return np.asarray(
            list(self.feature_names_in_)
            + list(self.ENGINEERED_NUMERICAL_FEATURES),
            dtype=object
        )
