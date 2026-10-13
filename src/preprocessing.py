
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class HeartDataPreprocessor:
    """
    Reusable preprocessing component for Heart Disease KNN
    classification.
    """

    NUMERICAL_FEATURES = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]

    CATEGORICAL_FEATURES = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    
    def validate_data(self, X, y):
        """Validate dataset structure, values, and classification target."""

        # 1. Verify input types
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")

        # 2. Check dataset is not empty
        if X.empty or y.empty:
            raise ValueError("Dataset cannot be empty.")

        # 3. Check required features
        expected = set(
            self.NUMERICAL_FEATURES
            + self.CATEGORICAL_FEATURES
        )

        missing_columns = expected - set(X.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        # 4. Check duplicate column names
        if X.columns.duplicated().any():
            raise ValueError("Duplicate feature column names detected.")

        # 5. Check feature and target alignment
        if len(X) != len(y):
            raise ValueError("Feature and target lengths do not match.")

        if not X.index.equals(y.index):
            raise ValueError("Feature and target indices are misaligned.")

        # 6. Validate classification target
        if y.isna().any():
            raise ValueError("Target contains missing values.")

        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("Target must contain binary labels 0 and 1.")

        if y.nunique() != 2:
            raise ValueError("Both target classes must be present.")

        # 7. Validate numerical representations
        for column in expected:
            converted = pd.to_numeric(X[column], errors="coerce")

            invalid = X[column].notna() & converted.isna()

            if invalid.any():
                raise ValueError(
                    f"Non-numeric values detected in '{column}'."
                )

            if np.isinf(converted.to_numpy(dtype=float)).any():
                raise ValueError(
                    f"Infinite values detected in '{column}'."
                )

        # Missing feature values are permitted here.
        # They will be handled by the training-fitted imputers.

        print("Data validation: PASSED")
        print("Samples:", len(X))
        print("Features:", X.shape[1])
        print("Missing feature values:", X.isna().sum().sum())
        print("Target classes:", sorted(y.unique().tolist()))
        print("Feature-target alignment: PASSED")



    def split_data(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )



    def build_preprocessor(self, additional_numerical_features=None):
        """
        Build a leakage-safe preprocessing transformer.

        Optionally include engineered numerical features.
        Existing baseline behavior remains unchanged.
        """

        extra_features = (
            list(additional_numerical_features)
            if additional_numerical_features is not None
            else []
        )

        if len(extra_features) != len(set(extra_features)):
            raise ValueError("Duplicate engineered features detected.")

        existing_features = set(
            self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES
        )

        if existing_features.intersection(extra_features):
            raise ValueError(
                "Engineered feature names overlap existing features."
            )

        numerical_features = (
            list(self.NUMERICAL_FEATURES) + extra_features
        )

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ])

        return ColumnTransformer([
            (
                "numerical",
                numeric_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                self.CATEGORICAL_FEATURES,
            ),
        ])
