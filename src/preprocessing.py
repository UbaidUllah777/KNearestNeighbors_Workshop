
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
        expected = set(
            self.NUMERICAL_FEATURES
            + self.CATEGORICAL_FEATURES
        )

        missing_columns = expected - set(X.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        if len(X) != len(y):
            raise ValueError(
                "Feature and target lengths do not match."
            )

        if y.isna().any():
            raise ValueError(
                "Target contains missing values."
            )

        print("Data validation: PASSED")
        print("Samples:", len(X))
        print("Features:", X.shape[1])

    def split_data(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def build_preprocessor(self):
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
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
                self.NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                self.CATEGORICAL_FEATURES,
            ),
        ])
