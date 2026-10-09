
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class HeartKNNModel:
    """
    Object-oriented KNN classification component.

    Combines preprocessing and training in one
    sklearn Pipeline to prevent data leakage.
    """

    def __init__(self, preprocessor, n_neighbors=5):
        self.n_neighbors = n_neighbors

        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    metric="euclidean",
                    weights="uniform",
                ),
            ),
        ])

    def train(self, X_train, y_train):
        """Fit preprocessing and KNN on training data."""

        self.pipeline.fit(X_train, y_train)

        print("KNN model trained successfully!")
        print("Number of neighbors:", self.n_neighbors)
        print("Training samples:", len(X_train))

        return self

    def predict(self, X):
        """Generate predictions for unseen observations."""

        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance on held-out data."""

        predictions = self.predict(X_test)

        results = {
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ),
            "predictions": predictions,
        }

        print("\n=== KNN Evaluation Results ===")

        for metric in [
            "accuracy", "precision", "recall", "f1_score"
        ]:
            print(f"{metric}: {results[metric]:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=[
                    "No Heart Disease",
                    "Heart Disease",
                ],
                zero_division=0,
            )
        )

        print("\nConfusion Matrix:")
        print(results["confusion_matrix"])

        return results
