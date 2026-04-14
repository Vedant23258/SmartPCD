"""SVM training and inference utilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_NAMES = {
    0: "Normal",
    1: "Crack",
    2: "Pothole",
    3: "Severe Pothole",
}


def train_svm(features: np.ndarray, labels: np.ndarray) -> Pipeline:
    """Train an SVM classifier with feature scaling."""

    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Training data is empty. Add images to the dataset folders first.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)),
        ]
    )
    model.fit(features, labels)
    return model


def save_model(model: Pipeline, model_path: str | Path) -> None:
    """Save the trained SVM model using joblib."""

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: str | Path) -> Pipeline:
    """Load a previously trained model."""

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)


def predict_class(model: Pipeline, feature_vector: np.ndarray) -> tuple[int, str, np.ndarray | None]:
    """Predict the defect class for a single feature vector."""

    reshaped = feature_vector.reshape(1, -1)
    label = int(model.predict(reshaped)[0])
    probabilities = model.predict_proba(reshaped)[0] if hasattr(model, "predict_proba") else None
    return label, CLASS_NAMES[label], probabilities


def evaluate_model(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Evaluate the trained classifier on a feature matrix."""

    predictions = model.predict(features)
    accuracy = float(accuracy_score(labels, predictions))
    report = classification_report(
        labels,
        predictions,
        labels=sorted(CLASS_NAMES.keys()),
        target_names=[CLASS_NAMES[idx] for idx in sorted(CLASS_NAMES.keys())],
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "report": report,
    }
