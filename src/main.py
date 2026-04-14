"""Main training and inference pipeline for road defect detection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from classifier import (
    CLASS_NAMES,
    evaluate_model,
    load_model,
    predict_class,
    save_model,
    train_svm,
)
from features import calculate_damage_percentage, extract_features, severity_from_area
from preprocessing import preprocess_image
from segmentation import segment_defects


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_MAP = {
    "normal": 0,
    "crack": 1,
    "pothole": 2,
    "severe": 3,
}


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk in BGR format."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def process_image(image: np.ndarray) -> dict[str, object]:
    """Run the full preprocessing, segmentation, and feature extraction pipeline."""

    processed_image, prep_outputs = preprocess_image(image)
    defect_mask, seg_outputs = segment_defects(processed_image)
    feature_vector, feature_map = extract_features(processed_image, defect_mask)
    damage_percentage = calculate_damage_percentage(
        defect_mask, seg_outputs.get("roi_mask")
    )
    severity = severity_from_area(feature_map["area"])

    return {
        "processed_image": processed_image,
        "defect_mask": defect_mask,
        "feature_vector": feature_vector,
        "feature_map": feature_map,
        "damage_percentage": damage_percentage,
        "severity": severity,
        "prep_outputs": prep_outputs,
        "seg_outputs": seg_outputs,
    }


def iter_dataset_images(dataset_dir: str | Path):
    """Yield dataset image paths with class labels."""

    dataset_dir = Path(dataset_dir)
    for folder_name, label in LABEL_MAP.items():
        class_dir = dataset_dir / folder_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in VALID_EXTENSIONS:
                yield image_path, label


def prepare_dataset(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset images, extract features, and prepare training arrays."""

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for image_path, label in iter_dataset_images(dataset_dir):
        try:
            image = load_image(image_path)
            outputs = process_image(image)
            feature_rows.append(outputs["feature_vector"])
            labels.append(label)
        except Exception as exc:
            print(f"Skipping {image_path.name}: {exc}")

    if not feature_rows:
        raise ValueError(
            "No valid training images were found. Please place images in "
            "dataset/normal, dataset/crack, dataset/pothole, and dataset/severe."
        )

    return np.vstack(feature_rows), np.array(labels, dtype=np.int32)


def train_pipeline(dataset_dir: str | Path, model_path: str | Path) -> None:
    """Train the SVM classifier from the dataset folders and save it."""

    features, labels = prepare_dataset(dataset_dir)
    unique_labels = np.unique(labels)

    if len(labels) >= 8 and len(unique_labels) > 1:
        stratify = labels if np.min(np.bincount(labels)) >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
        model = train_svm(x_train, y_train)
        evaluation = evaluate_model(model, x_test, y_test)
        print(f"Validation accuracy: {evaluation['accuracy'] * 100:.2f}%")
        print("Classification report:")
        print(evaluation["report"])
    else:
        model = train_svm(features, labels)
        print("Dataset is too small for a stable validation split. Model trained on all samples.")

    save_model(model, model_path)
    print(f"Training complete. Model saved to: {model_path}")
    print(f"Samples used: {len(labels)}")
    for class_id, class_name in CLASS_NAMES.items():
        class_count = int(np.sum(labels == class_id))
        print(f"{class_name}: {class_count}")


def save_figure(output_path: str | Path) -> None:
    """Save the current matplotlib figure."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    print(f"Result figure saved to: {output_path}")


def save_processing_outputs(
    processed_image: np.ndarray,
    defect_mask: np.ndarray,
    output_path: str | Path | None,
) -> None:
    """Save processed grayscale and defect mask images alongside the figure."""

    if output_path is None:
        return

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_path = output_dir / "processed_image.png"
    mask_path = output_dir / "segmented_defect_mask.png"

    cv2.imwrite(str(processed_path), processed_image)
    cv2.imwrite(str(mask_path), defect_mask)

    print(f"Processed image saved to: {processed_path}")
    print(f"Defect mask saved to: {mask_path}")


def visualize_results(
    original_image: np.ndarray,
    processed_image: np.ndarray,
    defect_mask: np.ndarray,
    predicted_name: str,
    feature_map: dict[str, float],
    damage_percentage: float,
    severity: str,
    output_path: str | Path | None = None,
) -> None:
    """Display original image, processed image, segmented defect, and prediction."""

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(processed_image, cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(defect_mask, cmap="gray")
    plt.title(
        f"Segmented Defect\nClass: {predicted_name}\n"
        f"Damage: {damage_percentage:.2f}% | Severity: {severity}"
    )
    plt.axis("off")

    plt.suptitle(
        "Smart Pothole and Crack Detection\n"
        f"Mean: {feature_map['mean_intensity']:.2f}, "
        f"Contrast: {feature_map['contrast']:.2f}, "
        f"Area: {feature_map['area']:.2f}"
    )
    plt.tight_layout()

    if output_path is not None:
        save_figure(output_path)
    plt.show()


def predict_pipeline(
    image_path: str | Path,
    model_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """Run prediction on a single road image."""

    model = load_model(model_path)
    original_image = load_image(image_path)
    outputs = process_image(original_image)
    label, predicted_name, probabilities = predict_class(model, outputs["feature_vector"])

    print(f"Predicted label: {label}")
    print(f"Predicted class: {predicted_name}")
    print(f"Estimated severity: {outputs['severity']}")
    print(f"Damage percentage: {outputs['damage_percentage']:.2f}%")
    if probabilities is not None:
        for class_id, probability in enumerate(probabilities):
            print(f"{CLASS_NAMES[class_id]} probability: {probability:.4f}")

    visualize_results(
        original_image=original_image,
        processed_image=outputs["processed_image"],
        defect_mask=outputs["defect_mask"],
        predicted_name=predicted_name,
        feature_map=outputs["feature_map"],
        damage_percentage=outputs["damage_percentage"],
        severity=outputs["severity"],
        output_path=output_path,
    )
    save_processing_outputs(
        processed_image=outputs["processed_image"],
        defect_mask=outputs["defect_mask"],
        output_path=output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create a command-line interface for training and prediction."""

    parser = argparse.ArgumentParser(
        description="Smart Pothole and Crack Detection using Digital Image Processing and SVM"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the SVM model from dataset images")
    train_parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "dataset"),
        help="Path to dataset directory containing normal/crack/pothole/severe folders",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "models" / "svm_model.joblib"),
        help="Path to save the trained model",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict the class of a single road image")
    predict_parser.add_argument("--image", type=str, required=True, help="Path to input road image")
    predict_parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "models" / "svm_model.joblib"),
        help="Path to the trained model",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "prediction_result.png"),
        help="Path to save the output visualization",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_pipeline(dataset_dir=args.dataset, model_path=args.model)
    elif args.command == "predict":
        predict_pipeline(image_path=args.image, model_path=args.model, output_path=args.output)


if __name__ == "__main__":
    main()
