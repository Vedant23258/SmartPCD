"""Feature extraction for road defect classification."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_features(processed_image: np.ndarray, defect_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Extract intensity, texture, and shape features for SVM classification.

    Args:
        processed_image: Preprocessed grayscale image.
        defect_mask: Binary mask of segmented defect regions.

    Returns:
        A tuple of:
        - 1D NumPy feature vector
        - Dictionary with named feature values
    """

    if processed_image is None or defect_mask is None:
        raise ValueError("Processed image and defect mask must both be valid arrays.")

    mask_binary = (defect_mask > 0).astype(np.uint8)
    masked_pixels = processed_image[mask_binary == 1]

    if masked_pixels.size == 0:
        masked_pixels = processed_image.flatten()

    mean_intensity = float(np.mean(masked_pixels))
    variance_intensity = float(np.var(masked_pixels))

    glcm_input = cv2.resize(processed_image, (128, 128), interpolation=cv2.INTER_AREA)
    glcm = graycomatrix(
        glcm_input,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    energy = float(graycoprops(glcm, "energy")[0, 0])
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])

    largest_contour = _largest_contour(defect_mask)
    if largest_contour is not None:
        area = float(cv2.contourArea(largest_contour))
        perimeter = float(cv2.arcLength(largest_contour, True))
    else:
        area = 0.0
        perimeter = 0.0

    feature_map = {
        "mean_intensity": mean_intensity,
        "variance_intensity": variance_intensity,
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "area": area,
        "perimeter": perimeter,
    }
    feature_vector = np.array(list(feature_map.values()), dtype=np.float32)
    return feature_vector, feature_map


def calculate_damage_percentage(defect_mask: np.ndarray, roi_mask: np.ndarray | None = None) -> float:
    """Estimate defect severity as a percentage of damaged pixels.

    Args:
        defect_mask: Binary defect mask.
        roi_mask: Optional road ROI mask. If provided, damage is measured only
            inside the road region.

    Returns:
        Percentage of damaged area.
    """

    if roi_mask is not None and np.count_nonzero(roi_mask) > 0:
        total_region = float(np.count_nonzero(roi_mask))
    else:
        total_region = float(defect_mask.size)

    if total_region == 0:
        return 0.0

    defect_pixels = float(np.count_nonzero(defect_mask))
    return (defect_pixels / total_region) * 100.0


def severity_from_area(area: float) -> str:
    """Provide a simple rule-based severity description from contour area."""

    if area < 500:
        return "Low"
    if area < 2000:
        return "Moderate"
    if area < 5000:
        return "High"
    return "Critical"

