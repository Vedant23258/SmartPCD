"""Segmentation utilities for crack and pothole region extraction."""

from __future__ import annotations

import cv2
import numpy as np


def extract_road_roi(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract an approximate road region of interest.

    For many front-facing road images, the most relevant road surface occupies
    the lower portion of the frame. This function creates a simple mask that
    prioritizes the lower 70% of the image to reduce background distractions.

    Args:
        image: Preprocessed grayscale image.

    Returns:
        A tuple of (roi_image, roi_mask).
    """

    height, width = image.shape[:2]
    roi_mask = np.zeros((height, width), dtype=np.uint8)

    # Keep the lower portion of the frame where the road is typically visible.
    polygon = np.array(
        [
            [
                (int(0.05 * width), height),
                (int(0.95 * width), height),
                (int(0.85 * width), int(0.30 * height)),
                (int(0.15 * width), int(0.30 * height)),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(roi_mask, polygon, 255)

    roi_image = cv2.bitwise_and(image, image, mask=roi_mask)
    return roi_image, roi_mask


def segment_defects(preprocessed_image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Segment likely crack and pothole regions from a grayscale image.

    The segmentation uses:
    - Canny edge detection for boundary emphasis
    - Binary thresholding for dark defect highlighting
    - Morphological closing/opening using dilation and erosion

    Args:
        preprocessed_image: Sharpened grayscale image.

    Returns:
        A tuple containing:
        - Final binary defect mask
        - Dictionary of intermediate segmentation outputs
    """

    if preprocessed_image is None:
        raise ValueError("Preprocessed image is None. Please provide a valid image.")

    roi_image, roi_mask = extract_road_roi(preprocessed_image)

    # Edge detection captures thin crack structures and pothole boundaries.
    edges = cv2.Canny(roi_image, threshold1=50, threshold2=150)

    # Invert Otsu thresholding highlights dark road defects against lighter asphalt.
    _, binary = cv2.threshold(
        roi_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = cv2.bitwise_and(binary, binary, mask=roi_mask)

    # Combine intensity-based and edge-based clues before morphology.
    combined = cv2.bitwise_or(binary, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Dilation connects nearby defect pixels into continuous regions.
    dilated = cv2.dilate(combined, kernel, iterations=2)

    # Erosion removes small false positives introduced by dilation.
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Morphological close helps fill small gaps inside defect regions.
    defect_mask = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=2)

    outputs = {
        "roi_image": roi_image,
        "roi_mask": roi_mask,
        "edges": edges,
        "binary": binary,
        "combined": combined,
        "dilated": dilated,
        "eroded": eroded,
        "defect_mask": defect_mask,
    }
    return defect_mask, outputs

