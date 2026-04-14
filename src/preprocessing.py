"""Image preprocessing utilities for road defect analysis.

This module keeps the early image enhancement steps in one place so the
pipeline remains easy to understand and debug.
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Preprocess an input BGR road image.

    Steps:
    1. Convert to grayscale so intensity-based enhancement is simpler.
    2. Equalize the histogram to improve contrast in dark or uneven scenes.
    3. Apply Gaussian blur to smooth broad noise while preserving general shape.
    4. Apply a median filter to suppress salt-and-pepper noise.
    5. Sharpen the result to make crack and pothole boundaries clearer.

    Args:
        image: Input image in BGR format as loaded by OpenCV.

    Returns:
        A tuple containing:
        - The final sharpened grayscale image.
        - A dictionary with intermediate images for visualization/debugging.
    """

    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    # Step 1: Convert the road image to grayscale to reduce color complexity.
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Improve contrast so damaged regions become more visible.
    equalized = cv2.equalizeHist(grayscale)

    # Step 3: Smooth the image using Gaussian blur to reduce high-frequency noise.
    gaussian_blur = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Step 4: Apply a median filter to remove isolated noise pixels.
    median_filtered = cv2.medianBlur(gaussian_blur, 5)

    # Step 5: Sharpen the image to enhance edge details of cracks/potholes.
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(median_filtered, -1, sharpening_kernel)

    intermediates = {
        "grayscale": grayscale,
        "equalized": equalized,
        "gaussian_blur": gaussian_blur,
        "median_filtered": median_filtered,
        "sharpened": sharpened,
    }
    return sharpened, intermediates

