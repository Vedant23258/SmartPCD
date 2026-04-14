# Project Report

## Title

Smart Pothole and Crack Detection using Digital Image Processing and SVM

## Abstract

This project presents a lightweight and explainable computer vision system for
detecting road defects such as cracks and potholes from road images. The system
uses classical digital image processing techniques to enhance the image,
segment damaged regions, and extract meaningful features. These features are
then classified using a Support Vector Machine (SVM) into four categories:
Normal, Crack, Pothole, and Severe Pothole. The project is designed to be
modular, beginner-friendly, and suitable for academic demonstrations.

## Objective

- Detect cracks and potholes from road images
- Improve visibility of road defects using preprocessing
- Segment damaged regions using image processing methods
- Extract interpretable features from the segmented region
- Classify road conditions using SVM

## Problem Statement

Manual road inspection is time-consuming, inconsistent, and costly. A simple
automated system can help identify road damage earlier and assist maintenance
planning. This project focuses on a non-deep-learning approach that is easy to
understand and implement.

## Tools and Technologies

- Python
- OpenCV
- NumPy
- scikit-image
- scikit-learn
- matplotlib
- joblib

## Methodology

### 1. Image Acquisition

Road images are collected and arranged into class folders:

- `normal`
- `crack`
- `pothole`
- `severe`

### 2. Preprocessing

The preprocessing stage improves image quality before segmentation:

- Convert image to grayscale
- Apply histogram equalization
- Apply Gaussian blur
- Apply median filtering
- Apply sharpening

### 3. ROI Extraction

The lower portion of the image is treated as the road region of interest. This
reduces distraction from sky, vehicles, and roadside background.

### 4. Segmentation

Defect regions are extracted using:

- Canny edge detection
- Binary inverse thresholding with Otsu method
- Morphological dilation
- Morphological erosion
- Morphological closing

### 5. Feature Extraction

Three types of features are extracted:

- Intensity features: mean, variance
- Texture features: contrast, energy, homogeneity
- Shape features: area, perimeter

### 6. Classification

The extracted feature vector is passed to an SVM classifier. The model predicts:

- 0: Normal
- 1: Crack
- 2: Pothole
- 3: Severe Pothole

## System Architecture

```text
Input Image
   -> Preprocessing
   -> ROI Extraction
   -> Segmentation
   -> Feature Extraction
   -> SVM Classification
   -> Result Visualization
```

## Modules

### `preprocessing.py`

Handles grayscale conversion, equalization, filtering, and sharpening.

### `segmentation.py`

Extracts road ROI and defect mask using edge detection, thresholding, and
morphological operations.

### `features.py`

Computes intensity, texture, shape, damage percentage, and simple severity.

### `classifier.py`

Trains, evaluates, saves, loads, and predicts using SVM.

### `main.py`

Runs the complete training and prediction pipeline and displays results.

## Output

The system displays:

- Original image
- Processed grayscale image
- Segmented defect mask
- Predicted class
- Damage percentage
- Severity level

It also saves:

- `outputs/prediction_result.png`
- `outputs/processed_image.png`
- `outputs/segmented_defect_mask.png`

## Advantages

- Lightweight and fast
- Easy to understand
- No deep learning required
- Suitable for beginner projects
- Modular code structure

## Limitations

- Performance depends on dataset quality
- ROI extraction is approximate
- Shadows and water on roads may affect segmentation
- Severe lighting changes can reduce accuracy

## Future Scope

- Use a larger and more diverse dataset
- Add better road extraction techniques
- Integrate GPS tagging for smart city maintenance
- Add live video or CCTV support
- Build a desktop or web dashboard

## Conclusion

This project demonstrates that classical image processing combined with SVM can
provide a practical and explainable solution for pothole and crack detection.
It is especially useful for learning computer vision fundamentals and for
building a low-cost prototype.

