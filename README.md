# Smart Pothole and Crack Detection using Digital Image Processing and SVM

## Overview

This project detects road surface defects such as cracks and potholes using
classical digital image processing and a Support Vector Machine (SVM)
classifier. It is lightweight, explainable, and suitable for academic
mini-projects, prototypes, and beginners learning computer vision.

The pipeline performs:

1. Image preprocessing
2. Road region extraction
3. Defect segmentation
4. Feature extraction
5. SVM-based classification

## Project Structure

```text
project/
│── dataset/
│   │── normal/
│   │── crack/
│   │── pothole/
│   │── severe/
│── src/
│   │── preprocessing.py
│   │── segmentation.py
│   │── features.py
│   │── classifier.py
│   │── main.py
│── models/
│── outputs/
│── requirements.txt
│── README.md
```

## Libraries Used

- OpenCV
- NumPy
- scikit-image
- scikit-learn
- matplotlib
- joblib

## Processing Pipeline

### 1. Preprocessing

The input road image is cleaned and enhanced using:

- Grayscale conversion
- Histogram equalization
- Gaussian blur
- Median filtering
- Sharpening

These steps improve contrast and reduce noise before segmentation.

### 2. Segmentation

Potential crack and pothole regions are extracted using:

- Road ROI masking
- Canny edge detection
- Binary thresholding
- Morphological dilation and erosion

This produces a defect mask highlighting suspicious regions.

### 3. Feature Extraction

The system extracts three feature groups:

- Intensity features: mean, variance
- Texture features from GLCM: contrast, energy, homogeneity
- Shape features: area, perimeter

These features are compact, interpretable, and well-suited to traditional ML.

### 4. Classification

The SVM predicts one of the following classes:

- `0 = Normal`
- `1 = Crack`
- `2 = Pothole`
- `3 = Severe Pothole`

## Installation

This project already includes a portable Python runtime inside the project
folder, so you can run it even if Python is not available globally on PATH.

### Option 1: Use the included portable runtime

Run commands with:

```bash
python-runtime\python.exe src\main.py train
python-runtime\python.exe src\main.py predict --image dataset\crack\sample.jpg
```

### Option 2: Use your own Python installation

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

Place your images into the following folders:

```text
dataset/
│── normal/
│── crack/
│── pothole/
│── severe/
```

Example:

- `dataset/normal/img1.jpg`
- `dataset/crack/img2.jpg`
- `dataset/pothole/img3.jpg`
- `dataset/severe/img4.jpg`

## How to Run

### Frontend Dashboard

This project now includes a Streamlit frontend for:

- dataset overview
- model training
- single-image upload
- prediction visualization
- saved output preview
- mobile-friendly camera scanning
- prediction history and health score
- overlay view for detected defects

Run the frontend with:

```bash
python-runtime\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Or use your own Python installation:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

To open it on another device on the same Wi-Fi, use:

```text
http://YOUR-PC-IP:8501
```

### Train the Model

Run the training pipeline:

```bash
python-runtime\python.exe src\main.py train
```

Optional arguments:

```bash
python-runtime\python.exe src\main.py train --dataset dataset --model models/svm_model.joblib
```

If your dataset has enough images, the script also prints a validation accuracy
and classification report automatically.

### Predict a Single Image

After training, predict the class of one road image:

```bash
python-runtime\python.exe src\main.py predict --image dataset/crack/sample.jpg
```

Optional arguments:

```bash
python-runtime\python.exe src/main.py predict --image path/to/image.jpg --model models/svm_model.joblib --output outputs/result.png
```

## Sample Output Explanation

The prediction window displays:

- Original road image
- Processed grayscale image
- Segmented defect mask
- Predicted class label
- Estimated damage percentage
- Severity based on segmented defect area

Console output also prints class probabilities when available.

In the frontend, you also get:

- dataset image counts by class
- one-click training and retraining
- uploaded image preview
- batch image testing
- camera capture mode
- defect overlay view
- extracted feature values
- class probability chart
- confusion matrix after validation
- downloadable prediction report and batch CSV
- training history table
- prediction history table and health score
- saved output file paths

## Deployment

This project is now deployment-ready.

### Option 1: Streamlit Community Cloud

1. Push this project to GitHub
2. Open Streamlit Community Cloud
3. Select the repository
4. Set main file to `app.py`
5. Deploy

### Option 2: Docker-based hosting

Use the included `Dockerfile` on platforms like Render, Railway, or any VPS.

Build and run locally:

```bash
docker build -t smart-pothole-app .
docker run -p 8501:8501 smart-pothole-app
```

## Extra Feature

This project includes a simple severity estimation module:

- Low
- Moderate
- High
- Critical

It also reports the percentage of damaged pixels relative to the road ROI.

## Notes

- This is a classical image processing + machine learning project.
- It does not use deep learning.
- Real-world accuracy depends strongly on dataset quality and image diversity.
- For best results, use balanced samples for each class.
- When the dataset is very small, the project skips validation split and trains
  on all available samples.

## Future Improvements

- Add more robust road ROI extraction
- Improve segmentation for shadows and wet roads
- Add cross-validation and accuracy reporting
- Export predictions to CSV or GUI dashboards
