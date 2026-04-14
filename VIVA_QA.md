# Viva Questions and Answers

## 1. What is the aim of this project?

The aim is to detect road defects such as cracks and potholes from images using
digital image processing and classify them using SVM.

## 2. Why did you use grayscale conversion?

Grayscale reduces color complexity and makes intensity-based processing easier
and faster.

## 3. Why is histogram equalization used?

Histogram equalization improves contrast, which helps reveal cracks and dark
defect regions more clearly.

## 4. Why are Gaussian blur and median filter both used?

Gaussian blur removes general high-frequency noise, while median filtering is
good for removing isolated noisy pixels without damaging edges too much.

## 5. What is the purpose of sharpening?

Sharpening highlights edges and makes crack and pothole boundaries more
visible.

## 6. Why did you use Canny edge detection?

Canny is effective for detecting thin edges and boundaries, which is useful for
cracks and pothole outlines.

## 7. What are morphological operations?

Morphological operations such as dilation and erosion modify shapes in a binary
image to connect broken regions and remove small noise.

## 8. What is ROI in this project?

ROI means Region of Interest. Here it refers to the probable road area from
which defects are extracted.

## 9. What features are used for classification?

The project uses intensity features, GLCM texture features, and shape features.

## 10. What is GLCM?

GLCM stands for Gray Level Co-occurrence Matrix. It describes how often
different gray-level values occur together and is used for texture analysis.

## 11. Why did you choose SVM?

SVM performs well on small to medium-sized datasets, works effectively with
handcrafted features, and is easier to explain than deep learning models.

## 12. What are the output classes?

- Normal
- Crack
- Pothole
- Severe Pothole

## 13. What is damage percentage?

It is the estimated percentage of defective pixels relative to the selected
road region.

## 14. What are the limitations of the project?

The project may struggle with poor lighting, shadows, reflections, or highly
complex road scenes. Accuracy also depends on the dataset quality.

## 15. How can this project be improved in future?

It can be improved using larger datasets, better segmentation methods, video
analysis, GPS integration, or advanced machine learning models.

