# Dataset Guide

Add road images into these folders before training:

- `normal/` for healthy road surface images
- `crack/` for images containing visible cracks
- `pothole/` for images containing potholes
- `severe/` for images containing large or severe potholes

## Good Dataset Practices

- Use clear road images with minimum blur
- Keep image sizes reasonably similar if possible
- Try to collect balanced samples for all four classes
- Include different lighting conditions for better generalization
- Avoid mixing unrelated objects that dominate the frame

## Suggested Minimum

For a simple academic demo:

- 20 to 30 images per class is a reasonable starting point

For better results:

- 100 or more images per class is preferred

## Example

```text
dataset/
│── normal/
│   │── normal_01.jpg
│   │── normal_02.jpg
│── crack/
│   │── crack_01.jpg
│── pothole/
│   │── pothole_01.jpg
│── severe/
│   │── severe_01.jpg
```

