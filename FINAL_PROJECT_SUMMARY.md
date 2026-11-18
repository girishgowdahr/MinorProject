# AI-Based Visual Inspection and Defect Detection of Electrical Distribution Infrastructure Using YOLO

## Project Overview
This project uses a YOLO object detection model to identify and classify visual conditions of electrical distribution crossarms and top-cleats from images.

## Problem Statement
Manual inspection of electrical distribution infrastructure can be time-consuming. This project explores computer vision as a supporting tool for identifying visible structural conditions in inspection images.

## Dataset
Dataset: EPDCL/pd-defect
Scope: Crossarm and top-cleat object detection

Classes:
1. crossarm_straight
2. crossarm_tilted
3. topcleat_straight
4. topcleat_tilted
5. v-crossarm_straight
6. v-crossarm_tilted

Dataset split:
- Training images: 1175
- Validation images: 254
- Test images: 257
- Total images: 1686

## Model
- Model: YOLOv8n
- Image size: 640
- Training epochs: 5

## Test Results
- Precision: 0.6313
- Recall: 0.7855
- mAP@50: 0.7727
- mAP@50–95: 0.5964
- Recorded evaluation inference time: 6.33 ms/image

These values are from the recorded experiment. Performance may vary with hardware and input images.

## Application
A Flask web application provides an image upload interface and displays object detection results with bounding boxes and confidence scores.

## Testing
- Valid image upload and detection: tested
- Missing or invalid image handling: tested
- Application response for invalid uploads: HTTP 400
- Final project ZIP integrity: verified

## Limitations
The system detects only the six dataset classes. It does not perform a complete inspection of electrical poles, detect every possible defect, or replace professional field inspection. Results depend on image quality and dataset coverage.

## Future Work
- Expand the dataset with additional defect categories.
- Evaluate on more real-world images.
- Improve model performance through further experiments.
- Explore drone-assisted image collection and IoT integration.
- Add reporting and inspection history features.

## How to Run
Install dependencies listed in requirements.txt, then run:

python app.py

Open the Flask application at:

http://127.0.0.1:5000
