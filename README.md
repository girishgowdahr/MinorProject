# YOLO Electrical Distribution Infrastructure Inspection

## Overview
This project detects six crossarm and top-cleat condition classes in electrical distribution infrastructure images using YOLOv8n. A Flask web interface accepts JPG/PNG images and displays predicted boxes, class names, and confidence scores.

The application is an assistance prototype. Human review is required; it is not a complete electrical safety inspection.

## Dataset
Dataset: EPDCL/pd-defect, crossarm/top-cleat object-detection subset.
Train: 1,175 images | Validation: 254 | Test: 257 | Total: 1,686

Classes:
- crossarm_straight
- crossarm_tilted
- topcleat_straight
- topcleat_tilted
- v-crossarm_straight
- v-crossarm_tilted

Check and cite the dataset licence in the academic report.

## Model and Recorded Test Results
Model: YOLOv8n | Image size: 640 | Epochs: 5

| Metric | Value |
|---|---:|
| Precision | 0.6313 |
| Recall | 0.7855 |
| mAP@0.50 | 0.7727 |
| mAP@0.50:0.95 | 0.5964 |
| Recorded evaluation inference speed | 6.33 ms/image |

Qualitative sample predictions are not a separate accuracy measurement.

## Web App
- Uploads JPG/PNG images
- Default confidence threshold: 0.25
- Shows detections, confidence scores, and inference time
- Rejects missing/unreadable images with HTTP 400
- Maximum request size: 15 MB

## Setup
Install dependencies:
`pip install -r requirements.txt`

PyTorch should be installed for the target CPU or CUDA environment.

## Run
`python app.py --weights runs/detect/runs/exp1_baseline/weights/best.pt --port 5000`

Open http://127.0.0.1:5000

In Colab:
`output.serve_kernel_port_as_iframe(5000, height=800)`

## Key Files
- app.py: Flask API and inference
- templates/index.html: upload interface
- data/: dataset and YAML
- runs/detect/runs/exp1_baseline/weights/best.pt: trained weights
- results/experiment_results.csv: recorded metrics
- results/sample_predictions.csv: qualitative examples

## Limitations
The model detects only the six dataset-defined classes. It may miss or misclassify objects. Qualified human review is required before maintenance or safety decisions.
