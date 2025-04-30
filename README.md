# Potato Defect Detection using YOLOv11s

This project demonstrates the process of training a YOLOv11s object detection model to identify different types of potatoes, including damaged, defected, and sprouted ones, using a custom dataset. The training and prediction are performed using the Ultralytics YOLO library, leveraging a GPU environment (Colab).

## Project Overview

The goal of this project is to build a system capable of detecting and classifying potatoes based on their condition. This could be useful in agricultural sorting, quality control, or research applications.

**Classes Detected:**
1.  Damaged Potato
2.  Defected Potato
3.  Potato (Healthy)
4.  Sprouted Potato

## Features

*   **Model:** Utilizes the YOLOv11s architecture via the Ultralytics library.
*   **Dataset:** Trained on a custom dataset of potato images (details below).
*   **Task:** Object Detection.
*   **Environment:** Designed and tested in a Google Colab environment with GPU acceleration (Tesla T4).
*   **Training:** Includes steps for data preparation, configuration, training, and validation.
*   **Prediction:** Demonstrates how to use the trained model for inference on new images.

## Dataset

The project uses a custom dataset (`data.zip`) containing images of potatoes and corresponding annotations in YOLO format (`.txt` files).

*   **Preparation:** The raw dataset is initially located in `/content/custom_data` after unzipping.
*   **Splitting:** The `train_val_split.py` script is used to split the dataset into training (90%) and validation (10%) sets, located in `/content/data/train` and `/content/data/validation` respectively.
*   **Configuration:** A `data.yaml` file is generated to configure the dataset paths and class information for the YOLO model.

```yaml
# Example data.yaml
path: /content/data
train: train/images
val: validation/images
nc: 4
names:
- Damaged Potato
- Defected Potato
- Potato
- Sprouted Potato
