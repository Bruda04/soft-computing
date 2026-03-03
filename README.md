# Soft Computing Project

This repository contains three practical exercises for the course "Soft Computing," each focused on applying neural networks and image processing techniques to real-world problems. The exercises are organized into separate folders, each with its own dataset and implementation notebook.

---

## 1. OCR: Handwritten Text Recognition using ANN

**Folder:** `ocr/`

This exercise implements an Optical Character Recognition (OCR) pipeline for reading handwritten or printed text from images. The system:

- **Preprocesses** images (grayscale, deskew, binarize, erode/dilate)
- **Segments** individual characters via contour detection and region merging
- **Trains** a fully-connected Artificial Neural Network (ANN) on the segmented characters
- **Predicts** text for each image and evaluates using a Hamming-based distance metric

**Dataset:**

- Images and ground truth texts are listed in `data/texts.csv`.

![Sample OCR Image](./ocr/data/picture_3.png)

---

## 2. Paw Patrol Bone Counter

**Folder:** `paw_patrol_bone_counter/`

This exercise focuses on detecting and counting bones in video frames using image processing and neural networks. The pipeline:

- Processes video frames to detect bone-like shapes
- Applies segmentation and feature extraction
- Trains a neural network to count bones in each frame

**Dataset:**

- Frame data and bone counts are stored in `data/count.csv`.

![Sample Bone Detection](./paw_patrol_bone_counter/data/video2.gif)

---

## 3. Super Mario Coin Counter

**Folder:** `super_mario_coin_counter/`

This exercise aims to detect and count coins in Super Mario game screenshots. The workflow includes:

- Preprocessing images to highlight coins
- Segmenting and extracting coin regions
- Using a neural network to count and classify coin values

**Dataset:**

- Coin counts and values are in `data/coin_value_count.csv`.

![Sample Coin Detection](./super_mario_coin_counter/data/image_3.jpg)

---

## Getting Started

Each exercise contains a Jupyter notebook (`*.ipynb`) and a `requirements.txt` file listing dependencies. To run an exercise:

1. Navigate to the exercise folder (e.g., `ocr/`).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook in Jupyter or VS Code and run the cells.

---

## Project Structure

```text
README.md
ocr/
 ocr_ann.ipynb
 requirements.txt
 data/
  texts.csv
  sample_ocr_image.png
paw_patrol_bone_counter/
 bone_detection_video.ipynb
 requirements.txt
 data/
  count.csv
  sample_bone_frame.png
super_mario_coin_counter/
 coin_detection.ipynb
 requirements.txt
 data/
  coin_value_count.csv
  sample_coin_image.png
```

---

## Author

- [Luka Bradić](https://github.com/bruda04)
