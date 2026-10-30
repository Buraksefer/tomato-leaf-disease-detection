# 🍅 Tomato Leaf Disease Classification Using CNN, Transfer Learning and Vision Transformers

## 📌 Project Overview

This project aims to automatically classify tomato leaf diseases using deep learning techniques. Different model families including custom CNN architectures, Transfer Learning models, and Transformer-based architectures were trained and compared on a tomato leaf disease dataset.

The study evaluates the effectiveness of:

* CNN-based models
* Transfer Learning models
* Vision Transformer models

for the classification of tomato leaf diseases.

---

## 🎯 Objectives

* Detect tomato leaf diseases from images
* Compare CNN, Transfer Learning and Transformer architectures
* Analyze model performance using multiple evaluation metrics
* Visualize model decisions using Grad-CAM
* Deploy a user-friendly Streamlit application

---

## 📂 Dataset

The dataset contains 10 classes:

1. Tomato Bacterial Spot
2. Tomato Early Blight
3. Tomato Late Blight
4. Tomato Leaf Mold
5. Tomato Septoria Leaf Spot
6. Tomato Spider Mites
7. Tomato Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy Tomato Leaf

Dataset Structure:

```text
Data/
└── tomato/
    ├── train/
    ├── val/
    └── test/
```

---

## 🔄 Data Preprocessing

The following preprocessing and augmentation techniques were applied:

* Resize (224 × 224)
* Random Horizontal Flip
* Random Rotation
* Color Jitter
* Normalization

---

## 🧠 Models

### CNN Models

* Basic CNN
* Improved CNN
* Deeper CNN

### Transfer Learning Models

* MobileNetV2
* DenseNet121
* ResNet50

### Transformer Models

* Vision Transformer (ViT)
* DeiT
* Swin Transformer

---

## 📊 Performance Comparison

| Model              | Accuracy (%) |
| ------------------ | -----------: |
| Swin Transformer   |        99.90 |
| Vision Transformer |        99.20 |
| ResNet50           |        99.10 |
| DenseNet121        |        98.90 |
| Deeper CNN         |        98.90 |
| DeiT               |        98.80 |
| MobileNetV2        |        98.30 |
| Improved CNN       |        94.60 |
| Basic CNN          |        78.60 |

---

## 📈 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 🔥 Grad-CAM Visualization

Grad-CAM was used to visualize the regions of tomato leaves that contributed most to model predictions.

---

## 💻 Streamlit Application

Run the web application:

```bash
streamlit run src/app.py
```

---

## ⚙️ Installation

The dataset and trained weights are not included in Git. Restore the dataset
under `Data/tomato/{train,val,test}/` for training and evaluation. For inference,
place trusted trained weights in `outputs/models/best_<model_name>.pth` (for
example `best_swin.pth`). The Grad-CAM view also needs `best_deeper_cnn.pth`.
Run commands from the project root. A fresh clone cannot classify images until
the required model weights have been provided or trained.

Clone the repository:

```bash
git clone https://github.com/your_username/tomato-disease-classification.git
cd tomato-disease-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional Gemini advice: copy `.env.example` to `.env` and set your own
`GEMINI_API_KEY`. Keep `.env` private. The classifier can run without a Gemini
key; generated advice requires a valid key and network access.

---

## 📁 Project Structure

```text
Tomato_Disease_Project/
│
├── Data/
├── outputs/
│   ├── models/
│   ├── results/
│   └── confusion_matrices/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── model_factory.py
│   ├── app.py
│   └── gradcam_utils.py
│
├── requirements.txt
└── README.md
```

---

## 🏆 Best Model

The Swin Transformer achieved the highest classification performance with:

* Accuracy: 99.90%
* Precision: 99.90%
* Recall: 99.90%
* F1-Score: 99.90%

---

## 👨‍💻 Author

Burak Sefer

Computer Engineering Department

Kastamonu University
