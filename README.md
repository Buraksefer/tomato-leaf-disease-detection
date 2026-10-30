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

Expected local dataset structure (not included in this repository):

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
git clone https://github.com/Buraksefer/tomato-leaf-disease-detection.git
cd tomato-leaf-disease-detection
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

The repository contains the source code and saved experiment outputs. Individual
report and image files are summarized by their containing directories below.

```text
tomato-leaf-disease-detection/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── outputs/
│   ├── comparison/          # Comparison charts, summary tables and winner report
│   ├── confusion_matrices/  # NumPy confusion matrices and doya.py inspection script
│   ├── gradcam/             # Saved Grad-CAM visualization
│   ├── history/             # Per-model training history in JSON format
│   └── results/             # JSON/TXT evaluation reports and summary_metrics.csv
└── src/
    ├── app.py              # Main Streamlit application with Gemini advice
    ├── app1.py             # Alternative Streamlit interface
    ├── compare_results.py  # Model comparison and visualization
    ├── dataset.py          # Data loading and image transforms
    ├── evaluate.py         # Model evaluation
    ├── external_test_swin.py # External evaluation script (currently uses DenseNet)
    ├── gemini_advice.py    # Optional Gemini API integration
    ├── gradcam.py          # Grad-CAM script
    ├── gradcam_utils.py    # Grad-CAM utilities used by the application
    ├── model_factory.py    # Model selection and construction
    ├── models/
    │   ├── basic_cnn.py
    │   ├── deeper_cnn.py
    │   ├── deit_transfer.py
    │   ├── densenet_transfer.py
    │   ├── improved_cnn.py
    │   ├── mobilenet_transfer.py
    │   ├── resnet_transfer.py
    │   ├── swin_transfer.py
    │   └── vit_transfer.py
    └── train.py            # Model training
```

The following paths are local only and excluded by `.gitignore`:

| Path | Purpose |
| --- | --- |
| `.env` | Private Gemini API key |
| `Data/` | Training, validation, test and external datasets |
| `outputs/models/` | Trained model weights (`.pth` files) |
| `ai_torch_new/`, `.venv/` | Local Python environments |
| `outputs/logs/` | Local execution logs |

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
