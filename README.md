# 🧠 Brain Tumor Classification with Deep Learning and XAI

A comprehensive deep learning project for automated brain tumor classification using MRI images, implementing state-of-the-art CNN architectures with Explainable AI (XAI) techniques for medical diagnosis transparency.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/shuvokumarbasakbd/brain-tumors-mri-crystal-clean-colorized-mri-data)

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results](#-results)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [License](#-license)

## 🎯 Overview

This project implements advanced deep learning techniques for classifying brain tumors from MRI scans into four categories:
- **Normal** - No tumor detected
- **Glioma Tumor** - Malignant brain tumor
- **Meningioma Tumor** - Typically benign tumor
- **Pituitary Tumor** - Tumor in the pituitary gland

The project emphasizes both high accuracy and interpretability through Explainable AI techniques, making it suitable for medical diagnostic support systems.

## ✨ Features

### Deep Learning Models
- **Transfer Learning**: Implementation of 5 state-of-the-art CNN architectures pre-trained on ImageNet:
  - VGG19 ⭐ **Best Performing Model** (91.70% accuracy)
  - ResNet50
  - ResNet152
  - EfficientNetB0
  - InceptionV3
- **Fine-tuning**: Partial unfreezing of layers for domain adaptation
- **Custom Architecture**: Tailored classification heads with dropout regularization

### Data Processing
- **Comprehensive Data Augmentation**:
  - Random horizontal flips
  - Brightness adjustments
  - Random rotation (±0.2 radians)
  - Random zoom (1.0-1.2x)
- **Balanced Dataset**: Equal distribution across all classes (1,400 images per class)
- **Efficient Pipeline**: TensorFlow data pipeline with caching and prefetching

### Model Training
- **Early Stopping**: Prevents overfitting with patience-based monitoring
- **Model Checkpointing**: Automatic saving of best-performing models
- **Learning Rate Optimization**: Adam optimizer with fine-tuned learning rate

### Evaluation & Visualization
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Detailed visualization of classification performance
- **Training History**: Loss and accuracy curves for both training and validation

### Explainable AI (XAI)
- **LIME** (Local Interpretable Model-agnostic Explanations): Local feature importance
- **SHAP** (SHapley Additive exPlanations): Game theory-based feature attribution
- **Grad-CAM** (Gradient-weighted Class Activation Mapping): Visual explanations via gradients
- **Grad-CAM++**: Enhanced localization with weighted activation maps
- Visual explanations for clinical decision support and model transparency

## 📊 Dataset

- **Source**: [Kaggle - Brain Tumors MRI Crystal Clean Colorized](https://www.kaggle.com/datasets/shuvokumarbasakbd/brain-tumors-mri-crystal-clean-colorized-mri-data)
- **License**: MIT
- **Total Images**: 5,600
- **Image Size**: 224×224×3 (RGB)
- **Classes**: 4 (Normal, Glioma, Meningioma, Pituitary)
- **Distribution**: Balanced with 1,400 images per class

### Data Split
- **Training Set**: 4,032 images (72%)
- **Validation Set**: 448 images (8%)
- **Test Set**: 1,120 images (20%)

## 🏗️ Models

### VGG19 Architecture ⭐ (Best Model - 91.70% Accuracy)
- **Base Model**: VGG19 pre-trained on ImageNet
- **Trainable Layers**: Last 10 layers unfrozen
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.5)
  - Dense(4, activation='softmax')

### ResNet50 Architecture
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Transfer Learning**: Partially fine-tuned for brain tumor classification
- **Similar custom classification head**

### ResNet152 Architecture
- **Base Model**: ResNet152 pre-trained on ImageNet
- **Deeper Architecture**: 152 layers for enhanced feature extraction
- **Transfer Learning**: Domain-adapted for medical imaging

### EfficientNetB0 Architecture
- **Base Model**: EfficientNetB0 pre-trained on ImageNet
- **Compound Scaling**: Balanced depth, width, and resolution
- **Efficient Architecture**: Optimized for performance and speed

### InceptionV3 Architecture
- **Base Model**: InceptionV3 pre-trained on ImageNet
- **Multi-scale Processing**: Inception modules for varied receptive fields
- **Transfer Learning**: Adapted for tumor classification

## 📈 Results

### VGG19 Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 91.70% |
| **Weighted Precision** | 91.80% |
| **Weighted Recall** | 91.70% |
| **Weighted F1-Score** | 91.74% |

### Per-Class Performance (VGG19)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Normal** | 0.9724 | 0.9400 | 0.9559 | 300 |
| **Glioma Tumor** | 0.8881 | 0.9097 | 0.8988 | 288 |
| **Meningioma Tumor** | 0.8593 | 0.8794 | 0.8692 | 257 |
| **Pituitary Tumor** | 0.9449 | 0.9345 | 0.9397 | 275 |

### Training Details
- **Epochs Trained**: 13 (Early stopping triggered)
- **Best Validation Accuracy**: 92.41%
- **Training Time**: ~50 seconds per epoch
- **Callbacks**: EarlyStopping (patience=3), ModelCheckpoint

## 💻 Usage

> **⚠️ Recommended: Use Google Colab**  
> This project is optimized for Google Colab with free GPU access. Running locally requires significant computational resources and CUDA-capable GPU.

### Option 1: Google Colab (Recommended) 🌟

1. **Open Notebooks in Colab:**
   
   [![Open CNN+XAI in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ghorbel37/brain-tumor-classification-cnn-xai/blob/main/Brain_Tumor_Classification_CNN_XAI.ipynb)
   
   [![Open ML in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ghorbel37/brain-tumor-classification-cnn-xai/blob/main/Brain_Tumor_Classification_ML.ipynb)
   
2. **Upload your `kaggle.json`:**
   - When prompted in the notebook, upload your Kaggle API credentials
   - The notebook will automatically configure, download, and extract the dataset

3. **Enable GPU:**
   - Go to `Runtime` → `Change runtime type`
   - Select `GPU` as Hardware accelerator
   - Click `Save`

**Getting Kaggle API Credentials:**
1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json` for upload to Colab

---

### Option 2: Run Locally (Advanced Users)

**Requirements:**
- Python 3.12+
- CUDA-capable GPU (NVIDIA)
- 16GB+ RAM recommended
- ~10GB free disk space

**Steps:**
```bash
# Clone repository
git clone https://github.com/Ghorbel37/brain-tumor-classification-cnn-xai.git
cd brain-tumor-classification-cnn-xai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place kaggle.json in project folder
# Run notebooks
jupyter notebook Brain_Tumor_Classification_CNN_XAI.ipynb
```

### 3. Load Pre-trained Model
```python
from tensorflow.keras.models import load_model

# Load the best VGG19 model
model = load_model('best_modelVGG19.keras')

# Make predictions
predictions = model.predict(test_dataset)
```

### 4. Evaluate Model
```python
# Get predictions
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred_classes))
```

## 📁 Project Structure

```
brain-tumor-classification-cnn-xai/
├── Brain_Tumor_Classification_CNN_XAI.ipynb    # Main CNN implementation with XAI
├── Brain_Tumor_Classification_ML.ipynb         # Traditional ML approaches
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## 🛠️ Technologies

### Core Frameworks
- **TensorFlow 2.19.0** - Deep learning framework
- **Keras** - High-level neural networks API
- **TensorFlow Addons** - Extended TensorFlow functionality

### Explainable AI (XAI) Libraries
- **LIME** - Local interpretable model-agnostic explanations
- **SHAP** - SHapley Additive exPlanations
- **tf-keras-vis** - Grad-CAM and Grad-CAM++ visualizations

### Data Science Stack
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities

### Visualization
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical visualizations

### Image Processing
- **OpenCV** - Computer vision library
- **Pillow (PIL)** - Image manipulation

### Utilities
- **tqdm** - Progress bars
- **Kaggle API** - Dataset management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>⭐ If you find this project helpful, please consider giving it a star! ⭐</strong>
</div>
