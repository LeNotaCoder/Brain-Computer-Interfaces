# 🧠 EEG Motor Imagery Classification using CNN-LSTM Transfer Learning

**Inspired by:**  
📄 *A transfer learning-based CNN and LSTM hybrid deep learning model to classify motor imagery EEG signals*  
**(Khademi et al., 2022, Biomedical Signal Processing and Control, Elsevier)**  

---

## 📘 Overview

This repository provides an **exact implementation and training pipeline** for the EEG-based motor imagery classification model proposed by Khademi *et al.* (2022).  
The model combines **Continuous Wavelet Transform (CWT)**–based feature extraction, **Convolutional Neural Networks (CNN)**, and **Long Short-Term Memory (LSTM)** layers to classify EEG signals corresponding to four motor imagery tasks:

- 🖐 Left hand  
- ✋ Right hand  
- 🦶 Feet  
- 👅 Tongue  

Three models are supported:

| Model | Description | Reported Accuracy | Reported Kappa |
|:------|:-------------|:------------------|:---------------|
| **Custom CNN-LSTM** | Trained from scratch | 86% | 0.81 |
| **ResNet50-LSTM** | Transfer learning with ResNet50 | 90% | 0.86 |
| **InceptionV3-LSTM** | Transfer learning with Inception-v3 | 92% | 0.88 |

---

## 🧩 Methodology Overview

### 1. Dataset
- **BCI Competition IV Dataset 2a**  
- 9 subjects × 4 classes (LH, RH, Feet, Tongue)  
- 22 EEG channels, sampled at **250 Hz**  
- Each trial: 6 seconds (1500 samples)  

### 2. Preprocessing  

| Step | Description |
|------|--------------|
| **CAR** | Apply Common Average Referencing (Eq. 1) |
| **Bandpass Filter** | 5th-order zero-phase Butterworth, 8–30 Hz |
| **Windowing** | Use EEG segment from **2–6 seconds** (1000 samples) |
| **Crop Augmentation** | 5 non-overlapping 0.8 s crops per trial |

### 3. Feature Transformation  
Each EEG crop is transformed into a **scalogram** using **Complex Morlet Wavelet (`cmor1.5-1.0`)**.  
The resulting time–frequency maps are resized to **224×224** (ResNet) or **299×299** (Inception).

### 4. Model Architectures  

#### 🧱 Custom CNN-LSTM
- 2 Conv–ELU–MaxPool blocks  
- 1 LSTM (250 units)  
- FC layers (100 → 4)  

#### 🧠 Transfer Learning Models
- Feature extraction using **ResNet50** or **Inception-v3** pretrained on ImageNet  
- Freeze all convolutional layers  
- LSTM (250 units) + FC (100 → 4) classifier head  

### 5. Training Setup
| Parameter | Value |
|:----------|:-------|
| Batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | 15 epochs |
| Validation split | 20% |
| Metric | Accuracy, Cohen’s Kappa |

---


---

## 🧮 Running the Code

### 1. Install dependencies
```bash
pip install torch torchvision numpy pandas scipy scikit-learn seaborn matplotlib pywt tqdm
```

