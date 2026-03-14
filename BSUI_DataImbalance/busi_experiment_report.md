
# BUSI Dataset Classification — Imbalance Handling Study

## Project Overview
This project investigates how different **class imbalance handling techniques** affect the performance of a deep learning model on the **BUSI (Breast Ultrasound Images) dataset**.

Breast ultrasound images are classified into three categories:

- **Benign**
- **Malignant**
- **Normal**

Because the dataset is **imbalanced**, accuracy alone is not a reliable metric. Therefore, the evaluation focuses on:

- **Macro F1 Score**
- **Weighted F1 Score**
- **Full Classification Reports**

These metrics better capture performance across all classes, especially minority classes such as **Malignant tumors**, which are critical for medical diagnosis.

---

# Dataset

The **BUSI dataset** contains ultrasound images labeled into three classes:

| Class | Description |
|------|-------------|
| Benign | Non‑cancerous tumor |
| Malignant | Cancerous tumor |
| Normal | No tumor present |

The dataset is **imbalanced**, meaning some classes contain more samples than others. This imbalance can bias models toward majority classes if not handled properly.

---

# Experimental Setup

## Model Architecture
All experiments used:

**ResNet‑18 (Transfer Learning)**

- Pretrained on ImageNet
- Final fully connected layer modified for **3‑class classification**
- Input size: **224×224**
- Optimizer: **Adam**
- Loss: Cross‑Entropy (or modified variants)
- Training performed using **PyTorch**

---

# Train / Validation / Test Split

A **stratified split** was used to maintain the class distribution across all sets:

- **Training:** 70%
- **Validation:** 15%
- **Test:** 15%

Stratification ensures that the **class ratios remain consistent**, which is important for fair evaluation on imbalanced datasets.

---

# Imbalance Handling Techniques Evaluated

Five training strategies were evaluated.

## 1. Baseline
Standard training using **Cross‑Entropy Loss** with no imbalance correction.

## 2. Class Weighting
Weighted Cross‑Entropy Loss where each class receives an inverse‑frequency weight.

This penalizes mistakes on minority classes more heavily.

## 3. Oversampling
Uses a **WeightedRandomSampler** to ensure minority class samples appear more frequently during training.

## 4. Data Augmentation
Applies random transformations during training:

- Random horizontal flip
- Random rotation
- Color jitter

Augmentation increases dataset diversity and improves generalization.

## 5. Focal Loss
A modified loss function that focuses training on **hard examples** by down‑weighting easy samples.

This is often used in **imbalanced classification tasks**.

---

# Evaluation Metrics

Because the dataset is imbalanced, the following metrics were used:

### Macro F1 Score
Treats each class equally and measures balanced performance.

### Weighted F1 Score
Accounts for class frequency and provides a weighted average.

### Classification Report
Includes:

- Precision
- Recall
- F1 Score
- Support for each class

---

# Experimental Results

| Method | Macro F1 | Weighted F1 |
|------|------|------|
| Baseline | 0.8086 | 0.8403 |
| Class Weighting | 0.9000 | 0.9037 |
| Oversampling | 0.9318 | 0.9367 |
| Augmentation | **0.9372** | **0.9451** |
| Focal Loss | 0.8964 | 0.9079 |

---

# Result Analysis

### Baseline
The baseline model achieved the lowest performance. This confirms that **class imbalance negatively impacts model learning** when not explicitly handled.

### Class Weighting
Applying class weights significantly improved performance by penalizing mistakes on minority classes.

### Oversampling
Oversampling further improved performance by balancing the training distribution.

### Data Augmentation (Best Performing)
Data augmentation achieved the **highest Macro F1 and Weighted F1 scores**, indicating improved generalization and better representation of all classes.

### Focal Loss
Focal Loss improved results compared to the baseline but performed slightly worse than oversampling and augmentation in this experiment.

---

# Key Findings

1. **Handling class imbalance significantly improves performance.**
2. **Data augmentation produced the best overall results.**
3. Oversampling also provided strong performance improvements.
4. Baseline models perform poorly on imbalanced datasets.

---

# Conclusion

This study demonstrates that **data imbalance must be addressed when training medical image classification models**.

Among the evaluated techniques:

**Data Augmentation + Transfer Learning provided the best performance on the BUSI dataset.**

Future improvements may include:

- Advanced augmentation (elastic deformation, CLAHE)
- Attention-based architectures
- Vision Transformers
- Ensemble models

---

# Technologies Used

- Python
- PyTorch
- Scikit‑learn
- Pandas
- Torchvision

---

# Reproducibility

To reproduce the experiments:

1. Install dependencies
2. Prepare BUSI dataset with class folders
3. Run the notebook provided in this repository
4. Evaluate results using the generated classification reports

---

# Author

Experiment conducted as part of a study on **handling class imbalance in medical image classification using deep learning**.
