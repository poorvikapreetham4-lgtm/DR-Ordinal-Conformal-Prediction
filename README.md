
# Diabetic Retinopathy Grading Using Ordinal Conformal Prediction

## 1. Project Overview

Diabetic Retinopathy (DR) is an eye disease caused by damage to the blood vessels in the retina due to diabetes. If it is not detected and treated early, it can lead to vision loss.

This project explores an AI-based approach to diabetic retinopathy grading using retinal fundus images. It uses an EfficientNet-B0 deep learning model to predict the severity of DR and applies conformal prediction methods to estimate uncertainty in the model's predictions.

Instead of providing only one predicted grade, conformal prediction generates a set of possible grades. This can help communicate uncertainty and support safer interpretation of AI predictions.

**Important:** This is an academic research prototype, not a clinically validated diagnostic system.

## 2. Problem Statement

Deep learning models can classify diabetic retinopathy from retinal fundus images. However, a model may produce an incorrect prediction with high confidence.

A single predicted grade does not always communicate uncertainty. This project investigates conformal prediction techniques that produce prediction sets, providing additional information about possible DR severity grades.

## 3. Objectives

- Develop a deep learning model for DR severity grading.
- Use EfficientNet-B0 for image classification.
- Implement four conformal prediction approaches:
  - Least Ambiguous Classification (LAC)
  - Adaptive Prediction Sets (APS)
  - Regularized Adaptive Prediction Sets (RAPS)
  - Ordinal Conformal Prediction
- Evaluate prediction coverage and average prediction set size.
- Study the continuity of prediction sets for ordinal DR grades.
- Compare the performance of different conformal methods.

## 4. DR Severity Grades

The project considers five DR severity classes:

| Grade | Description |
|---|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

These grades represent increasing disease severity.

## 5. Dataset

The project uses the APTOS 2019 Blindness Detection dataset.

Dataset information:

- Training images: 2,930
- Validation images: 366
- Test images: 366
- Total images: 3,662

The dataset contains retinal fundus images labelled with DR severity grades from 0 to 4.

The dataset is not included in this GitHub repository. Users must obtain it separately and arrange the files according to the project directory structure.

## 6. Model Architecture

**EfficientNet-B0** is used as the image classification backbone.

The model processes retinal fundus images and produces probability scores for the five DR severity classes.

The trained model's probability outputs are then used by the conformal prediction methods.

The initial prototype used ImageNet-pretrained EfficientNet-B0 weights, a 224 × 224 image size, a frozen feature backbone and one training epoch.

## 7. Conformal Prediction Methods

### 7.1 LAC — Least Ambiguous Classification

LAC calculates the nonconformity score using:

`1 - probability of the true class`

A calibration threshold is calculated from the validation data. Classes satisfying the prediction-set threshold are included in the final prediction set.

### 7.2 APS — Adaptive Prediction Sets

APS uses the cumulative sum of predicted class probabilities after sorting classes by probability.

It includes classes until the cumulative probability reaches the calibrated threshold.

### 7.3 RAPS — Regularized Adaptive Prediction Sets

RAPS extends APS by adding a regularization penalty to the cumulative probability score.

This helps control excessively large prediction sets.

### 7.4 Ordinal Conformal Prediction

DR grades have a natural severity order from 0 to 4.

The ordinal method uses this ordering to generate contiguous prediction intervals. It expands an interval by considering neighbouring grades and their probabilities.

This helps preserve the relationship between adjacent DR severity levels.

**Implementation note:** The current APS and RAPS implementations use deterministic cumulative scores. They should not be described as exact reproductions of every randomized procedure in the research paper.

## 8. Experimental Results

The following results were obtained from the project's current test evaluation.

| Method | Coverage (%) | Average Set Size | Continuity (%) |
|---|---:|---:|---:|
| LAC | 93.72 | 1.83 | 77.05 |
| APS | 97.54 | 2.93 | 57.65 |
| RAPS | 95.63 | 2.55 | 58.47 |
| Ordinal | 92.62 | 2.14 | 100.00 |

### Observations

- APS achieved the highest overall empirical coverage at 97.54%, but also produced the largest average prediction set.
- LAC produced the smallest average prediction set size of 1.83.
- RAPS provided an intermediate balance between coverage and set size.
- Ordinal Conformal Prediction achieved 100% continuity because its prediction sets are constructed as contiguous intervals.
- The ordinal method achieved 88.24% coverage for the Severe DR class, compared with 47.06% for LAC, 76.47% for APS and 58.82% for RAPS.
- However, ordinal coverage for Proliferative DR was 42.42%, showing that its performance was not equally strong across all classes.

### Class-Wise Coverage (%)

| DR Grade | LAC | APS | RAPS | Ordinal |
|---|---:|---:|---:|---:|
| No DR | 99.50 | 100.00 | 100.00 | 99.50 |
| Mild | 90.00 | 93.33 | 93.33 | 96.67 |
| Moderate | 97.70 | 98.85 | 98.85 | 95.40 |
| Severe | 47.06 | 76.47 | 58.82 | 88.24 |
| Proliferative DR | 75.76 | 93.94 | 81.82 | 42.42 |

**Result interpretation:** No single method performed best across every metric and every DR grade. The choice of method involves a trade-off between coverage, prediction-set size and continuity.

## 9. Technologies Used

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- Jupyter Notebook
- Git and GitHub

## 10. Project Structure

```text
DR_Ordinal_Project/
│
├── datasets/
│   └── README.md
│
├── outputs/
│   ├── aps_threshold.npy
│   ├── lac_threshold.npy
│   ├── ordinal_threshold.npy
│   ├── raps_calibration.npz
│   ├── class_wise_coverage.csv
│   ├── conformal_comparison.csv
│   └── confusion_matrix.png
│
├── src/
│   ├── model.py
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_conformal.py
│   ├── conformal_lac.py
│   ├── conformal_aps.py
│   ├── conformal_raps.py
│   ├── conformal_ordinal.py
│   ├── lac_predict.py
│   ├── aps_predict.py
│   ├── raps_predict.py
│   ├── ordinal_predict.py
│   └── view_dataset.py
│
├── download_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 11. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/poorvikapreetham4-lgtm/DR-Ordinal-Conformal-Prediction.git
cd DR-Ordinal-Conformal-Prediction
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download and arrange the dataset

Obtain the APTOS 2019 dataset separately and arrange the image folders and CSV files according to the paths expected by the dataset loader.

### Step 4: Train the model

```bash
python src/train.py
```

### Step 5: Evaluate the model

```bash
python src/evaluate.py
```

### Step 6: Evaluate conformal prediction methods

```bash
python src/evaluate_conformal.py
```

Check the source files for any dataset-path or checkpoint configuration that needs to be adjusted before running these commands.

## 12. Limitations

- The current model is an initial prototype trained for only one epoch.
- The feature backbone was frozen during the initial training.
- The dataset is relatively small for a medical imaging application.
- Severe and Proliferative DR have fewer test samples than No DR.
- Class-wise coverage estimates can be sensitive to small sample sizes.
- The current validation data was used during model development and calibration, so it is not a fully independent calibration set.
- External dataset validation has not yet been completed.
- The system is not approved for clinical diagnosis or treatment decisions.

## 13. Future Scope

- Train the model for more epochs and tune hyperparameters.
- Improve minority-class performance.
- Use an independent calibration dataset.
- Evaluate on external datasets such as IDRiD and Messidor-2.
- Investigate distribution shift and calibration reliability.
- Compare randomized APS and RAPS implementations.
- Develop a simple interface for uploading retinal images and viewing prediction sets.
- Explore deployment on mobile devices for research purposes.

## 14. Disclaimer

This project is developed for academic and research purposes. It is not a substitute for professional ophthalmological examination, diagnosis or treatment.

## 15. Author

Developed as an academic project exploring deep learning and uncertainty estimation for diabetic retinopathy grading.

GitHub: https://github.com/poorvikapreetham4-lgtm/DR-Ordinal-Conformal-Prediction

