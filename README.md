# Cardiovascular Disease Prediction: A Machine Learning Approach

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Summary](#pipeline-summary)
- [Technical Stack](#technical-stack)
- [Model Overview](#model-overview)
- [Evaluation & Metrics](#evaluation--metrics)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Comparison](#model-comparison)
- [Clinical Implications](#clinical-implications)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

## Project Overview

Cardiovascular diseases represent the leading cause of death worldwide, accounting for approximately **17.8 million deaths annually** according to the World Health Organization (WHO). Early identification of individuals at risk is essential for implementing preventive strategies and improving long-term patient outcomes.

This project develops a **complete machine learning pipeline** for predicting the presence of cardiovascular disease using clinical patient data. It demonstrates:

- **Robust data preprocessing** with missing value imputation and outlier detection
- **Comprehensive exploratory analysis** of clinical features and their relationships
- **Feature engineering** including normalization and selection (RFECV + Random Forest importance)
- **Comparative evaluation** of four classification algorithms
- **Fair performance assessment** using metrics suited for imbalanced medical data
- **Fully reproducible workflow** with saved models and preprocessing artifacts

The pipeline provides a clear example of designing and evaluating classifiers for medical diagnostic support, with particular attention to the **sensitivity-specificity trade-off** critical in healthcare applications.

## Dataset

**Source**: [UCI Machine Learning Repository (Heart Disease Dataset)](https://archive.ics.uci.edu/dataset/45/heart+disease)

The dataset combines patient records from four independent studies:
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology
- University Hospital Zurich, Switzerland
- VA Medical Center, Long Beach

**Dataset Characteristics**:
- **Total Samples**: 920 patient records (908 after outlier removal)
- **Features**: 13 clinical and diagnostic attributes
- **Target Classes**: 0 (absence) to 4 (presence with varying severity levels) → **Binary classification**

### Feature Description

| Feature | Type | Description | Clinical Significance |
|---------|------|-------------|----------------------|
| `age` | Numerical | Patient age in years | Progressive risk factor |
| `sex` | Binary | 1 = male, 0 = female | Men have higher baseline risk |
| `cp` | Categorical | Chest pain type (1-4) | Symptom indicator |
| `trestbps` | Numerical | Resting blood pressure (mm Hg) | Hypertension marker |
| `chol` | Numerical | Serum cholesterol (mg/dL) | Lipid profile |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dL | Diabetes indicator |
| `restecg` | Categorical | Resting ECG results | Cardiac electrical activity |
| `thalach` | Numerical | Maximum heart rate achieved | Exercise response |
| `exang` | Binary | Exercise-induced angina | Ischemia indicator |
| `oldpeak` | Numerical | ST depression during exercise | Ischemia severity |
| `slope` | Categorical | ST segment slope | Stress response pattern |
| `ca` | Numerical | Number of major vessels colored | Coronary obstruction |
| `thal` | Categorical | Thalassemia type | Perfusion status |

## 🔬 Pipeline Summary

| Step | Description |
|------|-------------|
| **1. Data Preprocessing** | Missing value imputation (median), categorical encoding, outlier removal (Z-score > 3) |
| **2. Exploratory Analysis** | Target distribution, demographic patterns, categorical feature impact, numerical feature distributions |
| **3. Data Preparation** | Train-test split (80/20 stratified), Min-Max scaling |
| **4. Feature Engineering** | RFECV with Logistic Regression, Random Forest feature importance, selection of top 10 predictors |
| **5. Model Training** | Logistic Regression, Gaussian Naive Bayes, SVM, Random Forest with GridSearchCV |
| **6. Evaluation** | Confusion matrices, ROC-AUC, sensitivity/specificity, precision, F1-score, MCC |

## 🛠️ Technical Stack

- **Data Handling**: pandas, NumPy
- **Modeling**: scikit-learn
- **Feature Selection**: RFECV, SelectKBest, Random Forest importance
- **Evaluation & Metrics**: scikit-learn, Matplotlib, Seaborn
- **Visualization**: matplotlib, seaborn
- **Reproducibility & Export**: joblib, pickle

## Model Overview

### Models Implemented

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| **Logistic Regression** | Linear classifier with probability outputs | C=10 (optimal), L2 penalty |
| **Gaussian Naive Bayes** | Probabilistic classifier assuming feature independence | var_smoothing=1e-9 |
| **SVM (RBF kernel)** | Margin-based classifier with non-linear boundary | C=4, gamma='scale' |
| **Random Forest** | Ensemble of decision trees | max_depth=5, n_estimators=100, min_samples_split=2 |

### Why These Models?

- **Logistic Regression**: Interpretable baseline, provides probability estimates
- **Gaussian Naive Bayes**: Fast, works well with continuous features, handles uncertainty
- **SVM**: Effective in high-dimensional spaces, flexible decision boundaries
- **Random Forest**: Robust to outliers, captures non-linear relationships, feature importance

## Evaluation & Metrics

Given the medical context, special attention was paid to metrics that matter in clinical settings:

| Metric | Formula | Clinical Relevance |
|--------|---------|---------------------|
| **Sensitivity (Recall)** | TP / (TP + FN) | Ability to detect diseased patients (minimize false negatives) |
| **Specificity** | TN / (TN + FP) | Ability to correctly identify healthy patients |
| **Precision** | TP / (TP + FP) | Confidence when predicting disease |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
| **MCC** | Matthews Correlation Coefficient | Balanced measure robust to imbalance |
| **ROC-AUC** | Area Under ROC Curve | Overall discriminative ability |

### Key Evaluation Techniques

- **Confusion Matrices**: Visual assessment of prediction patterns
- **ROC Curves**: Threshold-independent performance visualization
- **Cross-validation**: 5-fold stratified CV for hyperparameter tuning
- **Macro-averaged metrics**: Equal weight to both classes

## Key Findings

### Clinical Insights from EDA

1. **Age and Disease Severity**: Clear progression—healthy patients cluster at 45-65 years, severe cases predominantly occur in individuals over 65.

2. **Gender Disparity**: 
   - 78% of men have heart disease vs 12% of women
   - Men develop disease 7-10 years earlier and progress to severe stages more frequently

3. **Chest Pain**: Asymptomatic patients show 79% disease probability, highlighting that absence of symptoms does not exclude disease.

4. **ST Depression**: Strongest numerical predictor—healthy patients cluster near zero while disease patients show elevated values.

### Feature Importance (Top 10 Selected)

| Rank | Feature | Importance | Clinical Significance |
|------|---------|------------|----------------------|
| 1 | `chol` | 0.156 | Cholesterol drives atherosclerosis |
| 2 | `thalach` | 0.125 | Exercise capacity reflects cardiac function |
| 3 | `age` | 0.115 | Cumulative cardiovascular risk |
| 4 | `oldpeak` | 0.096 | Ischemia indicator |
| 5 | `exang` | 0.087 | Exercise-induced angina |
| 6 | `cp_atypical angina` | 0.082 | Atypical symptom presentation |
| 7 | `trestbps` | 0.076 | Hypertension marker |
| 8 | `thal` | 0.056 | Perfusion defects |
| 9 | `sex_male` | 0.051 | Demographic risk factor |
| 10 | `cp_non-anginal pain` | 0.043 | Non-cardiac chest pain |

## Project Structure

```
HeartDisease_ML/
│
├── datasets/                                 # Raw dataset files
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   ├── processed.va.data
│   └── heart_disease_combined.csv           # Merged and cleaned dataset
│
├── models/                                   # Trained models and artifacts
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── random_forest.joblib
│   ├── scaler.joblib                         # Fitted MinMaxScaler
│   └── feature_names.txt                     # Selected features
│
├── splits/                                    # Train-test splits
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── X_train_scaled.csv
│   └── X_test_scaled.csv
│
├── results/                                   # Evaluation outputs (tables only)
│   ├── model_comparison.csv                   # Metrics comparison table
│   └── best_models_per_metric.csv             # Best model for each metric
│
├── plots/                                   # All visualizations from the notebook
│   ├── EDA/                             
│   │   ├── target_distribution.png
│   │   ├── demographic_analysis.png
│   │   ├── heatmap_cp.png
│   │   ├── heatmap_ecg.png
│   │   ├── heatmap_slope.png
│   │   └── boxplot.png
│   ├── evaluation/             # Feature selection visualizations
│   │   └── gnb.png
│   │   ├── lr.png
│   │   ├── rf.png
│   │   └── svm.png
│   └── feature_importance.png                      # Model evaluation visualizations      
│
├── notebooks/
│   └── heart_disease.ipynb                     # Complete analysis pipeline
│
├── report/                                      # Final report
│   └── Cardiovascular_Disease_Prediction_Report.pdf
│
├── .gitignore                                   # Git ignore file
└── README.md                                    # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/sofianatale/HeartDisease_ML.git
cd HeartDisease_ML
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Requirements
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
joblib>=1.1.0
```

## Usage

### Quick Start: Make Predictions

```python
# Load trained model and scaler
import joblib
import pandas as pd

model = joblib.load('models/random_forest.joblib')
scaler = joblib.load('models/scaler.joblib')

# Prepare patient data (using top 10 features)
patient_data = pd.DataFrame([{
    'chol': 240.0,           # Cholesterol (mg/dL)
    'thalach': 150.0,        # Max heart rate
    'age': 55.0,             # Age (years)
    'oldpeak': 1.5,          # ST depression
    'exang': 0.0,            # Exercise angina (0 = No)
    'cp_atypical angina': 0.0,
    'trestbps': 130.0,        # Resting BP
    'thal': 3.0,              # Thalassemia type
    'sex_male': 1.0,          # 1 = Male
    'cp_non-anginal pain': 1.0
}])

# Scale features
scaled_data = scaler.transform(patient_data)

# Predict
prediction = model.predict(scaled_data)[0]
probability = model.predict_proba(scaled_data)[0][1]

print(f"Prediction: {'Heart Disease' if prediction == 1 else 'Healthy'}")
print(f"Probability of disease: {probability:.2%}")
```

### Run Full Pipeline

To reproduce the entire analysis:

```bash
jupyter notebook notebooks/heart_disease.ipynb
```

Or execute as Python script:
```bash
python -c "import notebook; notebook.run_notebook('notebooks/heart_disease.ipynb')"
```

## Model Comparison

### Final Test Set Performance

| Model | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | MCC |
|-------|----------|-------------|-------------|-----------|----------|---------|-----|
| **Random Forest** | **0.8626** | **0.94** | 0.7683 | 0.8319 | **0.8826** | **0.9107** | **0.7264** |
| Logistic Regression | 0.8516 | 0.89 | 0.8049 | 0.8476 | 0.8683 | 0.9106 | 0.6998 |
| Gaussian Naive Bayes | 0.8516 | 0.88 | **0.8171** | **0.8544** | 0.8670 | 0.9104 | 0.6998 |
| SVM | 0.8352 | 0.92 | 0.7317 | 0.8070 | 0.8598 | 0.8874 | 0.6703 |

### Best Model by Metric

| Metric | Best Model | Value |
|--------|------------|-------|
| **Accuracy** | Random Forest | 0.8626 |
| **Sensitivity** | Random Forest | 0.9400 |
| **Specificity** | Gaussian Naive Bayes | 0.8171 |
| **Precision** | Gaussian Naive Bayes | 0.8544 |
| **F1-Score** | Random Forest | 0.8826 |
| **ROC-AUC** | Random Forest | 0.9107 |
| **MCC** | Random Forest | 0.7264 |

### Key Observations

- **Random Forest** achieves the best overall performance with highest accuracy, sensitivity, and F1-score
- All models show strong discriminative ability with ROC-AUC > 0.88
- **Sensitivity-Specificity Trade-off**: SVM prioritizes sensitivity (0.92) at the cost of specificity
- Simpler models (Logistic Regression, Naive Bayes) perform remarkably well
- **Clinical perspective**: Random Forest minimizes false negatives (missed diagnoses) while maintaining good precision

## Clinical Implications

### Model Selection Rationale

For medical screening applications, **sensitivity (recall)** is often prioritized over specificity, as missing a disease diagnosis has more severe consequences than false alarms. Based on this criterion:

1. **Random Forest** (sensitivity = 0.94) → Best choice for screening
2. **SVM** (sensitivity = 0.92) → Acceptable alternative with higher false positive rate
3. **Logistic Regression** (sensitivity = 0.89) → Balanced option
4. **Naive Bayes** (sensitivity = 0.88) → Good specificity but lower sensitivity

### Risk Factor Confirmation

The feature importance analysis confirms established medical knowledge:
- Cholesterol is the dominant predictor
- Exercise capacity (thalach) strongly indicates cardiac function
- Age remains a fundamental risk factor
- ST depression validates as key diagnostic indicator

## Future Work

1. **External Validation**: Test on independent clinical datasets
2. **Multi-class Prediction**: Extend to severity levels (0-4) for risk stratification
3. **Deep Learning**: Compare with neural networks for potential improvements
4. **Explainable AI**: Implement SHAP/LIME for individual prediction explanations
5. **Clinical Deployment**: Develop web interface or API for healthcare integration
6. **Feature Engineering**: Create composite clinical indicators (e.g., rate-pressure product)
7. **Longitudinal Analysis**: Incorporate time-series data for progressive risk assessment

## References

### Dataset & Problem Domain
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*.

### Machine Learning Models
- **Logistic Regression**: Hosmer, D.W., Lemeshow, S., Sturdivant, R.X. (2013). *Applied Logistic Regression*.
- **Random Forest**: Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- **SVM**: Cortes, C., Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
- **Naive Bayes**: Zhang, H. (2004). The optimality of naive Bayes. *AAAI Conference*.

### Evaluation Metrics
- **Matthews Correlation Coefficient**: Matthews, B.W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta*.
- **ROC Analysis**: Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.
- **Medical Diagnostic Metrics**: Trevethan, R. (2017). Sensitivity, specificity, and predictive values. *Journal of Clinical and Diagnostic Research*.

### Feature Selection
- **RFECV**: Guyon, I., et al. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1), 389-422.
- **Feature Importance**: Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16(3), 199-231.

### Clinical Background
- World Health Organization. (2021). Cardiovascular diseases (CVDs). [Fact Sheet]
- Lloyd-Jones, D.M., et al. (2006). Prediction of lifetime risk for cardiovascular disease. *Circulation*, 113(6), 791-798.

## Author

**Sofia Natale**

- **Affiliation**: AML-BASIC 2025 – University of Bologna
- **Email**: sofia.natale@studio.unibo.it
- **GitHub**: sofianatale

---

*For questions, collaborations, or feedback, please open an issue on GitHub or contact the author directly.*
