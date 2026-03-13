# Cardiovascular Disease Prediction: A Machine Learning Approach
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557c?style=for-the-badge&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-3776AB?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine-Learning-ff6f00?style=for-the-badge)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare-00a1e0?style=for-the-badge)

![heart](https://github.com/sofianatale/HeartDisease_ML/blob/main/figures/Coronary-heart-disease.jpg)
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
- [Model Comparison](#model-comparison)
- [Clinical Implications](#clinical-implications)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)


## Project Overview

Cardiovascular diseases represent the leading cause of death worldwide, accounting for approximately **19.8 million deaths annually** according to the World Health Organization (WHO). Early identification of individuals at risk is essential for implementing preventive strategies and improving long-term patient outcomes.

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

## Pipeline Summary

![Visual_pipeline](https://github.com/sofianatale/HeartDisease_ML/blob/main/figures/visual_pipeline.png)

## Technical Stack

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
├── figures/                                     
│   └── Coronary-heart-disease.jpg
│
├── models/                                   # Trained models and artifacts
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── random_forest.joblib
│   ├── scaler.joblib                         # Fitted MinMaxScaler
│   └── feature_names.txt                     # Selected features
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
│   │   ├── gnb.png
│   │   ├── lr.png
│   │   ├── rf.png
│   │   └── svm.png
│   └── feature_importance.png                      # Model evaluation visualizations      
│
├── project/
│   ├── HeartDisease_Prediction_Report.pdf
│   └── heart_disease.ipynb                     # Complete analysis pipeline
│ 
├── results/                                   # Evaluation outputs (tables only)
│   ├── model_comparison.csv                   # Metrics comparison table
│   └── best_models_per_metric.csv             # Best model for each metric
│
├── splits/                                    # Train-test splits
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── X_train_scaled.csv
│   └── X_test_scaled.csv
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

3. **Requirements**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
joblib>=1.1.0
```

4. **Run Full Pipeline**

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

#### *Dataset & Problem Domain*
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Ali MM, Paul BK, Ahmed K, Bui FM, Quinn JMW, Moni MA. [Heart disease prediction using supervised machine learning algorithms: Performance analysis and comparison](https://pubmed.ncbi.nlm.nih.gov/34315030/). Comput Biol Med. 2021 Sep;136:104672. 

#### *Machine Learning Models*
- Nick TG, Campbell KM. [Logistic regression](https://pubmed.ncbi.nlm.nih.gov/18450055/). Methods Mol Biol. 2007;404:273-301. 
- Hu J, Szymczak S. [A review on longitudinal data analysis with random forest](https://pubmed.ncbi.nlm.nih.gov/36653905/). Brief Bioinform. 2023 Mar 19;24(2):bbad002.
- Lai Z, Chen X, Zhang J, Kong H, Wen J. [Maximal Margin Support Vector Machine for Feature Representation and Classification.](https://pubmed.ncbi.nlm.nih.gov/37018685/) IEEE Trans Cybern. 2023 Oct;53(10):6700-6713.
- Pavan Venkata, Vivek Pandya, [Data mining model and Gaussian Naive Bayes based fault diagnostic analysis of modern power system networks](https://www.sciencedirect.com/science/article/abs/pii/S2214785322013372)
Materials Today: Proceedings, Volume 62, Part 13, 2022, Pages 7156-7161, ISSN 2214-7853.


#### *Evaluation Metrics*
- *Chicco D, Tötsch N, Jurman G. [The Matthews correlation coefficient (MCC) is more reliable than balanced accuracy, bookmaker informedness, and markedness in two-class confusion matrix evaluation](https://pubmed.ncbi.nlm.nih.gov/33541410/). BioData Min. 2021 Feb 4;14(1):13. 
- Obuchowski NA, Bullen JA. [Receiver operating characteristic (ROC) curves: review of methods with applications in diagnostic medicine](https://pubmed.ncbi.nlm.nih.gov/29512515/). Phys Med Biol. 2018 Mar 29;63(7):07TR01.
- Hicks SA, Strümke I, Thambawita V, Hammou M, Riegler MA, Halvorsen P, Parasa S. [On evaluation metrics for medical applications of artificial intelligence](https://pmc.ncbi.nlm.nih.gov/articles/PMC8993826/). Sci Rep. 2022 Apr 8;12(1):5979.

#### *Feature Selection*
- [RFECV ](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
- [Feature Importance](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

#### *Clinical Background*
- [World Health Organization (2025). Cardiovascular diseases (CVDs)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
- Goldsborough E 3rd, Osuji N, Blaha MJ. [Assessment of Cardiovascular Disease Risk: A 2022 Update.](https://pubmed.ncbi.nlm.nih.gov/35963625/)Endocrinol Metab Clin North Am. 2022 Sep;51(3):483-509.


## Author

**Sofia Natale**

- **Affiliation**: AML-BASIC 2025 – University of Bologna
- **Email**: [sofia.natale@studio.unibo.it](sofia.natale@studio.unibo.it)
- **GitHub**: [sofianatale](https://github.com/sofianatale)

---

*For questions, collaborations, or feedback, please open an issue on GitHub or contact the author directly.*
