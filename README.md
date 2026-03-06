# Heart Disease Prediction using Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/downloads/release/python-3119/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.2-11557c.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-9c89cc.svg)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project presents a comprehensive analysis of the classic UCI Heart Disease dataset, applying a full end-to-end machine learning pipeline to predict the presence of heart disease in a patient. The primary goal is to build and evaluate several classification models, comparing their performance to identify the most effective predictor.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Data Preparation](#3-data-preparation)
  - [4. Feature Selection](#4-feature-selection)
  - [5. Model Training & Evaluation](#5-model-training--evaluation)
- [Key Findings](#key-findings)
- [Model Performance Comparison](#model-performance-comparison)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview
Cardiovascular diseases are a leading cause of death globally. Early and accurate detection can significantly improve patient outcomes. This project leverages machine learning to classify patients based on various clinical features. We explore, clean, and model data from four different sources (Cleveland, Hungary, Switzerland, and VA) to build a robust predictive system.

## Dataset
The dataset used is the [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) from the UCI Machine Learning Repository. It combines data from four independent sources:
*   Cleveland Clinic Foundation (cleveland.data)
*   Hungarian Institute of Cardiology, Budapest (hungarian.data)
*   University Hospital, Zurich, Switzerland (switzerland.data)
*   VA Medical Center, Long Beach, CA (va.data)

After cleaning and merging, the final dataset contains **908 patient records**. The target variable (`target`) indicates the presence of heart disease on a scale of 0 (no disease) to 4. For this analysis, the target is binarized into `Healthy` (0) and `Heart Disease` (1,2,3,4).

**Features:**
*   `age`: Age in years
*   `sex`: Male or Female
*   `cp`: Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic)
*   `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital)
*   `chol`: Serum cholesterol in mg/dl
*   `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
*   `restecg`: Resting electrocardiographic results (normal, ST-T wave abnormality, left ventricular hypertrophy)
*   `thalach`: Maximum heart rate achieved
*   `exang`: Exercise induced angina (1 = yes; 0 = no)
*   `oldpeak`: ST depression induced by exercise relative to rest
*   `slope`: The slope of the peak exercise ST segment (upsloping, flat, downsloping)
*   `ca`: Number of major vessels (0-3) colored by fluoroscopy
*   `thal`: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

## Methodology
The analysis follows a structured machine learning pipeline.

### 1. Data Loading & Preprocessing
*   Data from the four `.data` files is loaded, merged, and cleaned.
*   Missing values (originally marked as '?') are imputed using the **median** of their respective columns.
*   Categorical features (`cp`, `restecg`, `slope`, `sex`) are converted into meaningful text labels before being encoded.

### 2. Exploratory Data Analysis (EDA)
A thorough EDA was conducted to understand the data and extract insights:
*   **Target Distribution**: Analyzed both the binary (healthy/disease) and multi-level severity of the target.
*   **Demographics**: Visualized age density and gender distribution across the target groups.
*   **Categorical Features**: Created heatmaps to show the probability of heart disease for each category of `cp`, `restecg`, and `slope`, revealing strong predictors.
*   **Numerical Features**: Used boxplots to compare distributions and identify outliers for `thalach`, `oldpeak`, and `trestbps` between healthy and diseased patients.
*   **Outlier Detection**: Outliers in numerical features were identified using the **Z-score method** and removed to improve model robustness.

### 3. Data Preparation
*   **Feature Encoding**: Categorical variables were one-hot encoded using `pd.get_dummies()`.
*   **Train-Test Split**: The data was split into training (80%) and testing (20%) sets using **stratified splitting** to maintain the original class distribution.
*   **Feature Scaling**: Numerical features were normalized to a [0, 1] range using `MinMaxScaler` to ensure models like SVM and Logistic Regression perform optimally.

### 4. Feature Selection
To reduce dimensionality and focus on the most impactful features:
*   **Wrapper Method (RFECV)**: Recursive Feature Elimination with Cross-Validation was used with a Logistic Regression estimator.
*   **Embedded Method (Random Forest)**: Feature importance scores from a Random Forest classifier were used for ranking.
*   The top 10 features were selected for final model training based on the Random Forest importance ranking.

### 5. Model Training & Evaluation
Four different classifiers were trained and optimized using `GridSearchCV` with 5-fold cross-validation, optimizing for the `f1_macro` score:
*   **Logistic Regression**
*   **Gaussian Naive Bayes**
*   **Support Vector Machine (SVM)**
*   **Random Forest**

Each model was evaluated on the held-out test set using a variety of metrics:
*   Accuracy, Precision, Recall, F1-Score
*   Confusion Matrix
*   ROC Curve and AUC Score
*   Matthews Correlation Coefficient (MCC)

## Key Findings
*   **EDA Insights**: The analysis confirmed known medical knowledge. Patients with asymptomatic chest pain (`cp`), higher exercise-induced ST depression (`oldpeak`), and lower maximum heart rate (`thalach`) are significantly more likely to have heart disease.
*   **Best Model**: The **Random Forest** classifier consistently outperformed other models, achieving the highest accuracy, F1-score, and ROC-AUC.
*   **Feature Importance**: Key predictive features included `chol`, `thalach`, `age`, and `oldpeak`.

## Model Performance Comparison
Here is a comparison of the final optimized models on the test set.

| Model | Accuracy | Recall | Macro-F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.8626** | **0.94** | **0.8585** | **0.7264** |
| Logistic Regression | 0.8516 | 0.89 | 0.8492 | 0.6998 |
| Gaussian Naive Bayes | 0.8516 | 0.88 | 0.8496 | 0.6998 |
| SVM | 0.8352 | 0.92 | 0.8299 | 0.6703 |

*(Note: Performance metrics are based on the binary classification task of predicting the presence of heart disease.)*

## Conclusion
This project successfully demonstrates the application of a complete machine learning pipeline for heart disease prediction. Through rigorous analysis and model comparison, the **Random Forest** classifier was identified as the most effective model, achieving a balanced and high-performing prediction. The analysis also highlighted the most critical clinical features, aligning with established medical understanding. This work can serve as a strong foundation for further development of clinical decision support tools.

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[your-username]/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2.  **Download the dataset:**
    *   Download the four `.data` files (`processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`) from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease).
    *   Place them in the root directory of the project.

3.  **Run the Jupyter Notebook:**
    *   Ensure you have Jupyter installed (`pip install jupyter`).
    *   Launch the notebook:
        ```bash
        jupyter notebook heart_disease.ipynb
        ```
    *   Execute all cells to see the full analysis and results.

## Dependencies
*   Python 3.11+
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scipy
*   scikit-learn

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
