# IBM Telco Customer Churn Prediction

## Overview

This project explores machine learning approaches to predict customer churn using the IBM Telco Customer Churn dataset. The primary objective is to compare classification models and evaluate strategies for addressing class imbalance, prioritizing **Recall** as the main performance metric.

In a customer retention context, high Recall is important because it helps identify customers who are likely to leave, reducing the number of potential churn cases missed by the model.

## Dataset

The dataset contains 7,043 customer records and 21 columns, including demographic information, subscribed services, contract details, billing information, and the target variable, `Churn`.

- **Target:** `Churn` (`Yes` or `No`), encoded as 1 or 0.
- **Initial class distribution:** 73.46% non-churn customers and 26.54% churn customers.
- **Missing values:** 11 records have missing `TotalCharges` values after numeric conversion.
- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The 11 records with missing `TotalCharges` are removed before modeling, leaving 7,032 records.

## Methodology

### 1. Data Exploration and Preprocessing

Exploratory data analysis examines numerical correlations and churn rates across selected categorical variables to identify potential relationships between customer characteristics and churn.

The preprocessing pipeline applies the following transformations:

- **Numerical features:** missing-value imputation using the median, followed by standardization.
- **Categorical features:** missing-value imputation using the most frequent value, followed by one-hot encoding.
- **Train-test split:** 80% training data and 20% test data, using stratification to preserve the target distribution.
- **Reproducibility:** fixed random seeds are used where applicable.

Using pipelines ensures that preprocessing transformations are fitted on the training data and then applied to the test data, reducing data leakage during preprocessing.

### 2. Class Imbalance Strategies

Three approaches are initially compared using Logistic Regression.

| Approach | Description |
|---|---|
| Baseline | Logistic Regression without explicit class balancing. |
| Class Weights | Logistic Regression with `class_weight="balanced"`, increasing the relative penalty associated with misclassifying the minority class. |
| SMOTENC | Synthetic minority oversampling for mixed numerical and categorical data, applied to the training data only. |

**Why retain the class-weighted approach?**

Class weighting increases Recall from 57.2% to 75.4% compared with the baseline, making it a strong reference for subsequent experiments. Unlike oversampling, it does not create synthetic observations or alter the training dataset's class distribution.

The class-weighted configuration is therefore retained as a reference strategy, not assumed to be the final best-performing solution.

### 3. Additional Models

Two additional classifiers are evaluated:

- **Random Forest:** an ensemble of 300 decision trees, using balanced class weights to address class imbalance.
- **XGBoost:** a gradient-boosted tree model using 300 estimators, a learning rate of 0.05, maximum depth of 4, and `scale_pos_weight` calculated from the training class distribution.

These models extend the comparison beyond Logistic Regression, allowing the evaluation of different approaches to learning patterns in customer data.

### 4. Evaluation Metrics

The models are evaluated on the same held-out test set using:

- **Recall:** primary metric, measuring the proportion of actual churn cases correctly identified.
- **Precision:** measures how many predicted churn cases are actual churn cases.
- **F1-score:** balances Precision and Recall.
- **Accuracy:** measures overall classification correctness.
- **ROC-AUC:** evaluates the ability to distinguish between churn and non-churn customers across classification thresholds.

Recall is prioritized because missing a customer who will churn may represent a lost opportunity for retention. However, Precision is also relevant because retention resources are limited and false positives can lead to unnecessary interventions.

## Results

The following results are reported by the notebook, rounded to three decimal places.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression — Baseline | 0.804 | 0.648 | 0.572 | 0.608 | 0.836 |
| Logistic Regression — Class Weight | 0.744 | 0.513 | 0.754 | 0.610 | 0.835 |
| Logistic Regression — SMOTENC | 0.764 | 0.543 | 0.709 | 0.615 | 0.830 |
| Random Forest — Class Weight | 0.790 | 0.628 | 0.511 | 0.563 | 0.823 |
| XGBoost — Class Weight | 0.735 | 0.501 | **0.783** | 0.611 | 0.831 |

### Key Findings

- **XGBoost achieves the highest Recall (78.3%)**, identifying the largest proportion of actual churn cases among the evaluated models.
- **Class-weighted Logistic Regression achieves 75.4% Recall**, substantially improving on the baseline while retaining a relatively simple and interpretable model.
- **SMOTENC reaches the highest F1-score (0.615)** among the evaluated models, illustrating that the model with the highest Recall does not necessarily provide the best balance between Precision and Recall.
- The baseline achieves the highest Accuracy (80.4%), but its lower Recall demonstrates why Accuracy alone is insufficient for this imbalanced classification problem.

No single model is designated as the final winner. Model selection requires considering Recall alongside Precision, overall performance, interpretability, and the practical costs of false positives and false negatives.

## Feature Importance

Feature importance is examined for Random Forest and XGBoost to understand which variables contribute most to their predictions.

The notebook highlights contract type, internet service, online security, technical support, and customer tenure among the relevant features. These rankings describe model-specific predictive importance and should not be interpreted as evidence of causal relationships.

## Known Limitations

- **Model selection:** Results are based on a single stratified train-test split. Cross-validation, threshold optimization, and additional validation would help establish the robustness of the comparisons.
- **Operational trade-offs:** A higher Recall may come at the expense of Precision. The acceptable balance depends on the business cost of missed churn cases and unnecessary retention interventions.

The reported metrics should therefore be interpreted as exploratory results from the current notebook implementation, particularly until the identifier and confusion-matrix issues are corrected.

## Technologies

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- imbalanced-learn
- XGBoost
- Jupyter Notebook

## How to Run

1. Clone this repository.
2. Install the required dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
   ```

3. Place `data/IBM-Churn.csv` in the location expected by the notebook.
4. Open and execute `IBM_Churn.ipynb` in Jupyter Notebook or JupyterLab.

## Project Goal

This project demonstrates how class imbalance influences churn prediction and why evaluation metrics must reflect the intended business objective. By prioritizing Recall and comparing multiple modeling strategies, it provides a foundation for developing customer retention models that identify a larger proportion of customers at risk of leaving.
