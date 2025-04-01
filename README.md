# Telco Customer Churn Prediction

This project predicts customer churn using a Random Forest classifier trained on the IBM Telco dataset. The full pipeline includes data preprocessing, model training with hyperparameter tuning, threshold optimization for F1/recall, and performance visualization.

## Overview

- **Goal:** Predict which customers are likely to churn.
- **Approach:** Decision Tree and Random Forest classifiers with GridSearchCV and custom threshold tuning.
- **Output:** Optimized model metrics and export-ready results for Tableau.

## Methods

- **Modeling:** Random Forest with GridSearchCV
- **Metrics:** AUC, Accuracy, Precision, Recall, F1
- **Threshold Optimization:** F1- and recall-based tuning
- **Visualizations:** ROC, Precision-Recall, Confusion Matrix, Feature Importance

## Results

- **Best Validation AUC:** ~0.86
- **Improved Recall via Threshold Tuning**
- Exported CSVs for Tableau analysis