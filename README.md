# Churn Prediction Statistical Model

This repository contains a comprehensive statistical model for identifying churn reasons and their impact percentages in a dataset with 89 features, where many features have more than 80% zero values.

## Project Overview

The goal of this project is to create a statistical model that can:
1. Identify the key features causing customer churn
2. Quantify the percentage impact of each feature on churn
3. Provide actionable insights for reducing churn

## Approach

### Data Preprocessing
- Handle missing values and zero values
- Remove features with >80% missing or zero values
- Remove data leakage columns
- Standardize numerical features
- Encode categorical features

### Statistical Modeling
- Multiple classification models (Logistic Regression, Random Forest, Gradient Boosting)
- Survival analysis using Cox Proportional Hazards model
- Feature importance analysis using multiple methods (model-based, permutation, SHAP)

### Churn Reason Identification
- Identify top features contributing to churn
- Calculate percentage impact of each feature
- Provide actionable insights

## Files in this Repository

- `churn_statistical_model.py`: Python script implementing the complete statistical model
- `churn_statistical_analysis.ipynb`: Jupyter notebook with detailed analysis and visualizations
- `README.md`: This file

## How to Use

1. Clone this repository
2. Ensure you have the required dependencies installed (pandas, numpy, scikit-learn, lifelines, shap)
3. Run the Jupyter notebook for interactive analysis or the Python script for batch processing

## Key Insights

The model identifies the most significant features contributing to churn and quantifies their impact. This information can be used to:

1. Develop targeted retention strategies
2. Improve operational metrics that are strongly associated with churn
3. Create early warning systems for at-risk customers
4. Optimize resource allocation for customer retention efforts

## Dependencies

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- lifelines
- shap

## Future Work

- Implement more advanced survival analysis techniques
- Develop a real-time churn prediction system
- Create interactive dashboards for monitoring key churn indicators
- Integrate with customer relationship management systems

