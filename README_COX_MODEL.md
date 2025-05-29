# Robust Cox Proportional Hazards Model for Churn Prediction

This repository contains a robust implementation of a Cox proportional hazards model for churn prediction. The model is designed to handle highly skewed data, evaluate performance on test datasets, and improve prediction accuracy.

## Features

- **Advanced Preprocessing**: Handles highly skewed data with >80% zero values in many features
- **Feature Selection**: Identifies the most important features for churn prediction
- **Robust Evaluation**: Comprehensive evaluation on test datasets with multiple metrics
- **Accuracy Improvement**: Techniques to improve model performance
- **Calibration**: Calibration of survival probabilities for better prediction
- **API Integration**: Seamless integration with the existing API infrastructure

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, lifelines, matplotlib, seaborn, joblib

### Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Training the Cox Model

To train the Cox proportional hazards model:

```bash
python train_cox_model.py --data-path path/to/your/data.csv --output-path models/cox_churn_model.joblib
```

Options:
- `--data-path`: Path to the input data CSV (default: "../Churn/features_extracted.csv")
- `--output-path`: Path to save the trained model (default: "models/cox_churn_model.joblib")
- `--test-size`: Proportion of data to use for testing (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--penalizer`: Regularization strength for Cox model (default: 0.1)
- `--l1-ratio`: L1 vs L2 regularization mix (0=ridge, 1=lasso, default: 0.2)

### Evaluating Models

To evaluate and compare different models:

```bash
python evaluate_models.py --data-path path/to/your/data.csv --output-dir models --results-dir results
```

Options:
- `--data-path`: Path to the input data CSV (default: "../Churn/features_extracted.csv")
- `--output-dir`: Directory to save models (default: "models")
- `--results-dir`: Directory to save evaluation results (default: "results")
- `--target-col`: Target column name (default: "lost_program")
- `--test-size`: Proportion of data to use for testing (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

## Using the API

The Cox model is integrated with the existing API infrastructure. You can use the following endpoints:

### Single Prediction

```
POST /predict/cox
```

Example request:
```json
{
  "siteCenter": "Center A",
  "customerSize": "Medium",
  "days_rem_to_end_contract": 120,
  "total_sessions": 45,
  "avg_session_duration": 25.5
}
```

### Batch Prediction

```
POST /predict/batch/cox
```

Example request:
```json
{
  "customers": [
    {
      "siteCenter": "Center A",
      "customerSize": "Medium",
      "days_rem_to_end_contract": 120,
      "total_sessions": 45,
      "avg_session_duration": 25.5
    },
    {
      "siteCenter": "Center B",
      "customerSize": "Large",
      "days_rem_to_end_contract": 60,
      "total_sessions": 20,
      "avg_session_duration": 15.2
    }
  ]
}
```

### Model Information

```
GET /model/cox/info
```

## How the Cox Model Works

The Cox proportional hazards model is a statistical model for survival analysis. In the context of churn prediction:

1. **Survival Time**: The time until a customer churns
2. **Event**: Whether a customer has churned (1) or not (0)
3. **Hazard Rate**: The instantaneous rate of churn at a given time
4. **Baseline Hazard**: The hazard rate when all covariates are zero
5. **Partial Hazard**: The effect of covariates on the hazard rate

The model estimates the effect of various features on the hazard rate, allowing us to:
- Identify the most important factors contributing to churn
- Predict the probability of churn over time
- Understand the time-dependent nature of churn

### Advantages of Cox Model for Churn Prediction

1. **Handles Time-to-Event Data**: Explicitly models the time until churn
2. **Incorporates Censoring**: Can handle customers who haven't churned yet
3. **Provides Interpretable Results**: Hazard ratios are easy to interpret
4. **Non-Parametric Baseline**: No assumptions about the shape of the baseline hazard
5. **Robust to Skewed Data**: Works well with highly skewed feature distributions

## Model Performance

The Cox model is evaluated using several metrics:

- **C-index**: Concordance index, similar to AUC but for survival data
- **AUC**: Area under the ROC curve for binary prediction
- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of true positives among positive predictions
- **Recall**: Proportion of true positives identified
- **F1 Score**: Harmonic mean of precision and recall
- **PR-AUC**: Area under the precision-recall curve

## Customizing the Model

You can customize the Cox model by adjusting the following parameters:

- **Penalizer**: Controls the strength of regularization
- **L1 Ratio**: Controls the mix of L1 (lasso) and L2 (ridge) regularization
- **Robust**: Whether to use robust estimation for standard errors
- **Threshold**: Threshold for binary prediction (default: 0.5)

## Troubleshooting

If you encounter issues:

1. **Model Not Found**: Ensure the model file exists at the specified path
2. **Missing Features**: Check that your input data contains all required features
3. **Performance Issues**: Try adjusting the penalizer and l1_ratio parameters
4. **API Errors**: Check the logs for detailed error messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

