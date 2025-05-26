# Churn Prediction Inference API

This repository contains a FastAPI-based inference pipeline for the Churn Prediction model. The API allows you to make predictions about customer churn and identify the key factors contributing to churn.

## Features

- **Real-time Prediction**: Make predictions for individual customers or in batch mode
- **Churn Reasons**: Get detailed insights into the factors contributing to churn
- **Model Information**: View details about the currently loaded model
- **Swagger Documentation**: Interactive API documentation
- **Docker Support**: Easy deployment with Docker

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py            # Model loading and inference
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── input.py            # Input data schemas
│   │   └── output.py           # Output data schemas
│   └── utils/
│       ├── __init__.py
│       └── model_trainer.py    # Model training utilities
├── models/                     # Directory for saved models
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── train_model.py              # Script to train the model
└── README_API.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Pandas, NumPy, scikit-learn
- SHAP (for feature importance)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Training the Model

Before using the API, you need to train a model:

```bash
python train_model.py --data-path path/to/your/data.csv --model-type RandomForest
```

Options:
- `--data-path`: Path to your training data CSV
- `--output-path`: Where to save the model (default: models/churn_model.joblib)
- `--model-type`: Type of model to train (LogisticRegression, RandomForest, GradientBoosting)
- `--target-col`: Name of the target column (default: lost_program)

### Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### Using Docker

Build and run with Docker:

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api
```

## API Endpoints

### Health Check
- `GET /`: Check if the API is running

### Prediction
- `POST /predict`: Predict churn for a single customer
- `POST /predict/batch`: Predict churn for multiple customers

### Model Information
- `GET /model/info`: Get information about the currently loaded model

## Example Usage

### Single Prediction

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Customer data
data = {
    "siteCenter": "Center A",
    "customerSize": "Medium",
    "days_rem_to_end_contract": 120,
    "total_sessions": 45,
    "avg_session_duration": 25.5
}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

print(json.dumps(result, indent=2))
```

### Batch Prediction

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict/batch"

# Batch of customers
data = {
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

# Make batch prediction
response = requests.post(url, json=data)
result = response.json()

print(json.dumps(result, indent=2))
```

## Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Customization

### Input Schema

Update the `ChurnPredictionInput` class in `app/schemas/input.py` to match your model's features.

### Model Loading

By default, the API looks for a model at `models/churn_model.joblib`. You can change this by setting the `MODEL_PATH` environment variable.

## Future Improvements

- Add authentication
- Implement model versioning
- Add monitoring and logging
- Create a CI/CD pipeline for model updates
- Add A/B testing capabilities

