"""
Churn Prediction API
-------------------
FastAPI application for churn prediction inference.

Author: Codegen
Date: 2025-05-26
"""

import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Any, Optional

from app.schemas.input import ChurnPredictionInput, ChurnPredictionBatchInput
from app.schemas.output import ChurnPredictionOutput, ChurnPredictionBatchOutput, ChurnReason
from app.models.model import ChurnModel, get_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn and identifying churn reasons",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "message": "Churn Prediction API is running"}

@app.post("/predict", response_model=ChurnPredictionOutput, tags=["Prediction"])
async def predict_churn(
    input_data: ChurnPredictionInput,
    model: ChurnModel = Depends(get_model)
):
    """
    Predict churn for a single customer.
    
    Returns:
        - Churn probability
        - Binary churn prediction
        - Top churn reasons with impact percentages
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction_result = model.predict(input_df)
        
        return prediction_result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=ChurnPredictionBatchOutput, tags=["Prediction"])
async def predict_churn_batch(
    input_data: ChurnPredictionBatchInput,
    model: ChurnModel = Depends(get_model)
):
    """
    Predict churn for multiple customers.
    
    Returns:
        - List of predictions with churn probabilities and reasons
        - Summary statistics
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([item.dict() for item in input_data.customers])
        
        # Make batch predictions
        predictions = []
        for i in range(len(input_df)):
            single_prediction = model.predict(input_df.iloc[[i]])
            predictions.append(single_prediction)
        
        # Calculate summary statistics
        churn_count = sum(1 for p in predictions if p.churn_prediction)
        churn_rate = churn_count / len(predictions) if predictions else 0
        
        # Aggregate top reasons
        all_reasons = []
        for p in predictions:
            all_reasons.extend(p.churn_reasons)
        
        # Count reason frequencies
        reason_counts = {}
        for reason in all_reasons:
            if reason.feature_name in reason_counts:
                reason_counts[reason.feature_name] += 1
            else:
                reason_counts[reason.feature_name] = 1
        
        # Sort by frequency
        top_reasons = [
            ChurnReason(
                feature_name=feature,
                impact_percentage=sum(r.impact_percentage for r in all_reasons if r.feature_name == feature) / reason_counts[feature]
            )
            for feature in sorted(reason_counts, key=reason_counts.get, reverse=True)[:5]
        ]
        
        return ChurnPredictionBatchOutput(
            predictions=predictions,
            total_customers=len(predictions),
            churn_count=churn_count,
            churn_rate=churn_rate,
            top_reasons=top_reasons
        )
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info", tags=["Model"])
async def model_info(model: ChurnModel = Depends(get_model)):
    """Get information about the currently loaded model."""
    return {
        "model_type": model.model_type,
        "features": model.feature_names,
        "model_metrics": model.model_metrics,
        "last_trained": model.last_trained,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

