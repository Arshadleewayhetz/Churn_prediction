"""
Output schemas for the Churn Prediction API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChurnReason(BaseModel):
    """Schema for a single churn reason with its impact percentage."""
    feature_name: str = Field(..., description="Name of the feature contributing to churn")
    impact_percentage: float = Field(..., description="Percentage impact on churn probability")

class ChurnPredictionOutput(BaseModel):
    """Output schema for single customer churn prediction."""
    customer_id: Optional[str] = Field(None, description="Customer identifier if provided in input")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: bool = Field(..., description="Binary churn prediction (True/False)")
    churn_reasons: List[ChurnReason] = Field(..., description="Top reasons for churn with impact percentages")
    prediction_timestamp: str = Field(..., description="Timestamp when prediction was made")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": None,
                "churn_probability": 0.78,
                "churn_prediction": True,
                "churn_reasons": [
                    {"feature_name": "days_rem_to_end_contract", "impact_percentage": 35.2},
                    {"feature_name": "avg_session_duration", "impact_percentage": 22.8},
                    {"feature_name": "total_sessions", "impact_percentage": 18.5}
                ],
                "prediction_timestamp": "2025-05-26T12:34:56"
            }
        }

class ChurnPredictionBatchOutput(BaseModel):
    """Output schema for batch churn prediction."""
    predictions: List[ChurnPredictionOutput] = Field(..., description="List of individual predictions")
    total_customers: int = Field(..., description="Total number of customers in the batch")
    churn_count: int = Field(..., description="Number of customers predicted to churn")
    churn_rate: float = Field(..., description="Percentage of customers predicted to churn")
    top_reasons: List[ChurnReason] = Field(..., description="Top reasons for churn across the batch")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": None,
                        "churn_probability": 0.78,
                        "churn_prediction": True,
                        "churn_reasons": [
                            {"feature_name": "days_rem_to_end_contract", "impact_percentage": 35.2},
                            {"feature_name": "avg_session_duration", "impact_percentage": 22.8},
                            {"feature_name": "total_sessions", "impact_percentage": 18.5}
                        ],
                        "prediction_timestamp": "2025-05-26T12:34:56"
                    },
                    {
                        "customer_id": None,
                        "churn_probability": 0.23,
                        "churn_prediction": False,
                        "churn_reasons": [
                            {"feature_name": "days_rem_to_end_contract", "impact_percentage": 40.1},
                            {"feature_name": "total_sessions", "impact_percentage": 25.3},
                            {"feature_name": "avg_session_duration", "impact_percentage": 15.7}
                        ],
                        "prediction_timestamp": "2025-05-26T12:34:57"
                    }
                ],
                "total_customers": 2,
                "churn_count": 1,
                "churn_rate": 0.5,
                "top_reasons": [
                    {"feature_name": "days_rem_to_end_contract", "impact_percentage": 37.65},
                    {"feature_name": "total_sessions", "impact_percentage": 21.9},
                    {"feature_name": "avg_session_duration", "impact_percentage": 19.25}
                ]
            }
        }

