"""
Input schemas for the Churn Prediction API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChurnPredictionInput(BaseModel):
    """
    Input schema for single customer churn prediction.
    
    Note: This schema should be updated to match the actual features
    used in your model. The fields below are examples based on the
    churn_statistical_model.py file.
    """
    # Example fields - update these based on your actual model features
    siteCenter: Optional[str] = Field(None, description="Site center")
    customerSize: Optional[str] = Field(None, description="Customer size category")
    
    # Numerical features
    days_rem_to_end_contract: Optional[float] = Field(None, description="Days remaining to end of contract")
    total_sessions: Optional[int] = Field(0, description="Total number of sessions")
    avg_session_duration: Optional[float] = Field(None, description="Average session duration in minutes")
    
    # Add more fields as needed based on your model's features
    
    class Config:
        schema_extra = {
            "example": {
                "siteCenter": "Center A",
                "customerSize": "Medium",
                "days_rem_to_end_contract": 120,
                "total_sessions": 45,
                "avg_session_duration": 25.5
            }
        }

class ChurnPredictionBatchInput(BaseModel):
    """Input schema for batch churn prediction."""
    customers: List[ChurnPredictionInput] = Field(..., description="List of customers to predict")
    
    class Config:
        schema_extra = {
            "example": {
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
        }

