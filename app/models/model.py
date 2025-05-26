"""
Churn prediction model class for loading and inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import shap

from app.schemas.output import ChurnPredictionOutput, ChurnReason

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnModel:
    """
    Churn prediction model class that handles:
    - Model loading
    - Data preprocessing
    - Prediction
    - Feature importance calculation
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the churn model.
        
        Args:
            model_path: Path to the saved model file. If None, will use a default path.
        """
        self.model_path = model_path or os.environ.get("MODEL_PATH", "models/churn_model.joblib")
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.numerical_cols = []
        self.categorical_cols = []
        self.model_type = ""
        self.feature_importance = {}
        self.model_metrics = {}
        self.last_trained = None
        self.threshold = 0.5  # Default threshold for binary prediction
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and preprocessor."""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                model_data = joblib.load(self.model_path)
                
                # Extract components from the saved model data
                self.model = model_data.get("model")
                self.preprocessor = model_data.get("preprocessor")
                self.feature_names = model_data.get("feature_names", [])
                self.numerical_cols = model_data.get("numerical_cols", [])
                self.categorical_cols = model_data.get("categorical_cols", [])
                self.model_type = model_data.get("model_type", "Unknown")
                self.feature_importance = model_data.get("feature_importance", {})
                self.model_metrics = model_data.get("model_metrics", {})
                self.last_trained = model_data.get("last_trained", "Unknown")
                self.threshold = model_data.get("threshold", 0.5)
                
                logger.info(f"Model loaded successfully. Type: {self.model_type}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Using dummy model.")
                self._create_dummy_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Using dummy model instead.")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """
        Create a dummy model for testing when no trained model is available.
        This is useful for development and testing the API without a real model.
        """
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Creating dummy model for testing")
        
        # Define basic feature names
        self.numerical_cols = ["days_rem_to_end_contract", "total_sessions", "avg_session_duration"]
        self.categorical_cols = ["siteCenter", "customerSize"]
        self.feature_names = self.numerical_cols + self.categorical_cols
        
        # Create a simple preprocessor
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )
        
        # Create a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Set dummy feature importance
        self.feature_importance = {
            "days_rem_to_end_contract": 0.35,
            "total_sessions": 0.25,
            "avg_session_duration": 0.20,
            "siteCenter": 0.12,
            "customerSize": 0.08
        }
        
        self.model_type = "RandomForestClassifier (Dummy)"
        self.model_metrics = {"accuracy": 0.8, "auc": 0.85}
        self.last_trained = datetime.now().strftime("%Y-%m-%d")
        self.threshold = 0.5
    
    def _preprocess_data(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data using the trained preprocessor.
        
        Args:
            input_df: Input DataFrame with raw features
            
        Returns:
            Preprocessed data as numpy array
        """
        # Ensure all expected columns are present
        for col in self.numerical_cols + self.categorical_cols:
            if col not in input_df.columns:
                input_df[col] = None
        
        # If we have a preprocessor, use it
        if self.preprocessor:
            return self.preprocessor.transform(input_df)
        
        # Otherwise, return the raw data (not recommended for production)
        return input_df.values
    
    def _get_feature_importance(self, input_data: np.ndarray) -> List[Tuple[str, float]]:
        """
        Calculate feature importance for the given input.
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            List of (feature_name, importance) tuples
        """
        # If we have a real model, try to use SHAP for feature importance
        if hasattr(self.model, "feature_importances_") or hasattr(self.model, "coef_"):
            try:
                # For tree-based models
                if hasattr(self.model, "feature_importances_"):
                    importances = self.model.feature_importances_
                    # Get feature names after preprocessing
                    feature_names = []
                    feature_names.extend(self.numerical_cols)
                    
                    # Add one-hot encoded feature names if we have a preprocessor
                    if self.preprocessor and self.categorical_cols:
                        ohe = self.preprocessor.transformers_[1][1].named_steps['onehot']
                        cat_feature_names = ohe.get_feature_names_out(self.categorical_cols)
                        feature_names.extend(cat_feature_names)
                    
                    # Create importance dictionary
                    importance_dict = {}
                    for i, feature in enumerate(feature_names):
                        if i < len(importances):
                            importance_dict[feature] = importances[i]
                    
                    # Sort by importance
                    sorted_importance = sorted(
                        importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    return sorted_importance
                
                # For linear models
                elif hasattr(self.model, "coef_"):
                    importances = np.abs(self.model.coef_[0])
                    # Get feature names after preprocessing
                    feature_names = []
                    feature_names.extend(self.numerical_cols)
                    
                    # Add one-hot encoded feature names if we have a preprocessor
                    if self.preprocessor and self.categorical_cols:
                        ohe = self.preprocessor.transformers_[1][1].named_steps['onehot']
                        cat_feature_names = ohe.get_feature_names_out(self.categorical_cols)
                        feature_names.extend(cat_feature_names)
                    
                    # Create importance dictionary
                    importance_dict = {}
                    for i, feature in enumerate(feature_names):
                        if i < len(importances):
                            importance_dict[feature] = importances[i]
                    
                    # Sort by importance
                    sorted_importance = sorted(
                        importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    return sorted_importance
            
            except Exception as e:
                logger.warning(f"Error calculating feature importance: {str(e)}")
        
        # Fall back to pre-calculated feature importance
        return sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
    
    def predict(self, input_df: pd.DataFrame) -> ChurnPredictionOutput:
        """
        Make a churn prediction for the given input data.
        
        Args:
            input_df: DataFrame with input features
            
        Returns:
            ChurnPredictionOutput with prediction results
        """
        try:
            # Extract customer_id if present
            customer_id = None
            if 'customer_id' in input_df.columns:
                customer_id = input_df['customer_id'].iloc[0]
                input_df = input_df.drop(columns=['customer_id'])
            
            # Preprocess the data
            preprocessed_data = self._preprocess_data(input_df)
            
            # Make prediction
            if self.model:
                # Get probability
                churn_probability = float(self.model.predict_proba(preprocessed_data)[0, 1])
                
                # Get binary prediction
                churn_prediction = churn_probability >= self.threshold
                
                # Get feature importance
                feature_importance = self._get_feature_importance(preprocessed_data)
                
                # Calculate total importance
                total_importance = sum(imp for _, imp in feature_importance)
                
                # Convert to percentage and create ChurnReason objects
                churn_reasons = [
                    ChurnReason(
                        feature_name=feature,
                        impact_percentage=round((importance / total_importance) * 100, 2)
                    )
                    for feature, importance in feature_importance[:5]  # Top 5 reasons
                ]
            else:
                # Dummy prediction if no model is available
                churn_probability = 0.5
                churn_prediction = False
                churn_reasons = [
                    ChurnReason(feature_name="No model available", impact_percentage=100.0)
                ]
            
            # Create output
            return ChurnPredictionOutput(
                customer_id=customer_id,
                churn_probability=churn_probability,
                churn_prediction=bool(churn_prediction),
                churn_reasons=churn_reasons,
                prediction_timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            )
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return a default prediction with error information
            return ChurnPredictionOutput(
                customer_id=None,
                churn_probability=0.0,
                churn_prediction=False,
                churn_reasons=[
                    ChurnReason(
                        feature_name=f"Error: {str(e)}", 
                        impact_percentage=100.0
                    )
                ],
                prediction_timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            )

# Singleton model instance
_model_instance = None

def get_model() -> ChurnModel:
    """
    Get or create the ChurnModel instance.
    This function is used as a dependency in FastAPI endpoints.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ChurnModel()
    return _model_instance

