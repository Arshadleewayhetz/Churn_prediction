"""
Cox proportional hazards model class for loading and inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

from app.schemas.output import ChurnPredictionOutput, ChurnReason

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoxChurnModel:
    """
    Cox proportional hazards model class that handles:
    - Model loading
    - Data preprocessing
    - Prediction
    - Feature importance calculation
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the Cox churn model.
        
        Args:
            model_path: Path to the saved model file. If None, will use a default path.
        """
        self.model_path = model_path or os.environ.get("COX_MODEL_PATH", "models/cox_churn_model.joblib")
        self.model = None
        self.preprocessor = None
        self.feature_selector = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.duration_col = None
        self.event_col = None
        self.baseline_survival = None
        self.feature_importance = {}
        self.metrics = {}
        self.threshold = 0.5
        self.calibration_model = None
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained Cox model."""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading Cox model from {self.model_path}")
                model_data = joblib.load(self.model_path)
                
                # Extract components from the saved model data
                self.model = model_data.get("model")
                self.preprocessor = model_data.get("preprocessor")
                self.feature_selector = model_data.get("feature_selector")
                self.numerical_cols = model_data.get("numerical_cols", [])
                self.categorical_cols = model_data.get("categorical_cols", [])
                self.duration_col = model_data.get("duration_col")
                self.event_col = model_data.get("event_col")
                self.baseline_survival = model_data.get("baseline_survival")
                self.feature_importance = model_data.get("feature_importance", {})
                self.metrics = model_data.get("metrics", {})
                self.threshold = model_data.get("threshold", 0.5)
                self.calibration_model = model_data.get("calibration_model")
                
                logger.info(f"Cox model loaded successfully")
            else:
                logger.warning(f"Cox model file not found at {self.model_path}. Using dummy model.")
                self._create_dummy_model()
        except Exception as e:
            logger.error(f"Error loading Cox model: {str(e)}")
            logger.info("Using dummy Cox model instead.")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """
        Create a dummy Cox model for testing when no trained model is available.
        """
        from lifelines import CoxPHFitter
        
        logger.info("Creating dummy Cox model for testing")
        
        # Define basic feature names
        self.numerical_cols = ["days_rem_to_end_contract", "total_sessions", "avg_session_duration"]
        self.categorical_cols = ["siteCenter", "customerSize"]
        
        # Create a simple model
        self.model = CoxPHFitter()
        
        # Set dummy feature importance
        self.feature_importance = {
            "days_rem_to_end_contract": 0.35,
            "total_sessions": 0.25,
            "avg_session_duration": 0.20,
            "siteCenter": 0.12,
            "customerSize": 0.08
        }
        
        self.duration_col = "days_rem_to_end_contract"
        self.event_col = "lost_program"
        self.threshold = 0.5
        self.metrics = {"test_c_index": 0.75, "test_auc": 0.80}
    
    def _preprocess_data(self, input_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess input data using the trained preprocessor.
        
        Args:
            input_df: Input DataFrame with raw features
            
        Returns:
            Preprocessed and feature-selected data as numpy arrays
        """
        # Ensure all expected columns are present
        for col in self.numerical_cols + self.categorical_cols:
            if col not in input_df.columns:
                input_df[col] = None
        
        # If we have a preprocessor, use it
        if self.preprocessor:
            X_processed = self.preprocessor.transform(input_df)
            
            # Apply feature selection if available
            if self.feature_selector:
                X_selected = self.feature_selector.transform(X_processed)
                return X_processed, X_selected
            
            return X_processed, X_processed
        
        # Otherwise, return the raw data (not recommended for production)
        return input_df.values, input_df.values
    
    def _predict_churn_probability(self, X_selected, time_horizon=None):
        """
        Predict churn probability using the Cox model.
        
        Args:
            X_selected: Feature-selected data
            time_horizon: Time horizon for prediction
            
        Returns:
            Churn probability
        """
        if self.model is None:
            return np.array([0.5])
        
        try:
            # Create DataFrame for Cox model
            df = pd.DataFrame(
                X_selected, 
                columns=[f"feature_{i}" for i in range(X_selected.shape[1])]
            )
            
            # Get partial hazard
            partial_hazard = self.model.predict_partial_hazard(df)
            
            # Determine time horizon
            if time_horizon is None and self.baseline_survival is not None:
                time_horizon = self.baseline_survival.index.median()
            elif time_horizon is None:
                time_horizon = 180  # Default 6 months
            
            # Get baseline survival at time horizon
            if self.baseline_survival is not None:
                baseline_survival_at_horizon = self.baseline_survival[
                    self.baseline_survival.index <= time_horizon
                ].iloc[-1] if not self.baseline_survival.empty else 0.5
            else:
                baseline_survival_at_horizon = 0.5
            
            # Calculate survival probability
            survival_prob = np.power(baseline_survival_at_horizon, partial_hazard)
            
            # Convert to churn probability
            churn_prob = 1 - survival_prob
            
            # Apply calibration if available
            if self.calibration_model is not None:
                try:
                    churn_prob = self.calibration_model.predict(churn_prob)
                except:
                    pass
            
            return churn_prob
        
        except Exception as e:
            logger.error(f"Error predicting churn probability: {str(e)}")
            return np.array([0.5])
    
    def _get_feature_importance(self, X_processed, X_selected):
        """
        Get feature importance for the given input.
        
        Args:
            X_processed: Preprocessed input data
            X_selected: Feature-selected input data
            
        Returns:
            List of (feature_name, importance) tuples
        """
        # If we have a real model with coefficients
        if self.model is not None and hasattr(self.model, "params_"):
            try:
                # Get feature names
                feature_names = []
                feature_names.extend(self.numerical_cols)
                
                # Add one-hot encoded feature names
                if self.categorical_cols and self.preprocessor:
                    try:
                        ohe = self.preprocessor.transformers_[1][1].named_steps['onehot']
                        cat_feature_names = ohe.get_feature_names_out(self.categorical_cols)
                        feature_names.extend(cat_feature_names)
                    except:
                        pass
                
                # Get selected feature names if feature selector is available
                if self.feature_selector:
                    try:
                        selected_indices = self.feature_selector.get_support(indices=True)
                        selected_feature_names = [feature_names[i] for i in selected_indices]
                    except:
                        selected_feature_names = [f"feature_{i}" for i in range(X_selected.shape[1])]
                else:
                    selected_feature_names = feature_names
                
                # Get model coefficients
                coeffs = self.model.params_
                
                # Create importance dictionary
                importance_dict = {}
                for i, feature in enumerate(selected_feature_names):
                    if i < len(coeffs):
                        importance_dict[feature] = abs(coeffs.iloc[i])
                
                # Sort by importance
                sorted_importance = sorted(
                    importance_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                return sorted_importance
            
            except Exception as e:
                logger.warning(f"Error calculating feature importance from model: {str(e)}")
        
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
            X_processed, X_selected = self._preprocess_data(input_df)
            
            # Predict churn probability
            churn_probability = float(self._predict_churn_probability(X_selected)[0])
            
            # Get binary prediction
            churn_prediction = churn_probability >= self.threshold
            
            # Get feature importance
            feature_importance = self._get_feature_importance(X_processed, X_selected)
            
            # Calculate total importance
            total_importance = sum(imp for _, imp in feature_importance)
            
            # Convert to percentage and create ChurnReason objects
            churn_reasons = []
            for feature, importance in feature_importance[:5]:  # Top 5 reasons
                if total_importance > 0:
                    impact_percentage = round((importance / total_importance) * 100, 2)
                else:
                    impact_percentage = 20.0  # Default even distribution for 5 features
                
                churn_reasons.append(
                    ChurnReason(
                        feature_name=feature,
                        impact_percentage=impact_percentage
                    )
                )
            
            # Create output
            return ChurnPredictionOutput(
                customer_id=customer_id,
                churn_probability=churn_probability,
                churn_prediction=bool(churn_prediction),
                churn_reasons=churn_reasons,
                prediction_timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            )
        
        except Exception as e:
            logger.error(f"Error during Cox model prediction: {str(e)}")
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
_cox_model_instance = None

def get_cox_model() -> CoxChurnModel:
    """
    Get or create the CoxChurnModel instance.
    This function is used as a dependency in FastAPI endpoints.
    """
    global _cox_model_instance
    if _cox_model_instance is None:
        _cox_model_instance = CoxChurnModel()
    return _cox_model_instance

