#!/usr/bin/env python
# coding: utf-8

"""
Robust Cox Proportional Hazards Model for Churn Prediction
---------------------------------------------------------
This script implements a comprehensive Cox proportional hazards model for churn prediction
with robust evaluation on test datasets and accuracy improvement techniques.

Author: Codegen
Date: 2025-05-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.calibration import survival_probability_calibration
from lifelines.statistics import proportional_hazards_test
import joblib
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RobustCoxChurnModel:
    """
    A robust Cox proportional hazards model for churn prediction with:
    - Advanced preprocessing for highly skewed data
    - Feature selection optimized for survival analysis
    - Proper evaluation on test datasets
    - Accuracy improvement techniques
    - Calibration of survival probabilities
    """
    
    def __init__(self, penalizer=0.1, l1_ratio=0.0, robust=True):
        """
        Initialize the Cox model.
        
        Args:
            penalizer: Regularization strength (default: 0.1)
            l1_ratio: L1 vs L2 regularization mix (0=ridge, 1=lasso, default: 0.0)
            robust: Whether to use robust estimation (default: True)
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.robust = robust
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
    
    def _create_duration_and_event_cols(self, df, duration_col=None, event_col=None):
        """
        Create or identify duration and event columns for survival analysis.
        
        Args:
            df: Input DataFrame
            duration_col: Name of the duration column (if None, will try to identify)
            event_col: Name of the event column (if None, will use 'lost_program')
            
        Returns:
            DataFrame with duration and event columns
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Set event column
        if event_col is None:
            if 'lost_program' in df_copy.columns:
                self.event_col = 'lost_program'
            else:
                raise ValueError("Event column not specified and 'lost_program' not found in data")
        else:
            self.event_col = event_col
        
        # Set duration column
        if duration_col is not None:
            self.duration_col = duration_col
        else:
            # Try to identify or create a suitable duration column
            if 'days_rem_to_end_contract' in df_copy.columns:
                self.duration_col = 'days_rem_to_end_contract'
            elif 'endDate' in df_copy.columns and 'startDate' in df_copy.columns:
                # Calculate duration from start to end date
                df_copy['startDate'] = pd.to_datetime(df_copy['startDate'])
                df_copy['endDate'] = pd.to_datetime(df_copy['endDate'])
                df_copy['duration'] = (df_copy['endDate'] - df_copy['startDate']).dt.days
                self.duration_col = 'duration'
            elif 'tenure' in df_copy.columns:
                self.duration_col = 'tenure'
            else:
                raise ValueError("Could not identify a suitable duration column")
        
        # Ensure duration values are positive
        if df_copy[self.duration_col].min() <= 0:
            logger.warning(f"Found non-positive values in duration column. Adding 1 to all values.")
            df_copy[self.duration_col] = df_copy[self.duration_col] + 1
        
        return df_copy
    
    def _preprocess_data(self, df, missing_threshold=80, zero_threshold=80):
        """
        Preprocess the data by handling missing values, removing leakage, etc.
        
        Args:
            df: Input DataFrame
            missing_threshold: Threshold for removing columns with high missing values
            zero_threshold: Threshold for removing columns with high zero values
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Remove data leakage columns
        leakage_columns = ['churn_date', 'days_since_last_activity']
        id_columns = ['datalakeProgramId', 'customer_id']
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        
        # Keep some date columns that might be useful for time-based analysis
        keep_date_cols = ['startDate', 'endDate', 'last_activity_date']
        date_columns = [col for col in date_columns if col not in keep_date_cols]
        
        drop_columns = leakage_columns + id_columns + date_columns
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
        
        # Convert categorical columns
        categorical_columns = ['siteCenter', 'customerSize']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Remove columns with high missing values
        missing_values = df.isnull().mean() * 100
        high_missing_cols = missing_values[missing_values > missing_threshold].index.tolist()
        df = df.drop(columns=high_missing_cols)
        
        # Remove columns with high zero values
        zero_values = (df == 0).mean() * 100
        high_zero_cols = zero_values[zero_values > zero_threshold].index.tolist()
        df = df.drop(columns=high_zero_cols)
        
        logger.info(f"Removed {len(high_missing_cols)} columns with >{missing_threshold}% missing values")
        logger.info(f"Removed {len(high_zero_cols)} columns with >{zero_threshold}% zero values")
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def _build_preprocessing_pipeline(self, X_train):
        """
        Build a preprocessing pipeline for numerical and categorical features.
        
        Args:
            X_train: Training features
            
        Returns:
            preprocessor, numerical_cols, categorical_cols
        """
        # Identify numerical and categorical columns
        self.numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(include=['category']).columns.tolist()
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson', standardize=False)),  # Handle skewed data
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        
        return preprocessor
    
    def _select_features(self, X_train, y_train, X_test, y_test, max_features=None):
        """
        Select the most important features for the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
            max_features: Maximum number of features to select (if None, will use all)
            
        Returns:
            Feature selector
        """
        logger.info("Selecting important features...")
        
        # Get preprocessed data
        X_train_processed = self.preprocessor.transform(X_train)
        
        # Get feature names after preprocessing
        feature_names = []
        feature_names.extend(self.numerical_cols)
        
        # Add one-hot encoded feature names
        if self.categorical_cols:
            ohe = self.preprocessor.transformers_[1][1].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_feature_names)
        
        # Use a Random Forest to identify important features
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train_processed, y_train)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Create importance dictionary
        importance_dict = {}
        for i, feature in enumerate(feature_names):
            if i < len(importance):
                importance_dict[feature] = float(importance[i])
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Determine number of features to keep
        if max_features is None:
            max_features = len(sorted_features)
        else:
            max_features = min(max_features, len(sorted_features))
        
        # Keep only the top features
        top_features = [feature for feature, _ in sorted_features[:max_features]]
        
        # Create a feature selector
        feature_selector = SelectKBest(mutual_info_classif, k=max_features)
        feature_selector.fit(X_train_processed, y_train)
        
        # Store feature importance
        self.feature_importance = importance_dict
        
        logger.info(f"Selected {max_features} features")
        
        return feature_selector
    
    def fit(self, df, duration_col=None, event_col=None, test_size=0.2, random_state=42, max_features=None):
        """
        Fit the Cox proportional hazards model.
        
        Args:
            df: Input DataFrame
            duration_col: Name of the duration column (if None, will try to identify)
            event_col: Name of the event column (if None, will use 'lost_program')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            max_features: Maximum number of features to select (if None, will use all)
            
        Returns:
            self
        """
        logger.info("Fitting Cox proportional hazards model...")
        
        # Create duration and event columns
        df = self._create_duration_and_event_cols(df, duration_col, event_col)
        
        # Preprocess data
        df_processed = self._preprocess_data(df)
        
        # Split data
        X = df_processed.drop(columns=[self.duration_col, self.event_col])
        y = df_processed[self.event_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Build preprocessing pipeline
        self.preprocessor = self._build_preprocessing_pipeline(X_train)
        
        # Fit preprocessor
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Select features
        self.feature_selector = self._select_features(X_train, y_train, X_test, y_test, max_features)
        
        # Apply feature selection
        X_train_selected = self.feature_selector.transform(X_train_processed)
        X_test_selected = self.feature_selector.transform(X_test_processed)
        
        # Get feature names after preprocessing and selection
        feature_names = []
        feature_names.extend(self.numerical_cols)
        
        # Add one-hot encoded feature names
        if self.categorical_cols:
            ohe = self.preprocessor.transformers_[1][1].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_feature_names)
        
        # Keep only selected features
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        # Create DataFrame for Cox model
        train_df = pd.DataFrame(
            X_train_selected, 
            columns=[f"feature_{i}" for i in range(X_train_selected.shape[1])]
        )
        train_df[self.duration_col] = df_processed.loc[X_train.index, self.duration_col].values
        train_df[self.event_col] = df_processed.loc[X_train.index, self.event_col].values
        
        test_df = pd.DataFrame(
            X_test_selected, 
            columns=[f"feature_{i}" for i in range(X_test_selected.shape[1])]
        )
        test_df[self.duration_col] = df_processed.loc[X_test.index, self.duration_col].values
        test_df[self.event_col] = df_processed.loc[X_test.index, self.event_col].values
        
        # Fit Cox model
        self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio, robust=self.robust)
        self.model.fit(train_df, duration_col=self.duration_col, event_col=self.event_col)
        
        # Store baseline survival
        self.baseline_survival = self.model.baseline_survival_
        
        # Evaluate model
        self._evaluate_model(train_df, test_df)
        
        # Check proportional hazards assumption
        self._check_proportional_hazards(train_df)
        
        # Calibrate survival probabilities
        self._calibrate_survival_probabilities(train_df, test_df)
        
        logger.info("Model fitting complete")
        
        return self
    
    def _evaluate_model(self, train_df, test_df):
        """
        Evaluate the model on training and testing data.
        
        Args:
            train_df: Training data
            test_df: Testing data
        """
        logger.info("Evaluating model...")
        
        # Calculate concordance index (C-index)
        train_c_index = concordance_index(
            train_df[self.duration_col],
            -self.model.predict_partial_hazard(train_df),
            train_df[self.event_col]
        )
        
        test_c_index = concordance_index(
            test_df[self.duration_col],
            -self.model.predict_partial_hazard(test_df),
            test_df[self.event_col]
        )
        
        # Calculate AUC for binary prediction
        # For this, we need to convert survival predictions to binary churn predictions
        train_pred_prob = self._predict_churn_probability(train_df)
        test_pred_prob = self._predict_churn_probability(test_df)
        
        train_auc = roc_auc_score(train_df[self.event_col], train_pred_prob)
        test_auc = roc_auc_score(test_df[self.event_col], test_pred_prob)
        
        # Calculate accuracy
        train_pred = (train_pred_prob >= self.threshold).astype(int)
        test_pred = (test_pred_prob >= self.threshold).astype(int)
        
        train_accuracy = accuracy_score(train_df[self.event_col], train_pred)
        test_accuracy = accuracy_score(test_df[self.event_col], test_pred)
        
        # Calculate precision-recall AUC
        train_precision, train_recall, _ = precision_recall_curve(train_df[self.event_col], train_pred_prob)
        test_precision, test_recall, _ = precision_recall_curve(test_df[self.event_col], test_pred_prob)
        
        train_pr_auc = auc(train_recall, train_precision)
        test_pr_auc = auc(test_recall, test_precision)
        
        # Store metrics
        self.metrics = {
            'train_c_index': train_c_index,
            'test_c_index': test_c_index,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_pr_auc': train_pr_auc,
            'test_pr_auc': test_pr_auc
        }
        
        logger.info(f"Training C-index: {train_c_index:.4f}, Testing C-index: {test_c_index:.4f}")
        logger.info(f"Training AUC: {train_auc:.4f}, Testing AUC: {test_auc:.4f}")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}")
        logger.info(f"Training PR-AUC: {train_pr_auc:.4f}, Testing PR-AUC: {test_pr_auc:.4f}")
    
    def _check_proportional_hazards(self, df):
        """
        Check the proportional hazards assumption.
        
        Args:
            df: DataFrame with duration and event columns
        """
        try:
            # Run the proportional hazards test
            results = proportional_hazards_test(self.model, df, time_transform='rank')
            
            # Log results
            logger.info("Proportional Hazards Test Results:")
            logger.info(f"Global test p-value: {results.p_value}")
            
            # Check for violations
            violations = results.p_values[results.p_values < 0.05]
            if not violations.empty:
                logger.warning(f"Proportional hazards assumption violated for {len(violations)} features")
                for feature, p_value in violations.items():
                    logger.warning(f"Feature '{feature}' violates PH assumption (p={p_value:.4f})")
            else:
                logger.info("No violations of proportional hazards assumption detected")
        
        except Exception as e:
            logger.warning(f"Could not perform proportional hazards test: {str(e)}")
    
    def _calibrate_survival_probabilities(self, train_df, test_df):
        """
        Calibrate survival probabilities.
        
        Args:
            train_df: Training data
            test_df: Testing data
        """
        try:
            # Get predicted survival probabilities
            train_surv = self.model.predict_survival_function(train_df)
            
            # Get actual survival using Kaplan-Meier
            kmf = KaplanMeierFitter()
            kmf.fit(train_df[self.duration_col], event_observed=train_df[self.event_col])
            
            # Calibrate survival probabilities
            self.calibration_model = survival_probability_calibration(
                model_predicted=train_surv,
                actual_events=train_df[self.event_col],
                actual_times=train_df[self.duration_col]
            )
            
            logger.info("Survival probability calibration complete")
        
        except Exception as e:
            logger.warning(f"Could not calibrate survival probabilities: {str(e)}")
    
    def _predict_churn_probability(self, df, time_horizon=None):
        """
        Predict churn probability.
        
        Args:
            df: Input DataFrame
            time_horizon: Time horizon for prediction (if None, will use median duration)
            
        Returns:
            Churn probability
        """
        # Get partial hazard
        partial_hazard = self.model.predict_partial_hazard(df)
        
        # Determine time horizon
        if time_horizon is None:
            time_horizon = np.median(df[self.duration_col])
        
        # Get baseline survival at time horizon
        baseline_survival_at_horizon = self.baseline_survival[
            self.baseline_survival.index <= time_horizon
        ].iloc[-1] if not self.baseline_survival.empty else 0.5
        
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
    
    def predict(self, X, time_horizon=None):
        """
        Predict churn for new data.
        
        Args:
            X: Input features
            time_horizon: Time horizon for prediction (if None, will use median duration)
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Preprocess data
            X_processed = self.preprocessor.transform(X)
            
            # Apply feature selection
            X_selected = self.feature_selector.transform(X_processed)
            
            # Create DataFrame for Cox model
            df = pd.DataFrame(
                X_selected, 
                columns=[f"feature_{i}" for i in range(X_selected.shape[1])]
            )
            
            # Predict churn probability
            churn_prob = self._predict_churn_probability(df, time_horizon)
            
            # Predict binary churn
            churn_pred = (churn_prob >= self.threshold).astype(int)
            
            # Get feature importance
            importance = self._get_feature_importance(X)
            
            return {
                'churn_probability': churn_prob,
                'churn_prediction': churn_pred,
                'feature_importance': importance
            }
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'churn_probability': np.array([0.5]),
                'churn_prediction': np.array([0]),
                'feature_importance': [],
                'error': str(e)
            }
    
    def _get_feature_importance(self, X):
        """
        Get feature importance for the given input.
        
        Args:
            X: Input features
            
        Returns:
            List of (feature_name, importance) tuples
        """
        # Get feature names
        feature_names = []
        feature_names.extend(self.numerical_cols)
        
        # Add one-hot encoded feature names
        if self.categorical_cols:
            ohe = self.preprocessor.transformers_[1][1].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_feature_names)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
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
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create model data dictionary
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_selector': self.feature_selector,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'duration_col': self.duration_col,
            'event_col': self.event_col,
            'baseline_survival': self.baseline_survival,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'threshold': self.threshold,
            'calibration_model': self.calibration_model,
            'penalizer': self.penalizer,
            'l1_ratio': self.l1_ratio,
            'robust': self.robust
        }
        
        # Save the model
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded model
        """
        # Load model data
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            penalizer=model_data.get('penalizer', 0.1),
            l1_ratio=model_data.get('l1_ratio', 0.0),
            robust=model_data.get('robust', True)
        )
        
        # Restore model attributes
        instance.model = model_data.get('model')
        instance.preprocessor = model_data.get('preprocessor')
        instance.feature_selector = model_data.get('feature_selector')
        instance.numerical_cols = model_data.get('numerical_cols', [])
        instance.categorical_cols = model_data.get('categorical_cols', [])
        instance.duration_col = model_data.get('duration_col')
        instance.event_col = model_data.get('event_col')
        instance.baseline_survival = model_data.get('baseline_survival')
        instance.feature_importance = model_data.get('feature_importance', {})
        instance.metrics = model_data.get('metrics', {})
        instance.threshold = model_data.get('threshold', 0.5)
        instance.calibration_model = model_data.get('calibration_model')
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance


def train_and_evaluate_cox_model(data_path, output_path=None, test_size=0.2, random_state=42):
    """
    Train and evaluate a Cox proportional hazards model.
    
    Args:
        data_path: Path to the input data CSV
        output_path: Path to save the model (if None, will not save)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Trained model and evaluation metrics
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Replace infinity values with NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Create and fit model
    model = RobustCoxChurnModel(penalizer=0.1, l1_ratio=0.2, robust=True)
    model.fit(df, test_size=test_size, random_state=random_state)
    
    # Print evaluation metrics
    logger.info("Model Evaluation Metrics:")
    for metric, value in model.metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save model if output path is provided
    if output_path:
        model.save(output_path)
    
    return model, model.metrics


def main():
    """Main function to run the Cox model training and evaluation."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a Cox proportional hazards model for churn prediction")
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="../Churn/features_extracted.csv",
        help="Path to the input data CSV"
    )
    
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="models/cox_churn_model.joblib",
        help="Path to save the trained model"
    )
    
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for testing"
    )
    
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Train and evaluate model
    model, metrics = train_and_evaluate_cox_model(
        data_path=args.data_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    logger.info("Model training and evaluation complete")


if __name__ == "__main__":
    main()

