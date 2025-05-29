"""
Utility functions for training and saving the churn prediction model.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset and perform initial preprocessing.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Replace infinity values with NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Dataset shape: {df.shape}")
    
    return df

def preprocess_data(
    df: pd.DataFrame, 
    target_col: str = 'lost_program',
    missing_threshold: float = 80, 
    zero_threshold: float = 80
) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values, removing leakage, etc.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        missing_threshold: Threshold for removing columns with high missing values
        zero_threshold: Threshold for removing columns with high zero values
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing data...")
    
    # Remove data leakage columns
    leakage_columns = ['churn_date', 'days_since_last_activity']
    id_columns = ['datalakeProgramId']
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

def split_data(
    df: pd.DataFrame, 
    target_col: str = 'lost_program', 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(X_train: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a preprocessing pipeline for numerical and categorical features.
    
    Args:
        X_train: Training features
        
    Returns:
        preprocessor, numerical_cols, categorical_cols
    """
    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['category']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor, numerical_cols, categorical_cols

def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    preprocessor: ColumnTransformer,
    model_type: str = 'RandomForest'
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a model and evaluate its performance.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        preprocessor: Preprocessing pipeline
        model_type: Type of model to train ('LogisticRegression', 'RandomForest', or 'GradientBoosting')
        
    Returns:
        Trained pipeline and metrics dictionary
    """
    logger.info(f"Training {model_type} model...")
    
    # Select model based on type
    if model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:  # Default to RandomForest
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    
    # Create pipeline with preprocessing
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = pipeline.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_prob)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score']
    }
    
    logger.info(f"{model_type} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return pipeline, metrics

def calculate_feature_importance(
    pipeline: Pipeline, 
    numerical_cols: List[str], 
    categorical_cols: List[str]
) -> Dict[str, float]:
    """
    Calculate feature importance from the trained model.
    
    Args:
        pipeline: Trained pipeline
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    model = pipeline.named_steps['model']
    
    # Get feature names after preprocessing
    feature_names = []
    feature_names.extend(numerical_cols)
    
    # Add one-hot encoded feature names
    if categorical_cols:
        ohe = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names)
    
    # Get feature importance based on model type
    if isinstance(model, LogisticRegression):
        # For logistic regression, use coefficients
        importance = np.abs(model.coef_[0])
    else:
        # For tree-based models, use feature importance
        importance = model.feature_importances_
    
    # Create importance dictionary
    importance_dict = {}
    for i, feature in enumerate(feature_names):
        if i < len(importance):
            importance_dict[feature] = float(importance[i])
    
    return importance_dict

def save_model(
    pipeline: Pipeline, 
    preprocessor: ColumnTransformer,
    numerical_cols: List[str],
    categorical_cols: List[str],
    metrics: Dict[str, Any],
    feature_importance: Dict[str, float],
    model_type: str,
    output_path: str = 'models/churn_model.joblib'
) -> None:
    """
    Save the trained model and associated metadata.
    
    Args:
        pipeline: Trained pipeline
        preprocessor: Preprocessing pipeline
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        metrics: Dictionary of model performance metrics
        feature_importance: Dictionary of feature importance scores
        model_type: Type of model
        output_path: Path to save the model
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract the model from the pipeline
    model = pipeline.named_steps['model']
    
    # Get feature names
    feature_names = numerical_cols + categorical_cols
    
    # Create model data dictionary
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'model_type': model_type,
        'feature_importance': feature_importance,
        'model_metrics': metrics,
        'last_trained': datetime.now().strftime("%Y-%m-%d"),
        'threshold': 0.5  # Default threshold for binary prediction
    }
    
    # Save the model
    joblib.dump(model_data, output_path)
    logger.info(f"Model saved to {output_path}")

def train_and_save_model(
    data_path: str = None,
    output_path: str = 'models/churn_model.joblib',
    model_type: str = 'RandomForest',
    target_col: str = 'lost_program',
    df: pd.DataFrame = None
) -> None:
    """
    End-to-end function to train and save a churn prediction model.
    
    Args:
        data_path: Path to the input data CSV (if None, df must be provided)
        output_path: Path to save the model
        model_type: Type of model to train
        target_col: Name of the target column
        df: DataFrame with data (if provided, data_path is ignored)
    """
    # Load data
    if df is None:
        if data_path is None:
            raise ValueError("Either data_path or df must be provided")
        df = load_data(data_path)
    
    # Preprocess data
    df_processed = preprocess_data(df, target_col=target_col)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_processed, target_col=target_col)
    
    # Build preprocessing pipeline
    preprocessor, numerical_cols, categorical_cols = build_preprocessing_pipeline(X_train)
    
    # Train model
    pipeline, metrics = train_model(
        X_train, y_train, X_test, y_test, preprocessor, model_type=model_type
    )
    
    # Calculate feature importance
    feature_importance = calculate_feature_importance(pipeline, numerical_cols, categorical_cols)
    
    # Save model
    save_model(
        pipeline, 
        preprocessor,
        numerical_cols,
        categorical_cols,
        metrics,
        feature_importance,
        model_type,
        output_path
    )
    
    logger.info(f"Model training and saving complete. Model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    train_and_save_model(
        data_path="../Churn/features_extracted.csv",
        output_path="models/churn_model.joblib",
        model_type="RandomForest"
    )
