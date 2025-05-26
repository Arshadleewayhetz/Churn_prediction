#!/usr/bin/env python
# coding: utf-8

"""
Churn Prediction Statistical Model
---------------------------------
This script implements a comprehensive statistical model to identify churn reasons
and their impact percentages. It handles highly skewed data with more than 80% zero values
in many features.

Author: Codegen
Date: 2025-05-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
import shap
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def load_data(file_path):
    """Load the dataset and perform initial preprocessing."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Replace infinity values with NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Print basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable distribution:\n{df['lost_program'].value_counts(normalize=True) * 100}")
    
    return df

def analyze_data_quality(df):
    """Analyze data quality issues like missing values and zero values."""
    # Check missing values
    missing_values = df.isnull().mean() * 100
    print("\nPercentage of missing values per column:")
    print(missing_values[missing_values > 0].sort_values(ascending=False).head(20))
    
    # Check zero values
    zero_values = (df == 0).mean() * 100
    print("\nPercentage of zero values per column:")
    print(zero_values[zero_values > 80].sort_values(ascending=False).head(20))
    
    return missing_values, zero_values

def preprocess_data(df, missing_threshold=80, zero_threshold=80):
    """Preprocess the data by handling missing values, removing leakage, etc."""
    print("\nPreprocessing data...")
    
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
    
    print(f"Removed {len(high_missing_cols)} columns with >{missing_threshold}% missing values")
    print(f"Removed {len(high_zero_cols)} columns with >{zero_threshold}% zero values")
    print(f"Final dataset shape: {df.shape}")
    
    return df

def split_data(df, target_col='lost_program', test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(X_train):
    """Build a preprocessing pipeline for numerical and categorical features."""
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

def train_models(X_train, y_train, X_test, y_test, preprocessor):
    """Train multiple models and evaluate their performance."""
    print("\nTraining models...")
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
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
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Track best model
        if auc > best_score:
            best_score = auc
            best_model = name
    
    print(f"\nBest model: {best_model} with AUC: {best_score:.4f}")
    
    return results, best_model

def analyze_feature_importance(results, best_model, X_train, y_train, X_test, numerical_cols, categorical_cols):
    """Analyze feature importance to identify churn reasons."""
    print("\nAnalyzing feature importance...")
    
    pipeline = results[best_model]['pipeline']
    model = pipeline.named_steps['model']
    
    # Get preprocessed data
    X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train)
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Add numerical feature names
    feature_names.extend(numerical_cols)
    
    # Add one-hot encoded feature names
    if categorical_cols:
        ohe = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names)
    
    # Get feature importance based on model type
    if best_model == 'Logistic Regression':
        # For logistic regression, use coefficients
        importance = np.abs(model.coef_[0])
        
    elif best_model == 'Random Forest' or best_model == 'Gradient Boosting':
        # For tree-based models, use feature importance
        importance = model.feature_importances_
        
        # Also calculate permutation importance for more reliable results
        perm_importance = permutation_importance(pipeline, X_test, y_train, n_repeats=10, random_state=42)
        perm_importance_mean = perm_importance.importances_mean
        
        # Create DataFrame for permutation importance
        perm_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Permutation_Importance': perm_importance_mean
        })
        perm_importance_df = perm_importance_df.sort_values('Permutation_Importance', ascending=False)
        
        print("\nTop 15 features by permutation importance:")
        print(perm_importance_df.head(15))
    
    # Create DataFrame for model-based importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    print("\nTop 15 features by model importance:")
    print(importance_df.head(15))
    
    # Calculate percentage contribution to churn
    total_importance = importance_df['Importance'].sum()
    importance_df['Contribution_Percentage'] = (importance_df['Importance'] / total_importance) * 100
    
    print("\nTop 15 features by contribution percentage:")
    print(importance_df[['Feature', 'Contribution_Percentage']].head(15))
    
    # Try to use SHAP for more detailed analysis
    try:
        # Create a SHAP explainer
        explainer = shap.Explainer(model, X_train_processed)
        shap_values = explainer(X_train_processed)
        
        # Get global feature importance
        shap_importance = np.abs(shap_values.values).mean(0)
        shap_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': shap_importance
        })
        shap_importance_df = shap_importance_df.sort_values('SHAP_Importance', ascending=False)
        
        print("\nTop 15 features by SHAP importance:")
        print(shap_importance_df.head(15))
        
        # Calculate percentage contribution based on SHAP
        total_shap_importance = shap_importance_df['SHAP_Importance'].sum()
        shap_importance_df['SHAP_Contribution_Percentage'] = (shap_importance_df['SHAP_Importance'] / total_shap_importance) * 100
        
        print("\nTop 15 features by SHAP contribution percentage:")
        print(shap_importance_df[['Feature', 'SHAP_Contribution_Percentage']].head(15))
        
        return importance_df, shap_importance_df
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return importance_df, None

def cox_proportional_hazards_analysis(df):
    """Perform survival analysis using Cox Proportional Hazards model."""
    print("\nPerforming Cox Proportional Hazards analysis...")
    
    # Prepare data for survival analysis
    # We need duration (time to event) and event indicator
    if 'days_rem_to_end_contract' in df.columns:
        # Use remaining days to end of contract as duration
        df['duration'] = df['days_rem_to_end_contract']
    elif 'endDate' in df.columns and 'startDate' in df.columns:
        # Calculate duration from start to end date
        df['startDate'] = pd.to_datetime(df['startDate'])
        df['endDate'] = pd.to_datetime(df['endDate'])
        df['duration'] = (df['endDate'] - df['startDate']).dt.days
    else:
        print("No suitable duration column found for survival analysis")
        return None
    
    # Event indicator is lost_program
    df['event'] = df['lost_program']
    
    # Select features for Cox model
    # Remove highly correlated features and non-numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cox_features = [col for col in numeric_cols if col not in ['lost_program', 'duration', 'event']]
    
    # Handle missing values
    cox_df = df[cox_features + ['duration', 'event']]
    cox_df = cox_df.replace([np.inf, -np.inf], np.nan)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    cox_df[cox_features] = imputer.fit_transform(cox_df[cox_features])
    
    # Standardize features
    scaler = StandardScaler()
    cox_df[cox_features] = scaler.fit_transform(cox_df[cox_features])
    
    # Fit Cox model
    cph = CoxPHFitter()
    try:
        cph.fit(cox_df, duration_col='duration', event_col='event')
        
        # Get summary
        cox_summary = cph.summary
        
        # Sort by p-value to find significant features
        cox_summary = cox_summary.sort_values('p')
        
        print("\nCox Proportional Hazards Model Summary (top 15 significant features):")
        print(cox_summary.head(15))
        
        # Calculate hazard ratio and percentage impact
        cox_summary['hazard_ratio'] = np.exp(cox_summary['coef'])
        cox_summary['percentage_impact'] = (cox_summary['hazard_ratio'] - 1) * 100
        
        print("\nFeature impact on churn risk (top 15):")
        impact_summary = cox_summary[['hazard_ratio', 'percentage_impact', 'p']].sort_values('percentage_impact', ascending=False)
        print(impact_summary.head(15))
        
        return cox_summary
        
    except Exception as e:
        print(f"Cox model fitting failed: {e}")
        return None

def main():
    """Main function to run the churn prediction analysis."""
    # Load data
    # Assuming the data file is in the same directory
    # If not, provide the correct path
    try:
        df = load_data('../Churn/features_extracted.csv')
    except:
        print("Could not find the data file. Please provide the correct path.")
        return
    
    # Analyze data quality
    missing_values, zero_values = analyze_data_quality(df)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Build preprocessing pipeline
    preprocessor, numerical_cols, categorical_cols = build_preprocessing_pipeline(X_train)
    
    # Train models
    results, best_model = train_models(X_train, y_train, X_test, y_test, preprocessor)
    
    # Analyze feature importance
    importance_df, shap_importance_df = analyze_feature_importance(
        results, best_model, X_train, y_train, X_test, numerical_cols, categorical_cols
    )
    
    # Perform Cox Proportional Hazards analysis
    cox_summary = cox_proportional_hazards_analysis(df_processed)
    
    # Print final conclusions
    print("\n=== FINAL CONCLUSIONS ===")
    print("\nTop 10 features contributing to churn:")
    
    if shap_importance_df is not None:
        # Use SHAP importance if available
        for i, (feature, contribution) in enumerate(
            zip(shap_importance_df['Feature'].head(10), 
                shap_importance_df['SHAP_Contribution_Percentage'].head(10))
        ):
            print(f"{i+1}. {feature}: {contribution:.2f}%")
    else:
        # Use model importance otherwise
        for i, (feature, contribution) in enumerate(
            zip(importance_df['Feature'].head(10), 
                importance_df['Contribution_Percentage'].head(10))
        ):
            print(f"{i+1}. {feature}: {contribution:.2f}%")
    
    # Additional insights from Cox model
    if cox_summary is not None:
        print("\nKey risk factors from survival analysis:")
        significant_features = cox_summary[cox_summary['p'] < 0.05].sort_values('percentage_impact', ascending=False)
        
        for i, (feature, impact, p_value) in enumerate(
            zip(significant_features.index[:10], 
                significant_features['percentage_impact'][:10],
                significant_features['p'][:10])
        ):
            impact_direction = "increases" if impact > 0 else "decreases"
            print(f"{i+1}. {feature}: {impact_direction} churn risk by {abs(impact):.2f}% (p={p_value:.4f})")

if __name__ == "__main__":
    main()

