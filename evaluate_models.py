#!/usr/bin/env python
"""
Script to evaluate and compare different churn prediction models.
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, 
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from lifelines.utils import concordance_index
import joblib

from cox_churn_model import RobustCoxChurnModel
from app.utils.model_trainer import train_and_save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Load and preprocess the dataset.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Replace infinity values with NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Dataset shape: {df.shape}")
    
    return df

def train_traditional_models(df, output_dir="models", target_col="lost_program"):
    """
    Train traditional models (Logistic Regression, Random Forest, Gradient Boosting).
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save models
        target_col: Target column name
        
    Returns:
        Dictionary of trained models and their metrics
    """
    logger.info("Training traditional models...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Train models
    model_types = ["LogisticRegression", "RandomForest", "GradientBoosting"]
    models = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type}...")
        output_path = os.path.join(output_dir, f"{model_type.lower()}_model.joblib")
        
        # Train and save model
        train_and_save_model(
            data_path=None,  # We'll pass the DataFrame directly
            output_path=output_path,
            model_type=model_type,
            target_col=target_col,
            df=df  # Pass DataFrame directly
        )
        
        # Load the trained model
        model_data = joblib.load(output_path)
        
        # Store model and metrics
        models[model_type] = {
            "model_data": model_data,
            "metrics": model_data.get("model_metrics", {})
        }
        
        logger.info(f"{model_type} metrics: {models[model_type]['metrics']}")
    
    return models

def train_cox_model(df, output_dir="models", target_col="lost_program"):
    """
    Train Cox proportional hazards model.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save models
        target_col: Target column name
        
    Returns:
        Trained Cox model and metrics
    """
    logger.info("Training Cox proportional hazards model...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(output_dir, "cox_model.joblib")
    
    # Create and train Cox model
    cox_model = RobustCoxChurnModel(penalizer=0.1, l1_ratio=0.2, robust=True)
    cox_model.fit(df, event_col=target_col)
    
    # Save model
    cox_model.save(output_path)
    
    logger.info(f"Cox model metrics: {cox_model.metrics}")
    
    return cox_model

def evaluate_models_on_test_data(df, traditional_models, cox_model, test_size=0.2, random_state=42, target_col="lost_program"):
    """
    Evaluate all models on a common test dataset.
    
    Args:
        df: Input DataFrame
        traditional_models: Dictionary of traditional models
        cox_model: Cox model
        test_size: Proportion of data to use for testing
        random_state: Random seed
        target_col: Target column name
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating models on test data...")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # Preprocess data (similar to what's done in model_trainer.py)
    from app.utils.model_trainer import preprocess_data
    df_processed = preprocess_data(df, target_col=target_col)
    
    # Split data
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Evaluate traditional models
    traditional_results = {}
    for model_type, model_info in traditional_models.items():
        logger.info(f"Evaluating {model_type}...")
        
        # Get model data
        model_data = model_info["model_data"]
        
        # Create pipeline
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', model_data["preprocessor"]),
            ('model', model_data["model"])
        ])
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Store results
        traditional_results[model_type] = metrics
        
        logger.info(f"{model_type} test metrics: {metrics}")
    
    # Evaluate Cox model
    logger.info("Evaluating Cox model...")
    
    # Preprocess test data for Cox model
    X_test_processed = cox_model.preprocessor.transform(X_test)
    X_test_selected = cox_model.feature_selector.transform(X_test_processed)
    
    # Create DataFrame for Cox model
    test_df = pd.DataFrame(
        X_test_selected, 
        columns=[f"feature_{i}" for i in range(X_test_selected.shape[1])]
    )
    test_df[cox_model.duration_col] = 180  # Default duration for prediction
    test_df[cox_model.event_col] = y_test.values
    
    # Make predictions
    churn_prob = cox_model._predict_churn_probability(X_test_selected)
    churn_pred = (churn_prob >= cox_model.threshold).astype(int)
    
    # Calculate metrics
    cox_metrics = calculate_metrics(y_test, churn_pred, churn_prob)
    
    # Add C-index
    cox_metrics["c_index"] = concordance_index(
        test_df[cox_model.duration_col],
        -cox_model.model.predict_partial_hazard(test_df),
        test_df[cox_model.event_col]
    )
    
    logger.info(f"Cox model test metrics: {cox_metrics}")
    
    # Combine results
    all_results = {
        **traditional_results,
        "CoxPH": cox_metrics
    }
    
    return all_results

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # Calculate other metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }

def plot_results(results, output_dir="results"):
    """
    Plot evaluation results.
    
    Args:
        results: Dictionary of evaluation metrics
        output_dir: Directory to save plots
    """
    logger.info("Plotting results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot metrics comparison
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    
    plt.figure(figsize=(12, 8))
    
    # Create DataFrame for plotting
    metrics_data = []
    for model, metrics in results.items():
        for metric in metrics_to_plot:
            if metric in metrics:
                metrics_data.append({
                    "Model": model,
                    "Metric": metric,
                    "Value": metrics[metric]
                })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot
    sns.barplot(x="Metric", y="Value", hue="Model", data=metrics_df)
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    
    # Plot confusion matrices
    for model, metrics in results.items():
        if "confusion_matrix" in metrics:
            plt.figure(figsize=(8, 6))
            cm = metrics["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Not Churn", "Churn"],
                        yticklabels=["Not Churn", "Churn"])
            plt.title(f"{model} Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model}_confusion_matrix.png"))
    
    logger.info(f"Plots saved to {output_dir}")

def main():
    """Main function to evaluate models."""
    parser = argparse.ArgumentParser(description="Evaluate and compare churn prediction models")
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="../Churn/features_extracted.csv",
        help="Path to the input data CSV"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models",
        help="Directory to save models"
    )
    
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--target-col", 
        type=str, 
        default="lost_program",
        help="Target column name"
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
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data_path)
    
    # Train traditional models
    traditional_models = train_traditional_models(
        df, 
        output_dir=args.output_dir, 
        target_col=args.target_col
    )
    
    # Train Cox model
    cox_model = train_cox_model(
        df, 
        output_dir=args.output_dir, 
        target_col=args.target_col
    )
    
    # Evaluate models
    results = evaluate_models_on_test_data(
        df,
        traditional_models,
        cox_model,
        test_size=args.test_size,
        random_state=args.random_state,
        target_col=args.target_col
    )
    
    # Plot results
    plot_results(results, output_dir=args.results_dir)
    
    # Print summary
    logger.info("Model Evaluation Summary:")
    for model, metrics in results.items():
        logger.info(f"{model}:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Model evaluation complete")

if __name__ == "__main__":
    main()

