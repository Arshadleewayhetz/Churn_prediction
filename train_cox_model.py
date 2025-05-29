#!/usr/bin/env python
"""
Script to train and save the Cox proportional hazards model for churn prediction.
"""

import argparse
import logging
import os
from cox_churn_model import train_and_evaluate_cox_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Parse arguments and train the Cox model."""
    parser = argparse.ArgumentParser(description="Train and save a Cox proportional hazards model for churn prediction")
    
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
    
    parser.add_argument(
        "--penalizer", 
        type=float, 
        default=0.1,
        help="Regularization strength for Cox model"
    )
    
    parser.add_argument(
        "--l1-ratio", 
        type=float, 
        default=0.2,
        help="L1 vs L2 regularization mix (0=ridge, 1=lasso)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    logger.info(f"Training Cox model with arguments: {args}")
    
    # Train and evaluate model
    model, metrics = train_and_evaluate_cox_model(
        data_path=args.data_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Print metrics
    logger.info("Cox Model Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"Cox model saved to {args.output_path}")
    logger.info("Model training complete")

if __name__ == "__main__":
    main()

