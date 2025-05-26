#!/usr/bin/env python
"""
Script to train and save the churn prediction model.
"""

import argparse
import logging
from app.utils.model_trainer import train_and_save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Parse arguments and train the model."""
    parser = argparse.ArgumentParser(description="Train and save a churn prediction model")
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="../Churn/features_extracted.csv",
        help="Path to the input data CSV"
    )
    
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="models/churn_model.joblib",
        help="Path to save the trained model"
    )
    
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="RandomForest",
        choices=["LogisticRegression", "RandomForest", "GradientBoosting"],
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--target-col", 
        type=str, 
        default="lost_program",
        help="Name of the target column"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Training model with arguments: {args}")
    
    # Train and save the model
    train_and_save_model(
        data_path=args.data_path,
        output_path=args.output_path,
        model_type=args.model_type,
        target_col=args.target_col
    )
    
    logger.info("Model training complete")

if __name__ == "__main__":
    main()

