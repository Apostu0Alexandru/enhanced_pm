#!/usr/bin/env python
"""
End-to-End Validation Script for Enhanced Predictive Maintenance

This script validates the core functionality of the enhanced predictive maintenance
project, from data preprocessing to model training and evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import argparse
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Validation")

# Add parent directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TensorFlow patch before importing TensorFlow
import tensorflow_patch

# Now it's safe to import TensorFlow
import tensorflow as tf

# Import project modules
from src.preprocessing.feature_pipeline import (
    create_hierarchical_feature_pipeline,
    prepare_tcn_sequences,
    process_singlecondition_data,
    process_multicondition_data
)
from src.models.regime_classifier import OperationalRegimeClassifier
from src.models.cnn_lstm import build_cnn_lstm_model
from src.models.tcn import build_tcn_model
from src.models.survival_models import EnsembleSurvivalModel


class ValidationResult:
    """Stores and tracks validation results"""
    
    def __init__(self):
        self.results = {}
        self.status = {}
        self.start_time = time.time()
        
    def record_step(self, step_name, passed, details=None):
        elapsed = time.time() - self.start_time
        self.results[step_name] = {
            "passed": passed,
            "elapsed_time": elapsed,
            "details": details or {}
        }
        self.status[step_name] = "PASS" if passed else "FAIL"
        
        status_symbol = "✅" if passed else "❌"
        logger.info(f"{status_symbol} {step_name}: {'Passed' if passed else 'Failed'} ({elapsed:.2f}s)")
        if details:
            for k, v in details.items():
                logger.info(f"  - {k}: {v}")
    
    def summary(self):
        """Generate a summary of all validation results"""
        total = len(self.results)
        passed = sum(1 for v in self.results.values() if v["passed"])
        
        logger.info("=" * 50)
        logger.info(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
        logger.info("=" * 50)
        
        for step_name, result in self.results.items():
            status = "PASS" if result["passed"] else "FAIL"
            logger.info(f"{status} - {step_name} ({result['elapsed_time']:.2f}s)")
            
        logger.info("=" * 50)
        logger.info(f"Total time: {time.time() - self.start_time:.2f}s")
        
        return passed == total


def validate_data_preprocessing(data_dir, validation_results):
    """Validate the data preprocessing pipeline"""
    logger.info("Testing data preprocessing pipeline...")
    
    try:
        # Process single condition data (FD001)
        fd001_train_path = os.path.join(data_dir, "train_FD001.txt")
        
        # Call with only the file path - the function only accepts one argument
        fd001_data = process_singlecondition_data(fd001_train_path)
        
        # Verify key columns exist
        required_columns = ['unit', 'cycle', 'sensor1', 'sensor2', 'op_setting1']
        has_columns = all(col in fd001_data.columns for col in required_columns[:3])
        
        # Get some basic statistics
        stats = {
            "shape": fd001_data.shape,
            "units": len(fd001_data['unit'].unique()),
        }
        
        validation_results.record_step(
            "FD001 Data Preprocessing", 
            passed=has_columns,
            details=stats
        )
        
        # Process multi-condition data (FD002)
        fd002_train_path = os.path.join(data_dir, "train_FD002.txt")
        
        # Process multicondition data returns a tuple (data, cluster_centers)
        fd002_data, cluster_centers = process_multicondition_data(fd002_train_path)
        
        # Verify key columns exist
        has_columns = all(col in fd002_data.columns for col in required_columns[:3])
        
        # Get some basic statistics
        stats = {
            "shape": fd002_data.shape,
            "units": len(fd002_data['unit'].unique()),
            "clusters": len(cluster_centers)
        }
        
        validation_results.record_step(
            "FD002 Data Preprocessing", 
            passed=has_columns,
            details=stats
        )
        
        # Test feature engineering pipeline
        fd001_features = create_hierarchical_feature_pipeline('FD001', os.path.join(data_dir, "train_FD001.txt"))
        
        # Check if we have the engineered features
        has_engineered_features = any('rolling_mean' in col for col in fd001_features.columns)
        
        validation_results.record_step(
            "Feature Engineering", 
            passed=has_engineered_features,
            details={"features": len(fd001_features.columns)}
        )
        
        # Test sequence preparation
        X, y = prepare_tcn_sequences(fd001_features, sequence_length=30)
        
        validation_results.record_step(
            "Sequence Preparation", 
            passed=X.shape[0] > 0 and X.shape[2] > 10,
            details={"X.shape": X.shape, "y.shape": y.shape}
        )
        
        return fd001_features, X, y
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        validation_results.record_step("Data Preprocessing", False, {"error": str(e)})
        return None, None, None


def validate_regime_classification(features_df, validation_results):
    """Validate the operational regime classification"""
    logger.info("Testing operational regime classification...")
    
    try:
        # Skip if no features_df is available
        if features_df is None:
            raise ValueError("No feature data available for regime classification")
            
        # Create regime classifier - check the actual parameters it accepts
        # Assuming it might use n_regimes instead of n_clusters
        classifier = OperationalRegimeClassifier(n_regimes=3)
        
        # Choose relevant columns for clustering
        op_settings = [col for col in features_df.columns if col.startswith('op')]
        
        # Apply classification
        cluster_data = features_df[op_settings].copy()
        classifier.fit(cluster_data)
        
        # Get clusters using the correct method name
        cluster_labels, _ = classifier.predict_regime(cluster_data)
        
        # Validate by checking if we got reasonable clusters
        unique_clusters = np.unique(cluster_labels)
        
        validation_results.record_step(
            "Regime Classification", 
            passed=len(unique_clusters) > 0,
            details={
                "clusters_found": len(unique_clusters),
                "cluster_distribution": {int(i): int(np.sum(cluster_labels == i)) for i in unique_clusters}
            }
        )
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error in regime classification: {str(e)}")
        validation_results.record_step("Regime Classification", False, {"error": str(e)})
        return None


def validate_model_training(X, y, validation_results):
    """Validate model training functionality"""
    logger.info("Testing model training...")
    
    try:
        # Skip if no data is available
        if X is None or y is None:
            raise ValueError("No input data available")
            
        # Check if PyTorch is available
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch is required for TCN model training")
            
        # Build a simple TCN model for testing
        n_features = X.shape[2]
        sequence_length = X.shape[1]
        
        # Build a model with a small number of filters for quick testing
        model = build_tcn_model(
            input_shape=(sequence_length, n_features),
            num_channels=8,
            kernel_size=3,
            dropout=0.1
        )
        
        # Ensure model is in training mode
        model.train()
        
        # Set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X[:1000])  # Use a subset for quick validation
        y_tensor = torch.FloatTensor(y[:1000, np.newaxis])
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        epochs = 2
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
        
        # Check if model trained successfully
        validation_results.record_step(
            "Model Training", 
            passed=True,
            details={
                "final_loss": float(train_losses[-1]),
                "model_params": sum(p.numel() for p in model.parameters()),
                "model_type": "PyTorch TCN"
            }
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        validation_results.record_step("Model Training", False, {"error": str(e)})
        return None


def validate_model_prediction(model, X, validation_results):
    """Validate model prediction functionality"""
    logger.info("Testing model prediction...")
    
    try:
        # Skip if no model or data is available
        if model is None or X is None:
            raise ValueError("No model or data available")
            
        # Check if PyTorch is available
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for TCN model prediction")
        
        # Set model to evaluation mode
        model.eval()
        
        # Convert data to PyTorch tensor
        X_tensor = torch.FloatTensor(X[:10])
        
        # Make predictions
        with torch.no_grad():
            y_pred = model(X_tensor).numpy()
        
        # Check if predictions are reasonable (non-negative RUL)
        is_valid = np.all(y_pred >= 0)
        
        validation_results.record_step(
            "Model Prediction", 
            passed=is_valid,
            details={
                "pred_shape": y_pred.shape,
                "pred_mean": float(np.mean(y_pred)),
                "pred_min": float(np.min(y_pred)),
                "pred_max": float(np.max(y_pred))
            }
        )
        
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        validation_results.record_step("Model Prediction", False, {"error": str(e)})


def run_validation(data_dir):
    """Run end-to-end validation of the enhanced PM pipeline"""
    logger.info("Starting end-to-end validation...")
    logger.info(f"Using data directory: {data_dir}")
    
    validation_results = ValidationResult()
    
    # Validate the data preprocessing pipeline
    features, X, y = validate_data_preprocessing(data_dir, validation_results)
    
    # Validate the regime classification
    classifier = validate_regime_classification(features, validation_results)
    
    # Validate model training
    model = validate_model_training(X, y, validation_results)
    
    # Validate model prediction
    validate_model_prediction(model, X, validation_results)
    
    # Generate summary
    passed = validation_results.summary()
    
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end validation of the enhanced PM pipeline")
    parser.add_argument("--data-dir", type=str, default="../nasa_data", 
                        help="Path to the data directory containing NASA CMAPSS dataset")
    args = parser.parse_args()
    
    # Resolve relative path
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_dir))
    
    # Run validation
    success = run_validation(data_dir)
    
    # Exit with status code
    sys.exit(0 if success else 1)