# tests/test_full_pipeline.py
import unittest
import numpy as np
import os
import tensorflow as tf
from src.preprocessing.feature_pipeline import create_hierarchical_feature_pipeline, prepare_tcn_sequences
from src.models.cnn_lstm import build_cnn_lstm_model

class TestFullPipeline(unittest.TestCase):
    def test_end_to_end_small_example(self):
        """Test full preprocessing and prediction pipeline on small example"""
        # Skip if no data available
        if not os.path.exists('data/FD001.txt'):
            self.skipTest("FD001.txt not available")
        
        # Process first 100 rows only for speed
        processed_data = create_hierarchical_feature_pipeline('FD001', 'data/FD001.txt')
        processed_data = processed_data.head(100)
        
        # Create sequences
        X, y = prepare_tcn_sequences(processed_data, sequence_length=10)
        
        # Build model
        model = build_cnn_lstm_model(X.shape[1:])
        
        # Make a prediction (just testing the flow, not the accuracy)
        predictions = model.predict(X[:5])
        
        # Check predictions shape
        self.assertEqual(predictions.shape, (5, 1))
