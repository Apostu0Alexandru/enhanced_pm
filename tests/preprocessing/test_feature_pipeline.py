# tests/preprocessing/test_feature_pipeline.py
import unittest
import numpy as np
import pandas as pd
from src.preprocessing.feature_pipeline import create_hierarchical_feature_pipeline, prepare_tcn_sequences

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Create a small mock dataset with CMAPSS structure
        self.mock_data = pd.DataFrame({
            'unit': [1, 1, 1, 2, 2],
            'cycle': [1, 2, 3, 1, 2],
            'op1': [0.5, 0.5, 0.6, 0.4, 0.5],
            'op2': [0.7, 0.7, 0.8, 0.6, 0.7],
            'op3': [0.3, 0.3, 0.4, 0.2, 0.3],
            'sensor1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sensor2': [0.2, 0.3, 0.4, 0.5, 0.6]
        })
        self.mock_data_path = 'mock_data.csv'
        self.mock_data.to_csv(self.mock_data_path, sep=' ', index=False)
        
    def test_fd001_processing(self):
        """Test single condition processing works"""
        # Mock preprocessing and patching the actual file reading
        processed_data = create_hierarchical_feature_pipeline('FD001', self.mock_data_path)
        
        # Check that expected columns are present
        expected_columns = ['sensor1_rolling_mean', 'sensor2_rolling_mean']
        for col in expected_columns:
            self.assertIn(col, processed_data.columns)
            
    def test_fd002_processing(self):
        """Test multi condition processing works"""
        processed_data = create_hierarchical_feature_pipeline('FD002', self.mock_data_path)
        
        # Check that operational regimes were identified
        self.assertIn('op_regime', processed_data.columns)
        
    def test_sequence_preparation(self):
        """Test sequence preparation for TCN"""
        processed_data = create_hierarchical_feature_pipeline('FD001', self.mock_data_path)
        X, y = prepare_tcn_sequences(processed_data, sequence_length=2)
        
        # Check dimensions make sense
        self.assertEqual(X.ndim, 3)  # (samples, sequence_length, features)
        self.assertEqual(X.shape[1], 2)  # sequence_length = 2
        
    def tearDown(self):
        import os
        if os.path.exists(self.mock_data_path):
            os.remove(self.mock_data_path)
