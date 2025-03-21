# tests/evaluation/test_cross_condition.py
import unittest
import numpy as np
import os
from src.evaluation.cross_condition import load_and_preprocess_data

class TestCrossCondition(unittest.TestCase):
    def test_cross_dataset_loading(self):
        """Test loading and preprocessing both FD001 and FD002 datasets"""
        # Skip if data not available
        if not os.path.exists('data/FD001.txt') or not os.path.exists('data/FD002.txt'):
            self.skipTest("Dataset files not available")
            
        # Load preprocessed data (limited to first 100 samples per file for speed)
        X_train, y_train, X_test_fd001, y_test_fd001, X_test_fd002, y_test_fd002 = \
            load_and_preprocess_data('data/FD001.txt', 'data/FD002.txt', sequence_length=10)
            
        # Verify data shapes make sense
        self.assertEqual(X_train.ndim, 3)
        self.assertEqual(X_test_fd001.ndim, 3)
        self.assertEqual(X_test_fd002.ndim, 3)
        
        # Verify consistent feature dimensions across datasets
        self.assertEqual(X_train.shape[2], X_test_fd001.shape[2])
        self.assertEqual(X_train.shape[2], X_test_fd002.shape[2])
