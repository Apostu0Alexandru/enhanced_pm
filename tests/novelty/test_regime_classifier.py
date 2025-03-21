# tests/novelty/test_regime_classifier.py
import unittest
import numpy as np
import pandas as pd
from src.models.regime_classifier import OperationalRegimeClassifier, MATPLOTLIB_AVAILABLE

class TestRegimeClassifier(unittest.TestCase):
    def setUp(self):
        self.mock_data = pd.DataFrame({
            'op1': np.random.random(100),
            'op2': np.random.random(100),
            'op3': np.random.random(100)
        })
        
    def test_regime_count(self):
        """Test that classifier identifies the expected number of regimes"""
        classifier = OperationalRegimeClassifier(n_regimes=4)
        classifier.fit(self.mock_data, epochs=2)  # Just 2 epochs for testing
        
        # Predict regimes
        labels, _ = classifier.predict_regime(self.mock_data)
        
        # Check unique regimes
        unique_regimes = np.unique(labels)
        self.assertLessEqual(len(unique_regimes), 4)
        
    def test_visualization(self):
        """Test that visualization functions don't crash"""
        classifier = OperationalRegimeClassifier(n_regimes=3)
        classifier.fit(self.mock_data, epochs=2)
        
        # Should produce a figure without errors
        fig = classifier.visualize_regimes(self.mock_data)
        
        # Skip figure assertion if matplotlib is not available
        if MATPLOTLIB_AVAILABLE:
            self.assertIsNotNone(fig)
        else:
            # Just make sure the method runs without error when matplotlib is not available
            self.assertIsNone(fig)  # Expected behavior when matplotlib is not available
