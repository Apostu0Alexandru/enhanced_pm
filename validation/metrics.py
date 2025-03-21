import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ValidationSuite:
    """Implements NASA's PHM validation protocol"""
    
    @staticmethod
    def phm_score(y_true, y_pred):
        """NASA's proprietary scoring function"""
        d = y_pred - y_true
        return np.mean(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1))
    
    @classmethod
    def full_report(cls, y_true, y_pred):
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'PHM': cls.phm_score(y_true, y_pred),
            'Early_Detection_Rate': cls._early_detection(y_true, y_pred)
        }
    
    @staticmethod
    def _early_detection(y_true, y_pred, threshold=50):
        valid = y_true > threshold
        return (y_pred[valid] < threshold).mean()
