import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.survival_models import main_survival

if __name__ == "__main__":
    ensemble_model, metrics = main_survival("../../data/FD001.txt")
    print(f"Model training completed with RMSE: {metrics['rmse']:.4f}")
