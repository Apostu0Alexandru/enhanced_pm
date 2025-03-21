import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pinn import main_pinn

if __name__ == "__main__":
    model, history, rmse = main_pinn("../../data/FD001.txt", physics_weight=0.3)
    print(f"Model training completed with RMSE: {rmse:.4f}")
