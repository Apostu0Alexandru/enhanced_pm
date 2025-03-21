import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_lstm import main

if __name__ == "__main__":
    model, history, rmse = main("../../data/FD001.txt")
    print(f"Model training completed with RMSE: {rmse:.4f}")
