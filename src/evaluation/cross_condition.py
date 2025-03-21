import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.feature_pipeline import create_hierarchical_feature_pipeline, prepare_tcn_sequences
from models.cnn_lstm import build_cnn_lstm_model
from models.pinn import PINN, TurbofanPhysics
from models.survival_models import EnsembleSurvivalModel

def load_and_preprocess_data(fd001_path, fd002_path, sequence_length=100):
    """
    Load and preprocess both FD001 and FD002 datasets
    
    Returns:
        Tuple of (X_train, y_train, X_test_fd001, y_test_fd001, X_test_fd002, y_test_fd002)
    """
    print("Processing FD001 dataset (single condition)...")
    fd001_data = create_hierarchical_feature_pipeline('FD001', fd001_path)
    
    # Split FD001 into train and test (80/20)
    units = fd001_data['unit'].unique()
    np.random.seed(42)
    train_units = np.random.choice(units, size=int(0.8 * len(units)), replace=False)
    test_units = np.array([u for u in units if u not in train_units])
    
    train_data = fd001_data[fd001_data['unit'].isin(train_units)]
    test_data_fd001 = fd001_data[fd001_data['unit'].isin(test_units)]
    
    # Create sequences for TCN/RNN models
    X_train, y_train = prepare_tcn_sequences(train_data, sequence_length)
    X_test_fd001, y_test_fd001 = prepare_tcn_sequences(test_data_fd001, sequence_length)
    
    print("Processing FD002 dataset (multiple conditions)...")
    fd002_data = create_hierarchical_feature_pipeline('FD002', fd002_path)
    X_test_fd002, y_test_fd002 = prepare_tcn_sequences(fd002_data, sequence_length)
    
    return X_train, y_train, X_test_fd001, y_test_fd001, X_test_fd002, y_test_fd002

def train_and_evaluate_model(model_name, model_builder, X_train, y_train, 
                            X_test_fd001, y_test_fd001, X_test_fd002, y_test_fd002,
                            epochs=100, batch_size=32):
    """
    Train a model on FD001 and evaluate on both FD001 and FD002
    
    Args:
        model_name: Name of the model for reporting
        model_builder: Function that returns a compiled model
        X_train, y_train: Training data from FD001
        X_test_fd001, y_test_fd001: Test data from FD001
        X_test_fd002, y_test_fd002: Test data from FD002
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with model name, RMSE on FD001, RMSE on FD002, and degradation factor
    """
    print(f"\nTraining {model_name} model...")
    
    # Build and train model
    model = model_builder(X_train.shape[1:])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on FD001
    y_pred_fd001 = model.predict(X_test_fd001)
    rmse_fd001 = np.sqrt(mean_squared_error(y_test_fd001, y_pred_fd001))
    mae_fd001 = mean_absolute_error(y_test_fd001, y_pred_fd001)
    
    # Evaluate on FD002
    y_pred_fd002 = model.predict(X_test_fd002)
    rmse_fd002 = np.sqrt(mean_squared_error(y_test_fd002, y_pred_fd002))
    mae_fd002 = mean_absolute_error(y_test_fd002, y_pred_fd002)
    
    # Calculate degradation factor
    degradation_factor = ((rmse_fd002 - rmse_fd001) / rmse_fd001) * 100
    
    print(f"{model_name} Results:")
    print(f"  RMSE on FD001 (single condition): {rmse_fd001:.4f}")
    print(f"  RMSE on FD002 (multiple conditions): {rmse_fd002:.4f}")
    print(f"  Degradation Factor: {degradation_factor:.2f}%")
    
    results = {
        'model_name': model_name,
        'rmse_fd001': rmse_fd001,
        'mae_fd001': mae_fd001,
        'rmse_fd002': rmse_fd002,
        'mae_fd002': mae_fd002,
        'degradation_factor': degradation_factor,
        'history': history.history
    }
    
    return results

def visualize_degradation(results_list):
    """
    Create visualizations of model performance and degradation
    
    Args:
        results_list: List of result dictionaries from train_and_evaluate_model
    """
    # Prepare data for plotting
    model_names = [r['model_name'] for r in results_list]
    rmse_fd001 = [r['rmse_fd001'] for r in results_list]
    rmse_fd002 = [r['rmse_fd002'] for r in results_list]
    degradation = [r['degradation_factor'] for r in results_list]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE comparison
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, rmse_fd001, width, label='FD001 (Single Condition)')
    ax1.bar(x + width/2, rmse_fd002, width, label='FD002 (Multiple Conditions)')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    
    # Degradation factor
    ax2.bar(model_names, degradation, color='salmon')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Degradation Factor (%)')
    ax2.set_title('Cross-Condition Degradation')
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('cross_condition_results.png', dpi=300)
    plt.show()
    
    return fig

def run_cross_condition_validation():
    """Run the complete cross-condition validation process"""
    # Define paths
    fd001_path = os.path.join('data', 'FD001.txt')
    fd002_path = os.path.join('data', 'FD002.txt')
    
    # Load and preprocess data
    X_train, y_train, X_test_fd001, y_test_fd001, X_test_fd002, y_test_fd002 = load_and_preprocess_data(
        fd001_path, fd002_path
    )
    
    # Define model builders for different architectures
    def build_cnn_lstm(input_shape):
        return build_cnn_lstm_model(input_shape)
    
    def build_pinn(input_shape):
        physics_model = TurbofanPhysics()
        pinn = PINN(input_shape[0], physics_model)
        pinn.compile(optimizer=tf.keras.optimizers.Adam(0.001))
        return pinn
    
    # Train and evaluate models
    results = []
    
    # CNN-LSTM model
    cnn_lstm_results = train_and_evaluate_model(
        'CNN-LSTM', build_cnn_lstm,
        X_train, y_train, X_test_fd001, y_test_fd001, X_test_fd002, y_test_fd002
    )
    results.append(cnn_lstm_results)
    
    # PINN model
    pinn_results = train_and_evaluate_model(
        'PINN', build_pinn,
        X_train, y_train, X_test_fd001, y_test_fd001, X_test_fd002, y_test_fd002
    )
    results.append(pinn_results)
    
    # Visualize results
    fig = visualize_degradation(results)
    
    return results, fig

if __name__ == "__main__":
    results, _ = run_cross_condition_validation()
    
    # Export results to CSV
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'history'} 
        for r in results
    ])
    
    results_df.to_csv('cross_condition_results.csv', index=False)
