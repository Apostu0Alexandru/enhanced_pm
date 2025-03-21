import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.feature_pipeline import create_hierarchical_feature_pipeline, prepare_tcn_sequences
from models.cnn_lstm import build_cnn_lstm_model
from models.pinn import PINN, TurbofanPhysics

# 1. Linear Degradation Model
class LinearDegradationModel:
    """Simple linear degradation model RUL = alpha * t + beta"""
    
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.model = LinearRegression()
    
    def fit(self, X_train, y_train):
        """
        Fit the linear model.
        
        Args:
            X_train: Training features (we only use time/cycle features)
            y_train: Target RUL values
        """
        # For linear degradation, we only need the time/cycle feature
        # We'll use the mean of each window as an indicator of the cycle
        if len(X_train.shape) == 3:  # If X is in sequence format (batch, seq_len, features)
            # Extract cycle information (average of sequence)
            X_cycle = np.mean(X_train[:, :, 1], axis=1).reshape(-1, 1)  # Assuming cycle is feature index 1
        else:
            # If not sequence data, assume cycle is a direct feature
            X_cycle = X_train[:, 1].reshape(-1, 1)  # Assuming cycle is feature index 1
        
        self.model.fit(X_cycle, y_train)
        self.alpha = self.model.coef_[0]
        self.beta = self.model.intercept_
        
        print(f"Fitted Linear Degradation Model: RUL = {self.alpha:.4f} * t + {self.beta:.4f}")
        
        return self
    
    def predict(self, X_test):
        """Predict RUL values"""
        if len(X_test.shape) == 3:  # If X is in sequence format
            X_cycle = np.mean(X_test[:, :, 1], axis=1).reshape(-1, 1)
        else:
            X_cycle = X_test[:, 1].reshape(-1, 1)
            
        return self.model.predict(X_cycle)

# 2. Random Forest Implementation
class RandomForestRUL:
    """Random Forest for RUL prediction with 100 trees"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train):
        """
        Fit the Random Forest model.
        
        Args:
            X_train: Training features 
            y_train: Target RUL values
        """
        # If X is in sequence format, flatten it to 2D
        if len(X_train.shape) == 3:  # (batch, seq_len, features)
            X_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten to (batch, seq_len*features)
        else:
            X_flat = X_train
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Fit model
        self.model.fit(X_scaled, y_train)
        
        return self
    
    def predict(self, X_test):
        """Predict RUL values"""
        # If X is in sequence format, flatten it
        if len(X_test.shape) == 3:
            X_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_flat = X_test
            
        # Scale features
        X_scaled = self.scaler.transform(X_flat)
        
        return self.model.predict(X_scaled)

# 3. GRU Network Implementation
def build_gru_model(input_shape, units=64, dropout_rate=0.2):
    """
    Build a GRU network with 64 units.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        units: Number of GRU units
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(units, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.GRU(units//2),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for RUL prediction
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# 4. CMAPSS Competition Winner Implementation (simplified version based on literature)
def build_cmapss_winner_model(input_shape):
    """
    Implementation based on one of the top performers in the PHM08 CMAPSS competition.
    This is a simplified version of the approach described in the literature.
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        Compiled Keras model
    """
    # Based on the paper "RNN for Remaining Useful Life Estimation" 
    # by Heimes, which was among the top performers
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(20),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),  # Lower learning rate as in paper
        loss='mse',
        metrics=['mae']
    )
    
    return model

def benchmark_models(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Benchmark all four baseline models
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of epochs for neural network models
        batch_size: Batch size for neural network models
        
    Returns:
        Dictionary with results for each model
    """
    results = {}
    
    # 1. Linear Degradation Model
    print("\nTraining Linear Degradation Model...")
    linear_model = LinearDegradationModel()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    
    linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    
    results['Linear Degradation'] = {
        'rmse': linear_rmse,
        'mae': linear_mae,
        'model': linear_model
    }
    
    print(f"Linear Degradation Model - RMSE: {linear_rmse:.4f}, MAE: {linear_mae:.4f}")
    
    # 2. Random Forest with 100 trees
    print("\nTraining Random Forest Model (100 trees)...")
    rf_model = RandomForestRUL(n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    
    results['Random Forest'] = {
        'rmse': rf_rmse,
        'mae': rf_mae,
        'model': rf_model
    }
    
    print(f"Random Forest Model - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
    
    # Early stopping callback for neural network models
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # 3. GRU Network with 64 units
    print("\nTraining GRU Network (64 units)...")
    gru_model = build_gru_model(X_train.shape[1:], units=64)
    
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    y_pred_gru = gru_model.predict(X_test)
    
    gru_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gru))
    gru_mae = mean_absolute_error(y_test, y_pred_gru)
    
    results['GRU Network'] = {
        'rmse': gru_rmse,
        'mae': gru_mae,
        'model': gru_model,
        'history': gru_history.history
    }
    
    print(f"GRU Network - RMSE: {gru_rmse:.4f}, MAE: {gru_mae:.4f}")
    
    # 4. CMAPSS Competition Winner
    print("\nTraining CMAPSS Competition Winner Model...")
    cmapss_model = build_cmapss_winner_model(X_train.shape[1:])
    
    cmapss_history = cmapss_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    y_pred_cmapss = cmapss_model.predict(X_test)
    
    cmapss_rmse = np.sqrt(mean_squared_error(y_test, y_pred_cmapss))
    cmapss_mae = mean_absolute_error(y_test, y_pred_cmapss)
    
    results['CMAPSS Winner'] = {
        'rmse': cmapss_rmse,
        'mae': cmapss_mae,
        'model': cmapss_model,
        'history': cmapss_history.history
    }
    
    print(f"CMAPSS Winner Model - RMSE: {cmapss_rmse:.4f}, MAE: {cmapss_mae:.4f}")
    
    return results

def visualize_benchmark_results(benchmark_results, advanced_results=None):
    """
    Create visualizations comparing benchmark models and advanced models
    
    Args:
        benchmark_results: Dictionary with baseline model results
        advanced_results: Optional dictionary with advanced model results
    """
    # Prepare data for plotting
    models = list(benchmark_results.keys())
    rmse_values = [benchmark_results[m]['rmse'] for m in models]
    mae_values = [benchmark_results[m]['mae'] for m in models]
    
    # If advanced results provided, add them
    if advanced_results:
        for model_name, results in advanced_results.items():
            models.append(model_name)
            rmse_values.append(results['rmse'])
            mae_values.append(results['mae'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE comparison
    ax1.bar(models, rmse_values, color=sns.color_palette("viridis", len(models)))
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE Comparison')
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # MAE comparison
    ax2.bar(models, mae_values, color=sns.color_palette("viridis", len(models)))
    ax2.set_xlabel('Model')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE Comparison')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()
    
    # Create detailed comparison table
    results_table = {
        'Model': models,
        'RMSE': rmse_values,
        'MAE': mae_values,
        '% Improvement over Linear': [(1 - rmse/rmse_values[0])*100 for rmse in rmse_values]
    }
    
    results_df = pd.DataFrame(results_table)
    print("\nDetailed Benchmark Results:")
    print(results_df.to_string(index=False))
    
    return fig, results_df

def run_benchmarking():
    """Run the complete benchmarking process"""
    # Define paths
    data_path = os.path.join('data', 'FD001.txt')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    processed_data = create_hierarchical_feature_pipeline('FD001', data_path)
    
    # Split into train/test by unit
    units = processed_data['unit'].unique()
    np.random.seed(42)
    train_units = np.random.choice(units, size=int(0.8 * len(units)), replace=False)
    test_units = np.array([u for u in units if u not in train_units])
    
    train_data = processed_data[processed_data['unit'].isin(train_units)]
    test_data = processed_data[processed_data['unit'].isin(test_units)]
    
    # Create sequences
    X_train, y_train = prepare_tcn_sequences(train_data)
    X_test, y_test = prepare_tcn_sequences(test_data)
    
    # Run benchmarking
    benchmark_results = benchmark_models(X_train, y_train, X_test, y_test)
    
    # Create advanced model results (CNN-LSTM as example)
    print("\nTraining CNN-LSTM Model for comparison...")
    cnn_lstm_model = build_cnn_lstm_model(X_train.shape[1:])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    cnn_lstm_history = cnn_lstm_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    y_pred_cnn_lstm = cnn_lstm_model.predict(X_test)
    cnn_lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_cnn_lstm))
    cnn_lstm_mae = mean_absolute_error(y_test, y_pred_cnn_lstm)
    
    advanced_results = {
        'CNN-LSTM': {
            'rmse': cnn_lstm_rmse,
            'mae': cnn_lstm_mae
        }
    }
    
    # Add PINN model
    print("\nTraining PINN Model for comparison...")
    physics_model = TurbofanPhysics()
    pinn_model = PINN(X_train.shape[1], physics_model)
    pinn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    
    pinn_history = pinn_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    y_pred_pinn = pinn_model.predict(X_test)
    pinn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_pinn))
    pinn_mae = mean_absolute_error(y_test, y_pred_pinn)
    
    advanced_results['PINN'] = {
        'rmse': pinn_rmse,
        'mae': pinn_mae
    }
    
    # Visualize results
    fig, results_df = visualize_benchmark_results(benchmark_results, advanced_results)
    
    # Save results to CSV
    results_df.to_csv('benchmark_results.csv', index=False)
    
    return benchmark_results, advanced_results, results_df

if __name__ == "__main__":
    benchmark_results, advanced_results, _ = run_benchmarking()
    print("\nBenchmarking complete.")