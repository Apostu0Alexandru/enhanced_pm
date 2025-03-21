import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path, sequence_length=100, test_size=0.2):
    """Preprocess the turbofan engine dataset for CNN-LSTM model"""
    # Load dataset
    column_names = ['unit', 'cycle'] + [f'op{i}' for i in range(1, 4)] + \
                   [f'sensor{i}' for i in range(1, 22)]
    
    df = pd.read_csv(file_path, sep=' ', header=None)
    df = df.iloc[:, :-2]  # Drop last two columns (NaN)
    df.columns = column_names
    
    # Calculate RUL
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycles, on=['unit'], how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    
    # Normalize operational settings and sensor data
    sensor_cols = [f'sensor{i}' for i in range(1, 22)]
    op_cols = [f'op{i}' for i in range(1, 4)]
    feature_cols = op_cols + sensor_cols
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Create sequences for each unit
    units = df['unit'].unique()
    X, y = [], []
    
    for unit in units:
        unit_data = df[df['unit'] == unit]
        if len(unit_data) < sequence_length:
            continue
            
        for i in range(len(unit_data) - sequence_length + 1):
            X.append(unit_data[feature_cols].iloc[i:i+sequence_length].values)
            y.append(unit_data['RUL'].iloc[i+sequence_length-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def build_cnn_lstm_model(input_shape, filters=64, kernel_size=3, lstm_units=100, dropout_rate=0.2):
    """Build CNN-LSTM hybrid model for RUL prediction"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=filters*2, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers for sequence learning
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units//2)(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=100, patience=10):
    """Train the CNN-LSTM model with early stopping"""
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint = ModelCheckpoint('cnn_lstm_model.h5', monitor='val_loss', save_best_only=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    return model, history

def main(file_path, sequence_length=100):
    """Run the complete CNN-LSTM pipeline"""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(file_path, sequence_length)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape)
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean(np.square(y_test - y_pred.flatten())))
    print(f"Test RMSE: {rmse:.4f}")
    
    return model, history, rmse

if __name__ == "__main__":
    # Example: model, history, rmse = main("data/FD001.txt")
    pass
