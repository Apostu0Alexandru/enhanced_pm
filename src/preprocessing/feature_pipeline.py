import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def process_multicondition_data(file_path, n_clusters=6):
    """Process datasets with multiple operating conditions (FD002/FD004)"""
    # Load dataset
    column_names = ['unit', 'cycle'] + [f'op{i}' for i in range(1, 4)] + \
                   [f'sensor{i}' for i in range(1, 22)]
    data = pd.read_csv(file_path, sep=' ', header=None)
    data = data.drop(columns=[26, 27]) # Drop last two columns (NaN)
    data.columns = column_names
    
    # Identify operational settings
    op_settings = data[['op1', 'op2', 'op3']]
    
    # Cluster operational regimes
    scaler = StandardScaler()
    op_scaled = scaler.fit_transform(op_settings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['op_regime'] = kmeans.fit_predict(op_scaled)
    
    # Initialize dataframe for engineered features
    units = data['unit'].unique()
    feature_data = []
    
    # Process each unit's data
    for unit in units:
        unit_data = data[data['unit'] == unit].sort_values('cycle')
        
        # Extract regime-specific features
        for regime in range(n_clusters):
            regime_data = unit_data[unit_data['op_regime'] == regime]
            
            if len(regime_data) > 0:
                # 1. Rolling RMS for vibration sensors (columns 6-8)
                vibration_cols = ['sensor6', 'sensor7', 'sensor8']
                for col in vibration_cols:
                    if len(regime_data) >= 5:  # Need enough data for rolling window
                        regime_data[f'{col}_rms'] = regime_data[col].rolling(window=5, min_periods=1).apply(
                            lambda x: np.sqrt(np.mean(np.square(x))))
                    else:
                        regime_data[f'{col}_rms'] = regime_data[col]
                
                # 2. EWMA for temperature sensors (columns 12-14)
                temp_cols = ['sensor12', 'sensor13', 'sensor14']
                for col in temp_cols:
                    regime_data[f'{col}_ewma'] = regime_data[col].ewm(span=5).mean()
                
                # 3. Degradation rate for pressure sensors (columns 9-11)
                pressure_cols = ['sensor9', 'sensor10', 'sensor11']
                for col in pressure_cols:
                    if len(regime_data) > 1:
                        regime_data[f'{col}_derivative'] = regime_data[col].diff() / regime_data['cycle'].diff()
                        # Fill first value
                        regime_data[f'{col}_derivative'] = regime_data[f'{col}_derivative'].fillna(0)
                    else:
                        regime_data[f'{col}_derivative'] = 0
                
                feature_data.append(regime_data)
    
    # Combine all processed data
    processed_data = pd.concat(feature_data, ignore_index=True)
    return processed_data, kmeans.cluster_centers_

def process_singlecondition_data(file_path):
    """Process datasets with single operating condition (FD001/FD003)"""
    # Load dataset
    column_names = ['unit', 'cycle'] + [f'op{i}' for i in range(1, 4)] + \
                   [f'sensor{i}' for i in range(1, 22)]
    data = pd.read_csv(file_path, sep=' ', header=None)
    data = data.drop(columns=[26, 27]) # Drop last two columns (NaN)
    data.columns = column_names
    
    # Initialize dataframe for engineered features
    units = data['unit'].unique()
    processed_units = []
    
    # Process each unit to prepare sequence data for TCN
    for unit in units:
        unit_data = data[data['unit'] == unit].sort_values('cycle')
        
        # Calculate time-domain features
        # 1. Statistical features (mean, std, min, max)
        for i in range(1, 22):
            col = f'sensor{i}'
            # Apply rolling window stats (10 cycles)
            if len(unit_data) >= 10:
                unit_data[f'{col}_rolling_mean'] = unit_data[col].rolling(window=10, min_periods=1).mean()
                unit_data[f'{col}_rolling_std'] = unit_data[col].rolling(window=10, min_periods=1).std().fillna(0)
            else:
                unit_data[f'{col}_rolling_mean'] = unit_data[col]
                unit_data[f'{col}_rolling_std'] = 0
        
        # 2. Normalization by first cycle (for degradation patterns)
        first_cycle = unit_data.iloc[0]
        for i in range(1, 22):
            col = f'sensor{i}'
            if first_cycle[col] != 0:
                unit_data[f'{col}_norm'] = unit_data[col] / first_cycle[col]
            else:
                unit_data[f'{col}_norm'] = unit_data[col]
        
        processed_units.append(unit_data)
    
    # Combine all processed data
    processed_data = pd.concat(processed_units, ignore_index=True)
    return processed_data

def create_hierarchical_feature_pipeline(dataset_type, file_path):
    """
    Create a hierarchical feature engineering pipeline based on dataset type
    
    Parameters:
    -----------
    dataset_type: str
        One of 'FD001', 'FD002', 'FD003', 'FD004'
    file_path: str
        Path to the dataset file
        
    Returns:
    --------
    processed_data: DataFrame
        Data with engineered features
    """
    if dataset_type in ['FD002', 'FD004']:
        # Multi-condition datasets
        processed_data, cluster_centers = process_multicondition_data(file_path)
        print(f"Processed multi-condition data. Identified {len(cluster_centers)} operational regimes.")
    elif dataset_type in ['FD001', 'FD003']:
        # Single-condition datasets
        processed_data = process_singlecondition_data(file_path)
        print(f"Processed single-condition data. Prepared for TCN input.")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return processed_data

def prepare_tcn_sequences(processed_data, sequence_length=100, prediction_horizon=1):
    """
    Prepare sequences for Temporal Convolutional Network
    
    Parameters:
    -----------
    processed_data: DataFrame
        Processed data from single condition datasets
    sequence_length: int
        Length of input sequence (e.g., 100 cycles)
    prediction_horizon: int
        How many cycles ahead to predict
        
    Returns:
    --------
    X: numpy array
        Sequence input data for TCN
    y: numpy array
        Target RUL values
    """
    units = processed_data['unit'].unique()
    X, y = [], []
    
    # Select relevant features for TCN
    feature_cols = [col for col in processed_data.columns 
                   if ('sensor' in col and col != 'sensor') or ('op' in col and col != 'op')]
    
    for unit in units:
        unit_data = processed_data[processed_data['unit'] == unit].sort_values('cycle')
        
        # Calculate RUL (assuming we know the last cycle is the failure point)
        max_cycle = unit_data['cycle'].max()
        unit_data['RUL'] = max_cycle - unit_data['cycle']
        
        # Create sequences
        for i in range(len(unit_data) - sequence_length - prediction_horizon + 1):
            seq_x = unit_data.iloc[i:i+sequence_length][feature_cols].values
            seq_y = unit_data.iloc[i+sequence_length+prediction_horizon-1]['RUL']
            X.append(seq_x)
            y.append(seq_y)
    
    return np.array(X), np.array(y)
