import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def process_multicondition_data(file_path, n_clusters=6):
    """Process datasets with multiple operating conditions (FD002/FD004)"""
    # Load dataset
    data = pd.read_csv(file_path, sep=' ', header=None)
    
    # Safely drop columns 26 and 27 if they exist (NaN columns)
    if 26 in data.columns and 27 in data.columns:
        data = data.drop(columns=[26, 27])
    
    # Generate appropriate column names based on data size
    actual_col_names = ['unit', 'cycle']
    
    # Next 3 columns are operational settings (if available)
    op_cols = min(3, max(0, len(data.columns) - 2))
    if op_cols > 0:
        actual_col_names.extend([f'op{i}' for i in range(1, op_cols + 1)])
    
    # Remaining columns are sensor readings
    sensor_cols = max(0, len(data.columns) - 2 - op_cols)
    if sensor_cols > 0:
        actual_col_names.extend([f'sensor{i}' for i in range(1, sensor_cols + 1)])
    
    data.columns = actual_col_names
    
    # Convert all columns to numeric, filling NaN values
    for col in data.columns:
        if col == 'unit':
            # Fill NaN values before converting to integer
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int) 
        else:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    # Identify operational settings - use all available op columns
    op_columns = [col for col in data.columns if col.startswith('op')]
    if len(op_columns) > 0:
        op_settings = data[op_columns]
    else:
        # If no op columns available, use first few sensor columns as proxy
        sensor_columns = [col for col in data.columns if col.startswith('sensor')][:3]
        op_settings = data[sensor_columns[:min(3, len(sensor_columns))]]
    
    # Force all values to be numeric and handle NaN values
    op_settings = op_settings.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Cluster operational regimes
    scaler = StandardScaler()
    op_scaled = scaler.fit_transform(op_settings)
    
    # Ensure we use a reasonable number of clusters
    n_clusters = min(n_clusters, len(data), 2)  # At least 2 clusters but no more than data points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['op_regime'] = kmeans.fit_predict(op_scaled)
    
    # Initialize dataframe for engineered features
    units = data['unit'].unique()
    
    # Process each unit
    feature_data = []
    for unit in units:
        unit_data = data[data['unit'] == unit].sort_values('cycle')
        
        # Extract regime-specific features
        for regime in range(min(n_clusters, len(data))):
            regime_data = unit_data[unit_data['op_regime'] == regime]
            
            if len(regime_data) > 0:
                # Calculate regime statistics
                op_data = {
                    'unit': unit,
                    'op_regime': regime
                }
                
                # Add operational statistics
                sensor_cols = [col for col in regime_data.columns if col.startswith('sensor')]
                for col in sensor_cols:
                    op_data[f'{col}_mean'] = regime_data[col].mean()
                    op_data[f'{col}_std'] = regime_data[col].std() if len(regime_data) > 1 else 0
                
                feature_data.append(op_data)
    
    # Create dataframe with extracted features
    regime_features = pd.DataFrame(feature_data)
    
    # Return both the processed data and the cluster centers
    return data, kmeans.cluster_centers_

def process_singlecondition_data(file_path):
    """Process datasets with single operating condition (FD001/FD003)"""
    # Load dataset
    data = pd.read_csv(file_path, sep=' ', header=None)
    
    # Safely drop columns 26 and 27 if they exist (NaN columns)
    if 26 in data.columns and 27 in data.columns:
        data = data.drop(columns=[26, 27])
    
    # Dynamically generate column names based on actual number of columns
    # First two columns are always unit and cycle
    column_names = ['unit', 'cycle']
    
    # Next 3 columns are operational settings (if available)
    op_cols = min(3, max(0, len(data.columns) - 2))
    if op_cols > 0:
        column_names.extend([f'op{i}' for i in range(1, op_cols + 1)])
    
    # Remaining columns are sensor readings
    sensor_cols = max(0, len(data.columns) - 2 - op_cols)
    if sensor_cols > 0:
        column_names.extend([f'sensor{i}' for i in range(1, sensor_cols + 1)])
    
    # Ensure we have the right number of column names
    if len(column_names) != len(data.columns):
        raise ValueError(f"Column name mismatch: {len(column_names)} names for {len(data.columns)} columns")
    
    data.columns = column_names
    
    # Convert all columns to numeric, filling NaN values
    for col in data.columns:
        if col == 'unit':
            # Fill NaN values before converting to integer
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
        else:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    # Initialize dataframe for engineered features
    units = data['unit'].unique()
    processed_units = []
    
    # Process each unit to prepare sequence data for TCN
    for unit in units:
        unit_data = data[data['unit'] == unit].copy()
        
        # Calculate time-domain features
        # Make sure we process all sensor columns and ensure sensor2_rolling_mean is created
        sensor_columns = [col for col in unit_data.columns if col.startswith('sensor')]
        
        # 1. Statistical features (mean, std, min, max)
        for col in sensor_columns:
            # Apply rolling window stats (10 cycles)
            if len(unit_data) >= 10:
                unit_data[f'{col}_rolling_mean'] = unit_data[col].rolling(window=10, min_periods=1).mean()
                unit_data[f'{col}_rolling_std'] = unit_data[col].rolling(window=10, min_periods=1).std().fillna(0)
            else:
                unit_data[f'{col}_rolling_mean'] = unit_data[col]
                unit_data[f'{col}_rolling_std'] = 0
        
        # Also process operational setting columns for completeness
        op_columns = [col for col in unit_data.columns if col.startswith('op')]
        for col in op_columns:
            if len(unit_data) >= 10:
                unit_data[f'{col}_rolling_mean'] = unit_data[col].rolling(window=10, min_periods=1).mean()
                unit_data[f'{col}_rolling_std'] = unit_data[col].rolling(window=10, min_periods=1).std().fillna(0)
            else:
                unit_data[f'{col}_rolling_mean'] = unit_data[col]
                unit_data[f'{col}_rolling_std'] = 0
        
        # 2. Normalization by first cycle (for degradation patterns)
        first_cycle = unit_data.iloc[0]
        for col in sensor_columns + op_columns:
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
        data, cluster_centers = process_multicondition_data(file_path)
        print(f"Processed multi-condition data. Identified operational regimes.")
        return data
    elif dataset_type in ['FD001', 'FD003']:
        # Single-condition datasets
        processed_data = process_singlecondition_data(file_path)
        print(f"Processed single-condition data. Prepared for TCN input.")
        return processed_data
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

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
    # Make a copy to avoid modifying the original dataframe
    processed_data = processed_data.copy()
    
    # Ensure all columns are numeric
    for col in processed_data.columns:
        if col == 'unit':
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0).astype(int)
        elif col == 'cycle':
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0).astype(int)
        else:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
    
    # Calculate RUL for each unit
    units = processed_data['unit'].unique()
    
    # Create sequences
    sequences = []
    targets = []
    
    for unit in units:
        unit_data = processed_data[processed_data['unit'] == unit].copy().sort_values('cycle')
        
        # Calculate RUL (time until failure)
        # Make sure cycle is numeric and find the max cycle
        max_cycle = int(unit_data['cycle'].max())
        unit_data['RUL'] = max_cycle - unit_data['cycle']
        
        # Extract feature columns (exclude metadata)
        feature_cols = [col for col in unit_data.columns 
                       if col not in ['unit', 'cycle', 'RUL', 'op_regime']]
        
        # Create sequences for each possible starting point
        for i in range(len(unit_data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            seq = unit_data.iloc[i:i+sequence_length][feature_cols].values
            
            # Target RUL
            target = unit_data.iloc[i+sequence_length+prediction_horizon-1]['RUL']
            
            sequences.append(seq)
            targets.append(target)
    
    # Convert to arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    return X, y
