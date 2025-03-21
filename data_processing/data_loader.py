import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class NASADataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None  # Initialize 'data' attribute
        self.columns = [
            'unit', 'cycle',                      # Identifiers (2)
            'setting_1', 'setting_2', 'setting_3', # Operational settings (3)
            'sensor_1', 'sensor_2', 'sensor_3',    # Sensors (21)
            'sensor_4', 'sensor_5', 'sensor_6',
            'sensor_7', 'sensor_8', 'sensor_9',
            'sensor_10', 'sensor_11', 'sensor_12',
            'sensor_13', 'sensor_14', 'sensor_15',
            'sensor_16', 'sensor_17', 'sensor_18',
            'sensor_19', 'sensor_20', 'sensor_21'
        ]

    def preprocess(self):
        import pandas as pd
            
        # Load data with correct separator
        self.data = pd.read_csv(self.data_path, sep='\s+', header=None, names=self.columns)
        
        # Remove non-informative sensors
        drop_cols = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
        self.data = self.data.drop(columns=drop_cols)
        
        # Calculate RUL
        self.data['RUL'] = self.data.groupby('unit')['cycle'].transform(lambda x: x.max() - x)
        
        # Return features and labels
        features = self.data.drop(columns=['unit', 'cycle', 'RUL']).values
        labels = self.data['RUL'].values
        return features, labels





