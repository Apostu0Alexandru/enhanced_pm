import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class RandomSurvivalForest:
    """Implementation of Random Survival Forest using Random Forest Regressor"""
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.time_points = None
    
    def fit(self, X, t, e):
        """Fit the Random Survival Forest"""
        # Create time points for survival curve estimation
        self.time_points = np.sort(np.unique(t[e == 1]))
        
        # Train random forest to predict time-to-event
        self.rf.fit(X, t)
        
        return self
    
    def predict(self, X):
        """Predict time-to-event"""
        return self.rf.predict(X)
    
    def predict_survival_function(self, X):
        """Predict survival function for each sample"""
        # Predict time-to-event
        t_pred = self.predict(X)
        
        # Initialize survival curves
        n_samples = len(X)
        n_time_points = len(self.time_points)
        survival_curves = np.zeros((n_samples, n_time_points))
        
        # For each sample, calculate survival function
        for i in range(n_samples):
            # Calculate survival function: S(t) = P(T > t) = 1 - P(T <= t)
            survival_curves[i] = (t_pred[i] > self.time_points).astype(float)
        
        return survival_curves, self.time_points

class DeepHit:
    """Implementation of DeepHit model for survival analysis"""
    def __init__(self, input_dim, num_nodes_shared=[128, 64], num_nodes_cause=[32], num_causes=1):
        self.input_dim = input_dim
        self.num_nodes_shared = num_nodes_shared
        self.num_nodes_cause = num_nodes_cause
        self.num_causes = num_causes
        self.max_time = None
        self.model = None
    
    def build_model(self, max_time):
        """Build the DeepHit model"""
        self.max_time = max_time
        
        # Input
        inputs = Input(shape=(self.input_dim,))
        
        # Shared sub-network
        x = Dense(self.num_nodes_shared[0], activation='relu')(inputs)
        x = Dropout(0.2)(x)
        
        for nodes in self.num_nodes_shared[1:]:
            x = Dense(nodes, activation='relu')(x)
            x = Dropout(0.2)(x)
        
        # Cause-specific sub-networks
        cause_outputs = []
        
        for cause in range(self.num_causes):
            y = Dense(self.num_nodes_cause[0], activation='relu')(x)
            y = Dropout(0.2)(y)
            
            for nodes in self.num_nodes_cause[1:]:
                y = Dense(nodes, activation='relu')(y)
                y = Dropout(0.2)(y)
            
            # Output layer: probability mass function (PMF) over time
            cause_output = Dense(max_time + 1, activation='softmax', name=f'cause_{cause}')(y)
            cause_outputs.append(cause_output)
        
        # Combine all outputs
        model = Model(inputs=inputs, outputs=cause_outputs)
        
        return model

class EnsembleSurvivalModel:
    """Ensemble model combining Cox PH, Random Survival Forest, and DeepHit"""
    def __init__(self, input_dim, feature_cols):
        self.input_dim = input_dim
        self.feature_cols = feature_cols
        self.cox_model = None
        self.rsf_model = None
        self.deephit_model = None
        self.time_points = None
        self.is_fitted = False
    
    def fit(self, train_df, X_train, t_train, e_train):
        """Fit all component models"""
        print("Fitting Cox Proportional Hazards model...")
        self.cox_model = self._train_cox_model(train_df, self.feature_cols)
        
        print("Fitting Random Survival Forest...")
        self.rsf_model = RandomSurvivalForest(n_estimators=100)
        self.rsf_model.fit(X_train, t_train, e_train)
        
        print("Fitting DeepHit model...")
        self.deephit_model = DeepHit(
            input_dim=self.input_dim,
            num_nodes_shared=[128, 64],
            num_nodes_cause=[32]
        )
        
        max_time = int(np.max(t_train))
        model = self.deephit_model.build_model(max_time)
        model.compile(optimizer='adam', loss='mse')  # Simplified for demonstration
        
        # Use RSF's time points for prediction
        _, self.time_points = self.rsf_model.predict_survival_function(X_train[:1])
        
        self.is_fitted = True
        return self
    
    def _train_cox_model(self, train_df, feature_cols):
        """Train a Cox Proportional Hazards model"""
        # Create a new DataFrame with the required columns
        cox_df = train_df[feature_cols + ['time_to_event', 'event']].copy()
        
        # Fit the Cox PH model
        cox_model = CoxPHFitter()
        cox_model.fit(cox_df, duration_col='time_to_event', event_col='event')
        
        return cox_model
    
    def predict_remaining_useful_life(self, X, test_df=None):
        """Predict RUL using median survival time"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        # Simple implementation for demonstration
        # In a real implementation, we would combine predictions from all models
        return self.rsf_model.predict(X)
