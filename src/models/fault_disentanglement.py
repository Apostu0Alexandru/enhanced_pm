# src/models/fault_disentanglement.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns

class FaultDisentanglementAutoencoder:
    """
    Fault Disentanglement Autoencoder that separates normal system state from fault patterns.
    
    This model follows the objective:
    min_θ E[||x - D(E_s(x) + E_f(x))||^2 + λ||E_f(x)||_1]
    
    Where:
    - E_s encodes system state (normal operation)
    - E_f isolates faults (abnormal patterns)
    - D is a decoder that reconstructs the input
    - λ is a regularization parameter for the sparsity of fault encoding
    """
    
    def __init__(self, system_latent_dim=10, fault_latent_dim=5, lambda_reg=0.1):
        """
        Initialize the autoencoder.
        
        Args:
            system_latent_dim: Dimensionality of the system state encoding
            fault_latent_dim: Dimensionality of the fault encoding
            lambda_reg: Regularization parameter for fault encoding sparsity
        """
        self.system_latent_dim = system_latent_dim
        self.fault_latent_dim = fault_latent_dim
        self.lambda_reg = lambda_reg
        self.model = None
        self.encoder_system = None
        self.encoder_fault = None
        self.decoder = None
        self.scaler = StandardScaler()
    
    def build_encoder(self, input_dim, latent_dim, name, l1_reg=None):
        """
        Build an encoder network.
        
        Args:
            input_dim: Input dimensionality
            latent_dim: Latent dimensionality
            name: Name prefix for the encoder
            l1_reg: L1 regularization parameter (optional)
            
        Returns:
            Encoder model
        """
        # Input layer
        inputs = Input(shape=(input_dim,), name=f'{name}_input')
        
        # Hidden layers
        x = Dense(64, activation='relu', name=f'{name}_dense1')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu', name=f'{name}_dense2')(x)
        
        # Output layer
        if l1_reg is not None:
            # Add L1 regularization to enforce sparsity
            x = Dense(
                latent_dim, 
                activation='linear',
                activity_regularizer=tf.keras.regularizers.l1(l1_reg),
                name=f'{name}_latent'
            )(x)
        else:
            x = Dense(latent_dim, activation='linear', name=f'{name}_latent')(x)
        
        # Create model
        encoder = Model(inputs, x, name=name)
        return encoder
    
    def build_decoder(self, latent_dim, output_dim):
        """
        Build a decoder network.
        
        Args:
            latent_dim: Latent dimensionality (system + fault)
            output_dim: Output dimensionality
            
        Returns:
            Decoder model
        """
        # Input layer
        inputs = Input(shape=(latent_dim,), name='decoder_input')
        
        # Hidden layers
        x = Dense(32, activation='relu', name='decoder_dense1')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu', name='decoder_dense2')(x)
        
        # Output layer
        outputs = Dense(output_dim, activation='linear', name='decoder_output')(x)
        
        # Create model
        decoder = Model(inputs, outputs, name='decoder')
        return decoder
    
    def build_model(self, input_dim):
        """
        Build the complete disentanglement autoencoder.
        
        Args:
            input_dim: Input dimensionality
            
        Returns:
            Complete autoencoder model
        """
        # Build system encoder (no L1 regularization)
        self.encoder_system = self.build_encoder(
            input_dim, self.system_latent_dim, 'encoder_system'
        )
        
        # Build fault encoder (with L1 regularization for sparsity)
        self.encoder_fault = self.build_encoder(
            input_dim, self.fault_latent_dim, 'encoder_fault', l1_reg=self.lambda_reg
        )
        
        # Build decoder
        self.decoder = self.build_decoder(
            self.system_latent_dim + self.fault_latent_dim, input_dim
        )
        
        # Create combined model
        inputs = Input(shape=(input_dim,), name='input')
        
        # Get encodings
        system_encoding = self.encoder_system(inputs)
        fault_encoding = self.encoder_fault(inputs)
        
        # Concatenate encodings
        combined_encoding = Concatenate()([system_encoding, fault_encoding])
        
        # Decode
        reconstruction = self.decoder(combined_encoding)
        
        # Create model
        model = Model(inputs, reconstruction, name='disentanglement_autoencoder')
        
        # Get fault encoding for additional loss term
        model.fault_encoding = fault_encoding
        
        # Compile with custom loss function
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'  # Base reconstruction loss
            # L1 regularization is applied in the fault encoder directly
        )
        
        self.model = model
        return model
    
    def custom_loss(self, y_true, y_pred):
        """
        Custom loss function with reconstruction and L1 regularization.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            
        Returns:
            Loss value
        """
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # L1 regularization on fault encoding
        l1_loss = tf.reduce_mean(tf.abs(self.model.fault_encoding))
        
        # Combined loss
        return reconstruction_loss + self.lambda_reg * l1_loss
    
    def preprocess_data(self, df, is_training=True):
        """
        Preprocess the data by extracting sensor readings.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            is_training: Whether this is training data (for fitting scaler)
            
        Returns:
            Numpy array with preprocessed data
        """
        # Extract sensor columns
        sensor_cols = [col for col in df.columns if 'sensor' in col]
        X = df[sensor_cols].values
        
        # Scale the data
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled
    
    def fit(self, df, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the disentanglement autoencoder.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Preprocess data
        X = self.preprocess_data(df, is_training=True)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X.shape[1])
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, X,  # Autoencoder reconstructs its input
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def encode(self, df):
        """
        Encode data into system state and fault components.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            
        Returns:
            system_encoding: System state encoding
            fault_encoding: Fault encoding
        """
        # Preprocess data
        X = self.preprocess_data(df, is_training=False)
        
        # Encode
        system_encoding = self.encoder_system.predict(X)
        fault_encoding = self.encoder_fault.predict(X)
        
        return system_encoding, fault_encoding
    
    def reconstruct(self, df):
        """
        Reconstruct data from encodings.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            
        Returns:
            original: Original preprocessed data
            reconstructed: Reconstructed data
            reconstruction_error: Per-sample reconstruction error
        """
        # Preprocess data
        X = self.preprocess_data(df, is_training=False)
        
        # Reconstruct
        reconstructed = self.model.predict(X)
        
        # Calculate reconstruction error
        reconstruction_error = np.mean(np.square(X - reconstructed), axis=1)
        
        return X, reconstructed, reconstruction_error
    
    def visualize_fault_patterns(self, df, cycle_data=None, save_path=None):
        """
        Visualize extracted fault patterns and their evolution over time.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            cycle_data: DataFrame with cycle information (optional)
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        # Encode data
        _, fault_encoding = self.encode(df)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: t-SNE visualization of fault space
        tsne = TSNE(n_components=2, random_state=42)
        fault_2d = tsne.fit_transform(fault_encoding)
        
        # If cycle data is provided, use it for coloring
        if cycle_data is not None:
            # Normalize cycle to percentage of max cycles
            max_cycles = cycle_data.groupby('unit')['cycle'].max().reset_index()
            cycle_data = cycle_data.merge(max_cycles, on='unit', suffixes=('', '_max'))
            cycle_data['cycle_pct'] = cycle_data['cycle'] / cycle_data['cycle_max']
            
            scatter = axes[0, 0].scatter(
                fault_2d[:, 0], fault_2d[:, 1],
                c=cycle_data['cycle_pct'],
                cmap='viridis',
                alpha=0.7
            )
            axes[0, 0].set_title('t-SNE Visualization of Fault Space (colored by cycle %)')
            plt.colorbar(scatter, ax=axes[0, 0], label='Cycle %')
        else:
            axes[0, 0].scatter(fault_2d[:, 0], fault_2d[:, 1], alpha=0.7)
            axes[0, 0].set_title('t-SNE Visualization of Fault Space')
        
        axes[0, 0].set_xlabel('t-SNE Component 1')
        axes[0, 0].set_ylabel('t-SNE Component 2')
        
        # Plot 2: Fault encoding sparsity
        fault_avg = np.mean(np.abs(fault_encoding), axis=0)
        axes[0, 1].bar(range(self.fault_latent_dim), fault_avg)
        axes[0, 1].set_title('Average Activation of Fault Dimensions')
        axes[0, 1].set_xlabel('Fault Dimension')
        axes[0, 1].set_ylabel('Average Activation')
        
        # Plot 3: Evolution of fault encoding over cycles
        if cycle_data is not None:
            # Select a few units for visualization
            units = cycle_data['unit'].unique()
            selected_units = np.random.choice(units, min(5, len(units)), replace=False)
            
            for unit in selected_units:
                mask = cycle_data['unit'] == unit
                unit_cycles = cycle_data.loc[mask, 'cycle'].values
                unit_fault = fault_encoding[mask]
                
                # Calculate fault severity (L1 norm of fault encoding)
                fault_severity = np.sum(np.abs(unit_fault), axis=1)
                
                axes[1, 0].plot(unit_cycles, fault_severity, label=f'Unit {unit}')
                
            axes[1, 0].set_title('Fault Severity Evolution Over Cycles')
            axes[1, 0].set_xlabel('Cycle')
            axes[1, 0].set_ylabel('Fault Severity (L1 norm)')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'Cycle data not provided', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[1, 0].transAxes)
        
        # Plot 4: Reconstruction error analysis
        _, _, reconstruction_error = self.reconstruct(df)
        
        if cycle_data is not None:
            # Scatter plot of reconstruction error vs. cycle percentage
            axes[1, 1].scatter(
                cycle_data['cycle_pct'],
                reconstruction_error,
                alpha=0.5
            )
            axes[1, 1].set_title('Reconstruction Error vs. Cycle Percentage')
            axes[1, 1].set_xlabel('Cycle %')
            axes[1, 1].set_ylabel('Reconstruction Error')
        else:
            # Histogram of reconstruction error
            axes[1, 1].hist(reconstruction_error, bins=30)
            axes[1, 1].set_title('Reconstruction Error Distribution')
            axes[1, 1].set_xlabel('Reconstruction Error')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_system_vs_fault(self, df, save_path=None):
        """
        Visualize the separation between system state and fault encodings.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        # Encode data
        system_encoding, fault_encoding = self.encode(df)
        
        # Calculate encoding norms
        system_norm = np.linalg.norm(system_encoding, axis=1)
        fault_norm = np.linalg.norm(fault_encoding, axis=1)
        
        # Get reconstruction error
        _, _, reconstruction_error = self.reconstruct(df)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: System vs. Fault norm scatter
        scatter = axes[0, 0].scatter(
            system_norm, fault_norm,
            c=reconstruction_error,
            cmap='viridis',
            alpha=0.7
        )
        axes[0, 0].set_title('System vs. Fault Encoding Norms')
        axes[0, 0].set_xlabel('System Encoding Norm')
        axes[0, 0].set_ylabel('Fault Encoding Norm')
        plt.colorbar(scatter, ax=axes[0, 0], label='Reconstruction Error')
        
        # Plot 2: System encoding distribution
        sns.heatmap(
            pd.DataFrame(system_encoding).corr(),
            annot=False,
            cmap='coolwarm',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('System Encoding Correlation Matrix')
        
        # Plot 3: Fault encoding distribution
        sns.heatmap(
            pd.DataFrame(fault_encoding).corr(),
            annot=False,
            cmap='coolwarm',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Fault Encoding Correlation Matrix')
        
        # Plot 4: Reconstruction quality
        # Randomly select a sample
        idx = np.random.randint(0, len(df))
        X, reconstructed, _ = self.reconstruct(df.iloc[idx:idx+1])
        
        # Get original and reconstructed values
        original = X[0]
        reconstructed = reconstructed[0]
        
        # Plot original vs. reconstructed
        axes[1, 1].plot(original, label='Original')
        axes[1, 1].plot(reconstructed, label='Reconstructed')
        axes[1, 1].set_title('Original vs. Reconstructed Sample')
        axes[1, 1].set_xlabel('Sensor Index')
        axes[1, 1].set_ylabel('Sensor Value')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
