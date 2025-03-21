# src/models/regime_classifier.py
import numpy as np
import pandas as pd
# Import the patch before TensorFlow
import sys
import os
# Add project root to path to ensure the patch can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tensorflow_patch

# Try to import visualization libraries, but make them optional
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    # Create placeholder for testing environments where matplotlib is not available
    print("Warning: Matplotlib/Seaborn not available. Visualization functions will be limited.")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class OperationalRegimeClassifier:
    """
    Operational Regime Classifier using sensor 3-5 data (operational settings)
    for the CMAPSS Turbofan Engine Dataset.
    
    This model identifies distinct operational regimes that affect degradation patterns.
    """
    
    def __init__(self, n_regimes=6):
        """
        Initialize the classifier.
        
        Args:
            n_regimes: Number of operational regimes to identify
        """
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.regime_centers = None
        self.regime_counts = None
        
    def build_model(self, input_shape=(3,)):
        """
        Build the neural network classifier.
        
        Args:
            input_shape: Input shape for the model (default: 3 for sensors 3-5)
        """
        inputs = Input(shape=input_shape)
        
        # Deep network for regime classification
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer with softmax for regime probabilities
        outputs = Dense(self.n_regimes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with categorical crossentropy (multi-class classification)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, df, is_training=True):
        """
        Preprocess the data by extracting sensors 3-5 (operational settings).
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            is_training: Whether this is training data (for fitting scaler)
            
        Returns:
            Numpy array with preprocessed data
        """
        # Extract operational settings (sensors 3-5)
        # In CMAPSS, these are named 'op1', 'op2', 'op3'
        if 'op1' in df.columns:
            op_settings = df[['op1', 'op2', 'op3']].values
        else:
            # If columns are named differently, try to find them
            try:
                op_settings = df[['sensor3', 'sensor4', 'sensor5']].values
            except KeyError:
                raise ValueError("Could not find operational setting columns")
        
        # Scale the data
        if is_training:
            op_settings_scaled = self.scaler.fit_transform(op_settings)
        else:
            op_settings_scaled = self.scaler.transform(op_settings)
            
        return op_settings_scaled
    
    def generate_pseudolabels(self, X, method='kmeans'):
        """
        Generate pseudolabels for unsupervised regime identification.
        
        Args:
            X: Preprocessed operational settings data
            method: Method for generating labels ('kmeans' or 'gmm')
            
        Returns:
            Numpy array with regime labels
        """
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        
        if method == 'kmeans':
            # Use K-means clustering
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            self.regime_centers = kmeans.cluster_centers_
        elif method == 'gmm':
            # Use Gaussian Mixture Model for soft clustering
            gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
            labels = gmm.fit_predict(X)
            self.regime_centers = gmm.means_
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Count samples in each regime
        self.regime_counts = np.bincount(labels, minlength=self.n_regimes)
        
        # Convert to one-hot encoding for training
        one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=self.n_regimes)
        
        return one_hot_labels
    
    def fit(self, df, epochs=50, batch_size=32, validation_split=0.2, method='kmeans'):
        """
        Fit the regime classifier on CMAPSS data.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            method: Method for generating initial pseudolabels
            
        Returns:
            Training history
        """
        # Preprocess data
        X = self.preprocess_data(df, is_training=True)
        
        # Generate pseudolabels using clustering
        y = self.generate_pseudolabels(X, method=method)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_shape=(X.shape[1],))
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict_regime(self, df):
        """
        Predict operational regime for new data.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            
        Returns:
            Numpy array with regime predictions (one-hot encoded)
        """
        # Preprocess data
        X = self.preprocess_data(df, is_training=False)
        
        # Predict regimes
        regime_probs = self.model.predict(X)
        
        # Convert probabilities to hard assignments
        regime_labels = np.argmax(regime_probs, axis=1)
        
        return regime_labels, regime_probs
    
    def visualize_regimes(self, df, save_path=None):
        """
        Visualize the identified operational regimes.
        
        Args:
            df: Pandas DataFrame with CMAPSS data
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available. Cannot visualize regimes.")
            return None
        
        # Preprocess data
        X = self.preprocess_data(df, is_training=False)
        
        # Predict regimes
        regime_labels, _ = self.predict_regime(df)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: 3D scatter plot of operational settings colored by regime
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        scatter = ax1.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=regime_labels,
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        ax1.set_xlabel('Operational Setting 1')
        ax1.set_ylabel('Operational Setting 2')
        ax1.set_zlabel('Operational Setting 3')
        ax1.set_title('Operational Regimes in 3D Space')
        
        # Add legend
        legend1 = ax1.legend(*scatter.legend_elements(),
                            loc="upper right", title="Regimes")
        ax1.add_artist(legend1)
        
        # Plot 2: Regime distribution
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.bar(range(self.n_regimes), self.regime_counts)
        ax2.set_xlabel('Regime ID')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Distribution of Operational Regimes')
        ax2.set_xticks(range(self.n_regimes))
        
        # Plot 3: Pairplot of first two operational settings
        ax3 = fig.add_subplot(2, 2, 3)
        for regime in range(self.n_regimes):
            mask = (regime_labels == regime)
            ax3.scatter(
                X[mask, 0], X[mask, 1],
                label=f'Regime {regime}',
                alpha=0.7
            )
        ax3.set_xlabel('Operational Setting 1')
        ax3.set_ylabel('Operational Setting 2')
        ax3.set_title('Regime Separation (OS1 vs OS2)')
        ax3.legend()
        
        # Plot 4: Regime centers
        ax4 = fig.add_subplot(2, 2, 4)
        center_df = pd.DataFrame(
            self.regime_centers,
            columns=['OS1', 'OS2', 'OS3']
        )
        center_df['Regime'] = range(self.n_regimes)
        
        # Use radar chart to visualize regime centers
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
        
        # Radar chart helper functions
        def radar_factory(num_vars, frame='circle'):
            """Create a radar chart with `num_vars` axes."""
            # calculate evenly-spaced axis angles
            theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            
            class RadarAxes(plt.PolarAxes):
                name = 'radar'
                
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_theta_zero_location('N')
                
                def fill(self, *args, **kwargs):
                    """Override fill so that line is closed by default"""
                    closed = kwargs.pop('closed', True)
                    return super().fill(closed=closed, *args, **kwargs)
                
                def plot(self, *args, **kwargs):
                    """Override plot so that line is closed by default"""
                    lines = super().plot(*args, **kwargs)
                    for line in lines:
                        self._close_line(line)
                    return lines
                
                def _close_line(self, line):
                    x, y = line.get_data()
                    # FIXME: markers at x[0], y[0] get doubled-up
                    if x[0] != x[-1]:
                        x = np.concatenate((x, [x[0]]))
                        y = np.concatenate((y, [y[0]]))
                        line.set_data(x, y)
                
                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels)
                
            # Initialize axes instance
            fig = plt.figure(figsize=(9, 9))
            ax = RadarAxes(fig, [0.1, 0.1, 0.8, 0.8], 
                          projection='polar')
            return theta, ax
        
        # Set up radar chart
        labels = ['OS1', 'OS2', 'OS3']
        theta, ax = radar_factory(len(labels))
        
        # Draw one axis per variable + add labels
        plt.xticks(theta, labels)
        
        # Draw the regime profile for each regime
        for i, regime in enumerate(range(self.n_regimes)):
            values = self.regime_centers[regime]
            # Normalize values for radar chart
            values = (values - np.min(values)) / (np.max(values) - np.min(values))
            
            # Plot regime profile
            ax.plot(theta, values, label=f'Regime {regime}')
            ax.fill(theta, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Operational Regime Profiles')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_regime_distribution(self, save_path=None):
        """
        Plot the distribution of data points across regimes.
        
        Args:
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available. Cannot plot regime distribution.")
            return None
            
        # Count samples per regime
        if self.regime_counts is None:
            print("No regime counts available. Run fit() first.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot regime counts
        if MATPLOTLIB_AVAILABLE:
            sns.barplot(x=np.arange(self.n_regimes), y=self.regime_counts, ax=ax)
        
        # Add labels
        if MATPLOTLIB_AVAILABLE:
            ax.set_xlabel('Operational Regime')
            ax.set_ylabel('Number of Samples')
            ax.set_title('Distribution of Data Points Across Operational Regimes')
        
        # Add regime numbers as x-tick labels
        if MATPLOTLIB_AVAILABLE:
            ax.set_xticks(np.arange(self.n_regimes))
            ax.set_xticklabels([f'Regime {i}' for i in range(self.n_regimes)])
        
        # Add counts as text on top of bars
        if MATPLOTLIB_AVAILABLE:
            for i, count in enumerate(self.regime_counts):
                ax.text(i, count + 0.1 * max(self.regime_counts), 
                       str(count), ha='center')
        
        # Save if path provided
        if save_path and MATPLOTLIB_AVAILABLE:
            plt.tight_layout()
            plt.savefig(save_path)
            
        return fig

    def plot_training_history(self, history, save_path=None):
        """
        Plot the training history (loss and accuracy).
        
        Args:
            history: History object from model.fit()
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available. Cannot plot training history.")
            return None
            
        # Create figure with two subplots
        if MATPLOTLIB_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        if MATPLOTLIB_AVAILABLE:
            ax1.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                ax1.plot(history.history['val_loss'], label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
        
        # Plot accuracy
        if MATPLOTLIB_AVAILABLE:
            ax2.plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
        
        # Save if path provided
        if save_path and MATPLOTLIB_AVAILABLE:
            plt.tight_layout()
            plt.savefig(save_path)
            
        return fig
