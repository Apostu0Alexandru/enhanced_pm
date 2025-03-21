# src/models/uncertainty_quantifier.py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

tfd = tfp.distributions
tfpl = tfp.layers

class BayesianRULPredictor:
    """
    Uncertainty Quantification Module for RUL prediction using Bayesian Neural Networks.
    This model provides probabilistic predictions with uncertainty quantification.
    """
    
    def __init__(self, use_aleatoric=True, use_epistemic=True):
        """
        Initialize the Bayesian RUL predictor.
        
        Args:
            use_aleatoric: Whether to model aleatoric uncertainty (data noise)
            use_epistemic: Whether to model epistemic uncertainty (model uncertainty)
        """
        self.model = None
        self.use_aleatoric = use_aleatoric
        self.use_epistemic = use_epistemic
        self.kl_weight = None
        
    def prior(self, kernel_size, bias_size, dtype=None):
        """
        Define prior distribution for Bayesian layers.
        
        Args:
            kernel_size: Size of kernel weights
            bias_size: Size of bias weights
            dtype: Data type
            
        Returns:
            Prior distribution
        """
        n = kernel_size + bias_size
        return tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(n, dtype=dtype), scale=0.1),
            reinterpreted_batch_ndims=1
        )
    
    def posterior(self, kernel_size, bias_size, dtype=None):
        """
        Define variational posterior distribution for Bayesian layers.
        
        Args:
            kernel_size: Size of kernel weights
            bias_size: Size of bias weights
            dtype: Data type
            
        Returns:
            Posterior distribution
        """
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tfp.distributions.Independent(
            tfp.distributions.Normal(
                loc=tf.Variable(tf.random.normal([n], stddev=0.01, dtype=dtype)),
                scale=tfp.util.TransformedVariable(
                    tf.ones(n, dtype=dtype), tfp.bijectors.Softplus()
                )
            ),
            reinterpreted_batch_ndims=1
        )
    
    def build_model(self, input_shape, hidden_units=[64, 32]):
        """
        Build the Bayesian Neural Network for RUL prediction.
        
        Args:
            input_shape: Shape of input data
            hidden_units: List of hidden layer sizes
            
        Returns:
            Compiled Keras model
        """
        # Determine number of samples for training
        self.kl_weight = 1.0 / 100  # Scale KL divergence by dataset size
        
        # Define input layer
        inputs = Input(shape=input_shape)
        
        # Hidden layers
        x = inputs
        
        for i, units in enumerate(hidden_units):
            if self.use_epistemic:
                # Bayesian layer with prior and posterior distributions
                x = tfpl.DenseVariational(
                    units=units,
                    make_prior_fn=self.prior,
                    make_posterior_fn=self.posterior,
                    kl_weight=self.kl_weight,
                    activation='relu'
                )(x)
            else:
                # Standard dense layer
                x = Dense(units, activation='relu')(x)
            
            x = Dropout(0.2)(x)
        
        # Output layer
        if self.use_aleatoric:
            # Predict mean and log variance for aleatoric uncertainty
            mean = Dense(1, name='mean')(x)
            log_var = Dense(1, name='log_var')(x)
            
            # Create probabilistic output
            def negative_log_likelihood(y, mu_log_var):
                mu = mu_log_var[:, 0:1]
                log_var = mu_log_var[:, 1:2]
                precision = tf.exp(-log_var)
                return 0.5 * (tf.math.log(2 * np.pi) + log_var + precision * (y - mu)**2)
            
            # Concatenate mean and log_var as model output
            outputs = Concatenate(name='output')([mean, log_var])
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile with custom loss function
            model.compile(
                optimizer='adam',
                loss=negative_log_likelihood
            )
        else:
            # Standard output for deterministic prediction
            if self.use_epistemic:
                # For epistemic uncertainty only, use normal distribution output
                outputs = tfpl.DenseVariational(
                    units=1,
                    make_prior_fn=self.prior,
                    make_posterior_fn=self.posterior,
                    kl_weight=self.kl_weight
                )(x)
            else:
                # Deterministic output (no uncertainty)
                outputs = Dense(1)(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile with MSE loss
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the Bayesian Neural Network.
        
        Args:
            X_train: Training features
            y_train: Training targets (RUL values)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            input_shape = X_train.shape[1:]
            self.build_model(input_shape)
        
        # Early stopping and learning rate reduction callbacks
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
        
        # For aleatoric uncertainty, prepare target format
        if self.use_aleatoric:
            # Reshape y to match the expected output (dummy log_var)
            y_train_formatted = np.hstack([y_train.reshape(-1, 1), np.zeros_like(y_train.reshape(-1, 1))])
        else:
            y_train_formatted = y_train
        
        # Train model
        history = self.model.fit(
            X_train, y_train_formatted,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_with_uncertainty(self, X_test, n_samples=100):
        """
        Make predictions with uncertainty quantification.
        
        Args:
            X_test: Test features
            n_samples: Number of Monte Carlo samples for epistemic uncertainty
            
        Returns:
            mean: Mean predictions
            aleatoric: Aleatoric uncertainty (data noise)
            epistemic: Epistemic uncertainty (model uncertainty)
            total: Total uncertainty (combined)
        """
        if self.use_epistemic and self.use_aleatoric:
            # Both uncertainties
            predictions = [self.model.predict(X_test) for _ in range(n_samples)]
            predictions = np.array(predictions)
            
            # Extract means and log variances
            means = predictions[:, :, 0]  # Shape: [n_samples, n_instances]
            log_vars = predictions[:, :, 1]  # Shape: [n_samples, n_instances]
            
            # Calculate uncertainties
            mean_prediction = np.mean(means, axis=0)
            aleatoric = np.mean(np.exp(log_vars), axis=0)
            epistemic = np.var(means, axis=0)
            total = aleatoric + epistemic
            
        elif self.use_epistemic:
            # Only epistemic uncertainty
            predictions = [self.model.predict(X_test) for _ in range(n_samples)]
            predictions = np.array(predictions).squeeze()
            
            # Calculate statistics
            mean_prediction = np.mean(predictions, axis=0)
            epistemic = np.var(predictions, axis=0)
            aleatoric = np.zeros_like(epistemic)
            total = epistemic
            
        elif self.use_aleatoric:
            # Only aleatoric uncertainty
            prediction = self.model.predict(X_test)
            mean_prediction = prediction[:, 0]
            aleatoric = np.exp(prediction[:, 1])
            epistemic = np.zeros_like(aleatoric)
            total = aleatoric
            
        else:
            # No uncertainty (deterministic)
            mean_prediction = self.model.predict(X_test).flatten()
            aleatoric = np.zeros_like(mean_prediction)
            epistemic = np.zeros_like(mean_prediction)
            total = np.zeros_like(mean_prediction)
        
        return mean_prediction, aleatoric, epistemic, total
    
    def visualize_uncertainty(self, X_test, y_test, cycle_values=None, save_path=None):
        """
        Visualize predictions with uncertainty.
        
        Args:
            X_test: Test features
            y_test: Test targets (ground truth)
            cycle_values: Cycle values for x-axis (optional)
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        # Get predictions with uncertainty
        mean, aleatoric, epistemic, total = self.predict_with_uncertainty(X_test)
        
        # If cycle values not provided, use indices
        if cycle_values is None:
            cycle_values = np.arange(len(y_test))
        
        # Calculate 95% confidence intervals
        lower_bound = mean - 1.96 * np.sqrt(total)
        upper_bound = mean + 1.96 * np.sqrt(total)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Predictions with uncertainty
        ax1.plot(cycle_values, y_test, 'ko', label='Ground Truth', alpha=0.5)
        ax1.plot(cycle_values, mean, 'b-', label='Predicted RUL')
        ax1.fill_between(
            cycle_values, lower_bound, upper_bound,
            color='b', alpha=0.2, label='95% Confidence Interval'
        )
        ax1.set_ylabel('Remaining Useful Life (cycles)')
        ax1.set_title('RUL Predictions with Uncertainty')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Uncertainty decomposition
        ax2.plot(cycle_values, total, 'r-', label='Total Uncertainty')
        
        if self.use_aleatoric and self.use_epistemic:
            ax2.plot(cycle_values, aleatoric, 'g-', label='Aleatoric (Data Noise)')
            ax2.plot(cycle_values, epistemic, 'b-', label='Epistemic (Model Uncertainty)')
        elif self.use_aleatoric:
            ax2.plot(cycle_values, aleatoric, 'g-', label='Aleatoric (Data Noise)')
        elif self.use_epistemic:
            ax2.plot(cycle_values, epistemic, 'b-', label='Epistemic (Model Uncertainty)')
        
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Uncertainty (variance)')
        ax2.set_title('Uncertainty Decomposition')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def calibration_plot(self, X_test, y_test, n_bins=10, save_path=None):
        """
        Create a calibration plot to assess predictive uncertainty quality.
        
        Args:
            X_test: Test features
            y_test: Test targets (ground truth)
            n_bins: Number of bins for calibration curve
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        # Get predictions with uncertainty
        mean, _, _, total = self.predict_with_uncertainty(X_test)
        std_dev = np.sqrt(total)
        
        # Compute standardized residuals
        z_scores = (y_test - mean) / std_dev
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Calibration curve (reliability diagram)
        # For a well-calibrated model, the proportion of samples within
        # a given confidence interval should match the theoretical proportion
        quantiles = np.linspace(0, 1, n_bins+1)[1:]
        
        observed_proportions = []
        for q in quantiles:
            # Convert quantile to standard deviation units for normal distribution
            from scipy import stats
            z_value = stats.norm.ppf((1 + q) / 2)
            
            # Calculate proportion of samples within this interval
            within_interval = np.abs(z_scores) <= z_value
            observed_proportions.append(np.mean(within_interval))
        
        # Plot calibration curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Ideal')
        ax1.plot(quantiles, observed_proportions, 'bo-', label='Model')
        ax1.set_xlabel('Expected Proportion')
        ax1.set_ylabel('Observed Proportion')
        ax1.set_title('Calibration Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Histogram of standardized residuals
        ax2.hist(z_scores, bins=20, density=True, alpha=0.7)
        
        # Add normal distribution curve for comparison
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        ax2.plot(x, y, 'r-', label='Standard Normal')
        
        ax2.set_xlabel('Standardized Residuals (z-scores)')
        ax2.set_ylabel('Density')
        ax2.set_title('Histogram of Standardized Residuals')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
