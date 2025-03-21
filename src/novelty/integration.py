# src/novelty/integration.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.regime_classifier import OperationalRegimeClassifier
from src.models.uncertainty_quantifier import BayesianRULPredictor
from src.models.fault_disentanglement import FaultDisentanglementAutoencoder
from src.preprocessing.feature_pipeline import create_hierarchical_feature_pipeline

def demonstrate_novel_components(data_path='data/FD002.txt'):
    """
    Demonstrate all three novel components on CMAPSS data.
    
    Args:
        data_path: Path to CMAPSS data file
    """
    # Create output directory for figures
    os.makedirs('docs/figures', exist_ok=True)
    
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    data = create_hierarchical_feature_pipeline('FD002', data_path)
    
    # 2. Operational Regime Classifier
    print("\n1. DEMONSTRATING OPERATIONAL REGIME CLASSIFIER")
    regime_classifier = OperationalRegimeClassifier(n_regimes=6)
    regime_classifier.fit(data, epochs=50, batch_size=32)
    
    # Visualize regimes
    regime_fig = regime_classifier.visualize_regimes(data)
    regime_fig.savefig('docs/figures/operational_regimes.png', dpi=300, bbox_inches='tight')
    print("Regime classification complete. Visualization saved to 'docs/figures/operational_regimes.png'")
    
    # 3. Prepare data for BNN and autoencoder
    # Split into train/test
    units = data['unit'].unique()
    np.random.seed(42)
    train_units = np.random.choice(units, size=int(0.8 * len(units)), replace=False)
    test_units = np.array([u for u in units if u not in train_units])
    
    train_data = data[data['unit'].isin(train_units)]
    test_data = data[data['unit'].isin(test_units)]
    
    # 4. Uncertainty Quantification Module
    print("\n2. DEMONSTRATING UNCERTAINTY QUANTIFICATION MODULE")
    
    # Extract features and targets
    feature_cols = [col for col in data.columns if 'sensor' in col or 'op' in col]
    X_train = train_data[feature_cols].values
    
    # Calculate RUL
    max_cycles = train_data.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    train_data = train_data.merge(max_cycles, on=['unit'], how='left')
    train_data['RUL'] = train_data['max_cycle'] - train_data['cycle']
    
    y_train = train_data['RUL'].values
    
    # Prepare test data
    X_test = test_data[feature_cols].values
    
    max_cycles = test_data.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    test_data = test_data.merge(max_cycles, on=['unit'], how='left')
    test_data['RUL'] = test_data['max_cycle'] - test_data['cycle']
    
    y_test = test_data['RUL'].values
    
    # Create and train BNN
    bnn = BayesianRULPredictor(use_aleatoric=True, use_epistemic=True)
    bnn.build_model(X_train.shape[1])
    bnn.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # Generate visualizations
    uncertainty_fig = bnn.visualize_uncertainty(X_test, y_test)
    uncertainty_fig.savefig('docs/figures/uncertainty_quantification.png', dpi=300, bbox_inches='tight')
    
    calibration_fig = bnn.calibration_plot(X_test, y_test)
    calibration_fig.savefig('docs/figures/uncertainty_calibration.png', dpi=300, bbox_inches='tight')
    
    print("Uncertainty quantification complete. Visualizations saved to 'docs/figures/'")
    
    # 5. Fault Disentanglement Autoencoder
    print("\n3. DEMONSTRATING FAULT DISENTANGLEMENT AUTOENCODER")
    autoencoder = FaultDisentanglementAutoencoder(
        system_latent_dim=10,
        fault_latent_dim=5,
        lambda_reg=0.1
    )
    
    autoencoder.fit(train_data, epochs=50, batch_size=32)
    
    # Generate visualizations
    fault_fig = autoencoder.visualize_fault_patterns(
        test_data, 
        cycle_data=test_data[['unit', 'cycle']]
    )
    fault_fig.savefig('docs/figures/fault_patterns.png', dpi=300, bbox_inches='tight')
    
    disentangle_fig = autoencoder.visualize_system_vs_fault(test_data)
    disentangle_fig.savefig('docs/figures/system_vs_fault.png', dpi=300, bbox_inches='tight')
    
    print("Fault disentanglement complete. Visualizations saved to 'docs/figures/'")
    
    print("\nAll novel components successfully demonstrated!")
    
    return {
        'regime_classifier': regime_classifier,
        'bayesian_nn': bnn,
        'fault_autoencoder': autoencoder
    }

if __name__ == "__main__":
    # Run demonstration
    models = demonstrate_novel_components()
