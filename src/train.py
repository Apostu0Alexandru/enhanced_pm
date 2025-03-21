from preprocessing.feature_pipeline import create_hierarchical_feature_pipeline, prepare_tcn_sequences
from models.tcn import TemporalConvNet
import yaml

def main():
    # Load configuration
    with open('configs/feature_params.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Process data
    dataset_type = 'FD002'  # Change as needed
    processed_data = create_hierarchical_feature_pipeline(dataset_type, f'data/{dataset_type}.txt')
    
    # Prepare sequences for TCN (if using FD001 or FD003)
    if dataset_type in ['FD001', 'FD003']:
        X_tcn, y_tcn = prepare_tcn_sequences(
            processed_data, 
            sequence_length=config['feature_engineering']['single_condition']['sequence_length'],
            prediction_horizon=config['feature_engineering']['single_condition']['prediction_horizon']
        )
        
        # Initialize and train TCN model (placeholder)
        input_size = X_tcn.shape[2]  # number of features
        model = TemporalConvNet(input_size, 1)  # 1 for RUL prediction
        # Add training loop here
    
    else:
        # Add logic for multi-condition datasets (FD002, FD004)
        pass

if __name__ == "__main__":
    main()
