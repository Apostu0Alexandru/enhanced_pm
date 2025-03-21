import logging
import numpy as np
import pandas as pd
import os  # <-- Added
from data_processing.data_loader import NASADataProcessor
from data_processing.hybrid_generator import HybridDataEngine
from monitoring.alerts import MaintenanceAlertSystem
from models.fusion_model import HybridPredictiveModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    try:
        # 1. Load and preprocess NASA data
        processor = NASADataProcessor('nasa_data/train_FD001.txt')
        features, labels = processor.preprocess()
        logging.info(f"Loaded NASA data: {features.shape[0]} samples, {features.shape[1]} features")

        # 2. Train model on real data only
        model = HybridPredictiveModel(input_dim=features.shape[1])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        logging.info("Training model...")
        model.fit(features, labels, epochs=50, batch_size=256, validation_split=0.2)

        model.save('models/hybrid_model.h5', save_format='h5')
        logging.info("Model saved to models/hybrid_model.h5")

        # 3. Generate synthetic data for simulation
        generator = HybridDataEngine(features)
        synthetic_features = generator.generate_batch(1000)
        logging.info(f"Generated {synthetic_features.shape[0]} synthetic samples")

        # 4. Create hybrid dataset for monitoring
        hybrid_features = np.vstack([features, synthetic_features])

        # 5. Initialize alert system with real data stats
        alert_system = MaintenanceAlertSystem(features)

        alerts = alert_system.check_anomaly(hybrid_features)

        # 8. Save outputs
        max_length = max(len(v) for v in alerts.values())
        for key, value in alerts.items():
            if len(value) != max_length:
                logging.error(f"Array length mismatch: {key} has {len(value)} elements (expected {max_length})")
                raise ValueError(f"Array {key} has invalid length")

        # 9. Save outputs
        os.makedirs('output', exist_ok=True)
        pd.DataFrame(alerts).to_csv('output/alerts.csv', index=False)
        logging.info("Results saved to output/alerts.csv")
    except Exception as e:
            logging.error(f"Execution failed: {str(e)}")
            raise

if __name__ == "__main__":
    main()
