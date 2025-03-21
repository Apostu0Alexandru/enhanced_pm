# threshold_alert.py
import numpy as np
from scipy.stats import zscore

class MaintenanceAlertSystem:
    def __init__(self, baseline_data, sensitivity=3.0):
        self.baseline_mean = np.nanmean(baseline_data, axis=0)
        self.baseline_std = np.nanstd(baseline_data, axis=0, ddof=1)
        self.sensitivity = sensitivity  # 3σ for industrial standards
        
    def check_anomaly(self, sensor_data_batch):
        """Process sensor data and generate alerts"""
        z_scores = np.abs((sensor_data_batch - self.baseline_mean) / (self.baseline_std + 1e-9))
        alerts = z_scores > self.sensitivity
        
        # Structure output as a dictionary with equal-length arrays
        return {
            'sample_id': np.repeat(np.arange(len(sensor_data_batch)), sensor_data_batch.shape[1]).tolist(),
            'parameter_names': [f'Sensor_{i+1}' for _ in range(len(sensor_data_batch)) for i in range(sensor_data_batch.shape[1])],
            'alerts': alerts.flatten().tolist(),
            'severity': z_scores.flatten().tolist(),
            'recommended_action': self._get_actions(alerts.flatten())
        }

    def _get_actions(self, alerts_flat):
        """Process flattened alerts array directly"""
        return ["Inspect Sensor and check lubrication" if alert else "No action needed" 
                for alert in alerts_flat]