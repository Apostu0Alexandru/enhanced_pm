import numpy as np
from scipy.stats import entropy

class HybridDataEngine:
    def __init__(self, real_data_np):
        """Initialize with NaN-safe statistics"""
        with np.errstate(invalid='ignore'):
            self.real_stats = {
                'mean': np.nanmean(real_data_np, axis=0),
                'std': np.nanstd(real_data_np, axis=0, ddof=1)
            }
        
        # Replace NaN/zero values
        self.real_stats['mean'] = np.nan_to_num(self.real_stats['mean'], nan=0.0)
        self.real_stats['std'] = np.nan_to_num(self.real_stats['std'], nan=1.0)
        self.n_features = real_data_np.shape[1]

    def generate_batch(self, n_samples):
        return np.random.normal(
            loc=self.real_stats['mean'],
            scale=self.real_stats['std'],
            size=(n_samples, self.n_features)
        ) # Remove problematic tsaug transformations

    @staticmethod
    def dynamic_weight(real, synthetic):
        """Fixed entropy calculation with matching dimensions"""
        # Ensure same number of features
        real = real[:, :synthetic.shape[1]] if synthetic.shape[1] < real.shape[1] else real
        synthetic = synthetic[:, :real.shape[1]] if real.shape[1] < synthetic.shape[1] else synthetic
        
        # Calculate entropy per feature
        real_ent = np.apply_along_axis(lambda x: entropy(x, base=2), 0, real)
        synth_ent = np.apply_along_axis(lambda x: entropy(x, base=2), 0, synthetic)
        
        # Ensure compatible dimensions
        return 1 - (synth_ent / (real_ent + synth_ent + 1e-9)).mean()
