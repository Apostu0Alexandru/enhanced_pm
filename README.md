# Enhanced Predictive Maintenance System

## Overview
This project implements a novel approach to Remaining Useful Life (RUL) prediction for turbofan engines using hierarchical feature engineering and deep learning techniques. The system incorporates operational regime classification and temporal convolutional networks (TCN) to enhance prediction accuracy across diverse operating conditions.

## Research Background
Predictive maintenance has evolved beyond traditional condition monitoring to incorporate machine learning techniques that can forecast equipment failures before they occur. This project builds upon several key research advances:

- **Operational Regime Recognition**: Research by Heimes (2008) and Ramasso (2014) demonstrated that operational context significantly impacts degradation patterns.
- **Deep Learning for RUL Prediction**: Li et al. (2018) and Zhang et al. (2019) showed that deep learning approaches outperform traditional methods for complex time-series data.
- **Hierarchical Feature Engineering**: Saxena et al. (2008) and Wang et al. (2012) established that domain-specific feature extraction improves model performance in PHM applications.
- **Temporal Convolutional Networks**: Recent work by Bai et al. (2018) showed that TCNs can outperform recurrent architectures for many sequence modeling tasks.

## Technical Approach

### Multi-condition Architecture
For datasets with multiple operational conditions (FD002/FD004):
1. **Regime Classification**: Unsupervised clustering identifies operational regimes
2. **Regime-Specific Feature Extraction**: Different feature engineering pipelines for each regime
3. **Ensemble Model Training**: Specialized models for each operational regime
4. **Hierarchical Prediction**: Regime-aware RUL prediction

### Single-condition Architecture
For datasets with single operational conditions (FD001/FD003):
1. **Temporal Feature Extraction**: Time-domain and frequency-domain features
2. **Sequence Preparation**: Fixed-width windowing for TCN input
3. **TCN Model Architecture**: Dilated causal convolutions with residual connections
4. **Uncertainty Quantification**: Probabilistic predictions with confidence intervals

## Dataset

This project uses the NASA CMAPSS Turbofan Engine Degradation Simulation Dataset, which consists of multiple multivariate time series representing engine degradation:

- **FD001**: Single operating condition, single failure mode
- **FD002**: Six operating conditions, single failure mode
- **FD003**: Single operating condition, two failure modes
- **FD004**: Six operating conditions, two failure modes

The dataset is available from the [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced_pm.git
cd enhanced_pm

# Create and activate a virtual environment
python -m venv .thesisenv
source .thesisenv/bin/activate  # On Windows: .thesisenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NASA CMAPSS dataset
# Place dataset files in the nasa_data/ directory
```

## Usage

### Running the Full Validation Pipeline

```bash
python src/validate_end_to_end.py
```

This command validates the entire pipeline from data preprocessing to model prediction, ensuring all components work together correctly.

### Training Models

```bash
python src/train.py --dataset FD001 --epochs 100
```

### Making Predictions

```bash
python src/predict.py --model-path models/tcn_fd001.pt --test-data path/to/test_data.txt
```

## Project Structure

```
enhanced_pm/
│
├── src/                    # Source code
│   ├── preprocessing/      # Data preprocessing modules
│   │   ├── feature_pipeline.py     # Feature engineering pipeline
│   │   └── sequence_generation.py  # Sequence preparation for deep learning
│   │
│   ├── models/             # Model architectures
│   │   ├── tcn.py          # Temporal Convolutional Network
│   │   ├── regime_classifier.py  # Operational regime classifier
│   │   └── survival_models.py    # Survival analysis models
│   │
│   ├── validate_end_to_end.py  # End-to-end validation script
│   └── utils/              # Utility functions
│
├── tests/                  # Unit and integration tests
│   └── novelty/            # Tests for novelty detection
│
├── nasa_data/              # NASA CMAPSS dataset (not tracked in git)
│
├── notebooks/              # Jupyter notebooks for analysis
│
├── configs/                # Configuration files
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Technical Contributions

This project makes several technical contributions to the field of predictive maintenance:

1. **Operational Regime-Aware Feature Engineering**: A novel approach to extract regime-specific features
2. **Adaptive Multi-condition Pipeline**: Automatic pipeline selection based on dataset characteristics
3. **TCN Optimization for RUL Prediction**: Customized TCN architecture for time-to-failure prediction
4. **End-to-End Validation Framework**: Comprehensive testing of the entire predictive maintenance pipeline

## Performance Metrics

| Dataset | RMSE  | Score | R² |
|---------|-------|-------|-----|
| FD001   | 15.21 | 368   | 0.87|
| FD002   | 18.63 | 482   | 0.82|
| FD003   | 16.09 | 422   | 0.85|
| FD004   | 19.82 | 530   | 0.79|

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *International Conference on Prognostics and Health Management*.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*.
- Li, X., Ding, Q., & Sun, J. Q. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. *Reliability Engineering & System Safety*, 172, 1-11.
- Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation. *International Conference on Prognostics and Health Management*.
- Wang, T., Yu, J., Siegel, D., & Lee, J. (2012). A similarity-based prognostics approach for remaining useful life estimation of engineered systems. *International Conference on Prognostics and Health Management*.
