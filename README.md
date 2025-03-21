# Enhanced Predictive Maintenance System

This project implements a hierarchical feature engineering pipeline for the CMAPSS Turbofan Engine dataset, addressing both multi-condition (FD002/FD004) and single-condition (FD001/FD003) datasets.

## Feature Engineering Pipeline

Our hierarchical processing approach:
- For FD002/FD004 (multiple operational conditions):
  - K-means clustering of operational regimes (6 clusters)
  - Regime-specific feature extraction:
    * Rolling RMS for vibration sensors
    * EWMA for temperature sensors
    * Pressure derivatives for degradation rate

- For FD001/FD003 (single condition):
  - Sequence preparation for Temporal Convolutional Networks
  - Time-domain normalization and statistical features

## Usage

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Run the training script: `python src/train.py`

## Project Structure

- `src/`: Source code
  - `preprocessing/`: Feature engineering code
  - `models/`: Model architectures
  - `utils/`: Helper functions
- `notebooks/`: Jupyter notebooks for analysis
- `configs/`: Configuration files
- `data/`: Dataset files (not tracked in git)
