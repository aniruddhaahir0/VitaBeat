"""
VitaBeat Configuration
Global settings for data generation, model training, and evaluation
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Dataset Generation Parameters
DATASET_CONFIG = {
    'heart_processed': {
        'n_samples': 1000,
        'age_range': (29, 77),
        'save_path': RAW_DATA_DIR / 'heart_processed.csv'
    },
    'cardio_base': {
        'n_samples': 70000,
        'age_range_days': (10950, 25550),  # ~30-70 years in days
        'save_path': RAW_DATA_DIR / 'cardio_base.csv'
    },
    'cardiac_failure_processed': {
        'save_path': RAW_DATA_DIR / 'cardiac_failure_processed.csv'
    },
    'ecg_timeseries': {
        'n_patients': 250,
        'time_steps_range': (300, 500),
        'sampling_rate': 250,  # Hz
        'save_path': RAW_DATA_DIR / 'ecg_timeseries.csv'
    }
}

# Model Training Parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'validation_split': 0.2,
    'cv_folds': 5,
    'random_state': RANDOM_SEED,
    
    # Tabular Models
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_SEED
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    
    # ECG Deep Learning
    'ecg_cnn': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping_patience': 10
    }
}

# Risk Score Thresholds
RISK_THRESHOLDS = {
    'low': (0, 30),
    'moderate': (30, 70),
    'high': (70, 100)
}

# Visualization Settings
PLOT_CONFIG = {
    'figsize': (12, 8),
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'husl',
    'dpi': 300
}

print(f"✅ VitaBeat configuration loaded successfully")
print(f"📁 Data directory: {DATA_DIR}")
print(f"📊 Output directory: {OUTPUTS_DIR}")
print(f"🎲 Random seed: {RANDOM_SEED}")