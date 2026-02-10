# VitaBeat

An AI-powered cardiovascular risk detection system that uses clinical data and ECG signals to predict heart disease risk.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**Step 1 - Generate datasets:**
```bash
python data/generate_datasets.py
```

**Step 2 - Run the pipeline:**
```bash
python src/COMPLETE_PIPELINE.py
```

## Project Structure

```
VitaBeat/
├── config.py
├── requirements.txt
├── data/
│   ├── generate_datasets.py
│   └── raw/
├── src/
│   ├── COMPLETE_PIPELINE.py
│   └── data_generation/
│       ├── clinical_data.py
│       ├── cardio_data.py
│       └── ecg_data.py
└── outputs/
    ├── plots/
    └── models/
```

## Datasets

| File | Rows | Description |
|------|------|-------------|
| heart_processed.csv | 1,000 | Clinical heart disease data |
| cardio_base.csv | 70,000 | Large-scale cardiovascular data |
| cardiac_failure_processed.csv | 70,000 | Normalized ML-ready data |
| ecg_timeseries.csv | 100,627 | Synthetic ECG signals |

## Models

- Logistic Regression
- Random Forest
- XGBoost
- ECG CNN (deep learning)

