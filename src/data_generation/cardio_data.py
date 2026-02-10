"""
Large-Scale Cardiovascular Dataset Generator
Generates cardio_base.csv and cardiac_failure_processed.csv
"""

import numpy as np
import pandas as pd


def generate_cardio_base_data(n_samples=70000, random_seed=42):
    """Generate large-scale cardiovascular risk dataset."""
    np.random.seed(random_seed)
    
    print(f"❤️ Generating cardio_base.csv with {n_samples} patients...")
    
    # Patient ID
    patient_id = np.arange(1, n_samples + 1)
    
    # Age in days (30-70 years)
    age_years = np.random.gamma(shape=5, scale=8, size=n_samples)
    age_years = np.clip(age_years + 30, 30, 70)
    age_days = (age_years * 365.25).astype(int)
    
    # Gender (1=female, 2=male)
    gender = np.random.choice([1, 2], size=n_samples, p=[0.52, 0.48])
    
    # Height (cm) - gender-dependent
    height = np.where(
        gender == 1,
        np.random.normal(163, 7, n_samples),
        np.random.normal(176, 8, n_samples)
    )
    height = np.clip(height, 140, 210).astype(int)
    
    # Weight (kg) - BMI-based
    base_bmi = np.random.gamma(shape=6, scale=3.5, size=n_samples) + 18
    base_bmi = np.clip(base_bmi, 18, 45)
    weight = base_bmi * (height / 100) ** 2
    weight = np.clip(weight, 40, 150)
    
    # Blood Pressure
    age_effect = (age_years - 30) * 0.8
    bmi_effect = (base_bmi - 25) * 1.5
    
    ap_hi = 120 + age_effect + bmi_effect + np.random.normal(0, 15, n_samples)
    ap_hi = np.clip(ap_hi, 90, 200).astype(int)
    
    ap_lo = 80 + age_effect * 0.5 + bmi_effect * 0.8 + np.random.normal(0, 10, n_samples)
    ap_lo = np.clip(ap_lo, 60, 140).astype(int)
    ap_lo = np.minimum(ap_lo, ap_hi - 20)
    
    # Cholesterol (1=normal, 2=above, 3=well above)
    chol_prob = np.column_stack([
        np.maximum(0.1, 0.6 - (age_years - 30) / 100 - (base_bmi - 25) / 40),
        0.25 * np.ones(n_samples),
        np.maximum(0.05, (age_years - 30) / 150 + (base_bmi - 25) / 50)
    ])
    chol_prob = chol_prob / chol_prob.sum(axis=1, keepdims=True)
    cholesterol = np.array([np.random.choice([1, 2, 3], p=p) for p in chol_prob])
    
    # Glucose (1=normal, 2=above, 3=well above)
    gluc_prob = np.column_stack([
        np.maximum(0.1, 0.65 - (age_years - 30) / 120 - (base_bmi - 25) / 50),
        0.25 * np.ones(n_samples),
        np.maximum(0.05, (age_years - 30) / 180 + (base_bmi - 25) / 60)
    ])
    gluc_prob = gluc_prob / gluc_prob.sum(axis=1, keepdims=True)
    gluc = np.array([np.random.choice([1, 2, 3], p=p) for p in gluc_prob])
    
    # Smoking
    smoke_prob = np.where(gender == 2, 0.30 - (age_years - 30) / 200, 0.15 - (age_years - 30) / 300)
    smoke = (np.random.random(n_samples) < smoke_prob).astype(int)
    
    # Alcohol
    alco_prob = np.where(gender == 2, 0.20 + (age_years - 30) / 300, 0.10 + (age_years - 30) / 400)
    alco = (np.random.random(n_samples) < alco_prob).astype(int)
    
    # Physical activity
    active_prob = 0.65 - (age_years - 30) / 150 - (base_bmi - 25) / 80
    active_prob = np.clip(active_prob, 0.25, 0.85)
    active = (np.random.random(n_samples) < active_prob).astype(int)
    
    # Cardiovascular disease target
    hypertension = ((ap_hi > 140) | (ap_lo > 90)).astype(int)
    bmi_risk = np.clip((base_bmi - 25) / 20, 0, 1)
    
    risk_score = (
        (age_years - 30) / 40 * 20 +
        hypertension * 25 +
        (cholesterol - 1) * 10 +
        (gluc - 1) * 8 +
        smoke * 12 +
        (1 - active) * 10 +
        bmi_risk * 15 +
        alco * 5 +
        (gender == 2) * 5 +
        np.random.normal(0, 8, n_samples)
    )
    
    cardio_probability = 1 / (1 + np.exp(-(risk_score - 50) / 15))
    cardio = (np.random.random(n_samples) < cardio_probability).astype(int)
    
    df = pd.DataFrame({
        'id': patient_id,
        'age': age_days,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active,
        'cardio': cardio
    })
    
    print(f"✅ Generated {len(df)} patient records")
    print(f"📊 Cardiovascular disease prevalence: {cardio.mean():.1%}")
    
    return df


def generate_cardiac_failure_processed(cardio_df, random_seed=42):
    """Process cardio_base data into ML-ready format."""
    np.random.seed(random_seed)
    
    print(f"\n🔧 Processing cardiac_failure_processed.csv...")
    
    df = cardio_df.copy()
    
    # Normalize age
    df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
    
    # Remove outliers
    df = df[(df['ap_hi'] >= 90) & (df['ap_hi'] <= 200)]
    df = df[(df['ap_lo'] >= 60) & (df['ap_lo'] <= 140)]
    
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]
    
    # Normalize continuous features
    for col in ['height', 'weight', 'ap_hi', 'ap_lo']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    df = df.drop('bmi', axis=1)
    df = df.reset_index(drop=True)
    
    print(f"✅ Processed dataset: {len(df)} records")
    
    return df


if __name__ == "__main__":
    from pathlib import Path
    
    cardio_df = generate_cardio_base_data(n_samples=70000, random_seed=42)
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "cardio_base.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cardio_df.to_csv(output_path, index=False)
    
    processed_df = generate_cardiac_failure_processed(cardio_df)
    output_path_proc = Path(__file__).parent.parent.parent / "data" / "raw" / "cardiac_failure_processed.csv"
    processed_df.to_csv(output_path_proc, index=False)