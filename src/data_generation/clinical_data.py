"""
Clinical Heart Disease Dataset Generator
Generates heart_processed.csv with realistic clinical parameters
"""

import numpy as np
import pandas as pd
from scipy import stats


def generate_heart_processed_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic clinical heart disease dataset with medical realism.
    
    Parameters:
    -----------
    n_samples : int
        Number of patient records to generate
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Clinical dataset with cardiovascular risk factors
    """
    np.random.seed(random_seed)
    
    print(f"🫀 Generating heart_processed.csv with {n_samples} patients...")
    
    # Age distribution (29-77 years, skewed toward older ages)
    age = np.random.gamma(shape=3, scale=15, size=n_samples).astype(int)
    age = np.clip(age + 29, 29, 77)
    
    # Sex (0=Female, 1=Male) - slightly more males in heart disease studies
    sex_m = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    
    # Chest Pain Types (one-hot encoded)
    chest_pain_types = np.random.choice(
        ['ATA', 'NAP', 'TA', 'ASY'], 
        size=n_samples,
        p=[0.20, 0.25, 0.15, 0.40]  # ASY (asymptomatic) most common in early disease
    )
    chest_pain_ata = (chest_pain_types == 'ATA').astype(int)
    chest_pain_nap = (chest_pain_types == 'NAP').astype(int)
    chest_pain_ta = (chest_pain_types == 'TA').astype(int)
    
    # Resting Blood Pressure (mm Hg) - correlated with age
    resting_bp = 110 + (age - 29) * 0.5 + np.random.normal(0, 15, n_samples)
    resting_bp = np.clip(resting_bp, 90, 200).astype(int)
    
    # Cholesterol (mg/dl) - increases with age and risk
    cholesterol = 180 + (age - 29) * 1.2 + np.random.normal(0, 40, n_samples)
    cholesterol = np.clip(cholesterol, 120, 400).astype(int)
    
    # Fasting Blood Sugar > 120 mg/dl (diabetes indicator)
    fbs_prob = 0.10 + (age - 29) / 500 + (cholesterol > 240) * 0.15
    fasting_bs = (np.random.random(n_samples) < fbs_prob).astype(int)
    
    # Resting ECG (one-hot encoded)
    ecg_types = np.random.choice(
        ['Normal', 'ST', 'LVH'],
        size=n_samples,
        p=[0.60, 0.25, 0.15]
    )
    resting_ecg_normal = (ecg_types == 'Normal').astype(int)
    resting_ecg_st = (ecg_types == 'ST').astype(int)
    
    # Maximum Heart Rate - inversely correlated with age
    max_hr = 220 - age + np.random.normal(0, 15, n_samples)
    max_hr = np.clip(max_hr, 60, 202).astype(int)
    
    # Exercise-Induced Angina
    angina_prob = 0.15 + (age - 29) / 300 + (chest_pain_types == 'TA') * 0.30
    exercise_angina_y = (np.random.random(n_samples) < angina_prob).astype(int)
    
    # Oldpeak (ST depression) - strong heart disease indicator
    oldpeak = np.random.exponential(scale=0.8, size=n_samples)
    oldpeak = np.clip(oldpeak, 0, 6.2)
    oldpeak = np.round(oldpeak, 1)
    
    # ST Slope (one-hot encoded)
    st_slope_types = np.random.choice(
        ['Up', 'Flat', 'Down'],
        size=n_samples,
        p=[0.50, 0.35, 0.15]
    )
    st_slope_flat = (st_slope_types == 'Flat').astype(int)
    st_slope_up = (st_slope_types == 'Up').astype(int)
    
    # --- HEART DISEASE TARGET (Medical Risk Logic) ---
    risk_score = (
        (age - 29) / 50 * 15 +
        (cholesterol - 180) / 100 * 10 +
        (resting_bp - 120) / 40 * 8 +
        oldpeak * 12 +
        exercise_angina_y * 15 +
        fasting_bs * 8 +
        (max_hr < 120) * 10 +
        st_slope_flat * 8 +
        (chest_pain_types == 'TA') * 10 +
        (ecg_types == 'ST') * 7 +
        sex_m * 5 +
        np.random.normal(0, 10, n_samples)
    )
    
    disease_probability = 1 / (1 + np.exp(-(risk_score - 40) / 10))
    heart_disease = (np.random.random(n_samples) < disease_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Sex_M': sex_m,
        'ChestPainType_ATA': chest_pain_ata,
        'ChestPainType_NAP': chest_pain_nap,
        'ChestPainType_TA': chest_pain_ta,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG_Normal': resting_ecg_normal,
        'RestingECG_ST': resting_ecg_st,
        'MaxHR': max_hr,
        'ExerciseAngina_Y': exercise_angina_y,
        'Oldpeak': oldpeak,
        'ST_Slope_Flat': st_slope_flat,
        'ST_Slope_Up': st_slope_up,
        'HeartDisease': heart_disease
    })
    
    print(f"✅ Generated {len(df)} patient records")
    print(f"📊 Heart Disease prevalence: {heart_disease.mean():.1%}")
    print(f"👥 Male ratio: {sex_m.mean():.1%}")
    print(f"📈 Age range: {age.min()}-{age.max()} years")
    
    return df


if __name__ == "__main__":
    from pathlib import Path
    
    df = generate_heart_processed_data(n_samples=1000, random_seed=42)
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "heart_processed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved to: {output_path}")