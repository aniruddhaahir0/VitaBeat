"""
ECG Time-Series Dataset Generator
Generates synthetic ECG signals for deep learning
"""

import numpy as np
import pandas as pd


def generate_normal_ecg(time_steps, sampling_rate=250, heart_rate=75):
    """Generate normal ECG waveform."""
    t = np.arange(time_steps) / sampling_rate
    f_heart = heart_rate / 60
    
    # P wave
    p_wave = 0.15 * np.sin(2 * np.pi * f_heart * t)
    
    # QRS complex
    qrs_freq = f_heart * 5
    qrs_complex = 1.0 * np.sin(2 * np.pi * qrs_freq * t) * np.exp(-((t % (1/f_heart) - 0.15)**2) / 0.002)
    
    # T wave
    t_wave = 0.3 * np.sin(2 * np.pi * f_heart * t + np.pi/3)
    
    ecg = p_wave + qrs_complex + t_wave
    baseline = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    noise = np.random.normal(0, 0.02, time_steps)
    
    return ecg + baseline + noise


def generate_abnormal_ecg(time_steps, sampling_rate=250, abnormality_type='random'):
    """Generate abnormal ECG with pathological patterns."""
    t = np.arange(time_steps) / sampling_rate
    
    if abnormality_type == 'random':
        abnormality_type = np.random.choice(['arrhythmia', 'ischemia', 'hypertrophy'])
    
    if abnormality_type == 'arrhythmia':
        irregular_hr = 60 + 30 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 15, time_steps)
        irregular_hr = np.clip(irregular_hr, 40, 150)
        f_heart = irregular_hr / 60
        qrs_freq = f_heart * 5
        
        p_wave = 0.08 * np.sin(2 * np.pi * f_heart * t)
        qrs_complex = 0.9 * np.sin(2 * np.pi * qrs_freq * t) * np.exp(-((t % 0.8 - 0.15)**2) / 0.003)
        t_wave = 0.25 * np.sin(2 * np.pi * f_heart * t + np.pi/4)
        
    elif abnormality_type == 'ischemia':
        f_heart = 80 / 60
        qrs_freq = f_heart * 5
        
        p_wave = 0.15 * np.sin(2 * np.pi * f_heart * t)
        qrs_complex = 1.0 * np.sin(2 * np.pi * qrs_freq * t) * np.exp(-((t % (1/f_heart) - 0.15)**2) / 0.002)
        st_depression = -0.2 * np.ones_like(t)
        t_wave = 0.2 * np.sin(2 * np.pi * f_heart * t + np.pi/3) + st_depression
        
    else:  # hypertrophy
        f_heart = 70 / 60
        qrs_freq = f_heart * 5
        
        p_wave = 0.18 * np.sin(2 * np.pi * f_heart * t)
        qrs_complex = 1.5 * np.sin(2 * np.pi * qrs_freq * t) * np.exp(-((t % (1/f_heart) - 0.15)**2) / 0.002)
        t_wave = 0.35 * np.sin(2 * np.pi * f_heart * t + np.pi/3)
    
    ecg = p_wave + qrs_complex + t_wave
    baseline = 0.08 * np.sin(2 * np.pi * 0.4 * t)
    noise = np.random.normal(0, 0.04, time_steps)
    
    return ecg + baseline + noise


def generate_ecg_timeseries_data(n_patients=250, time_steps_range=(300, 500), 
                                  sampling_rate=250, random_seed=42):
    """Generate complete ECG time-series dataset."""
    np.random.seed(random_seed)
    
    print(f"💓 Generating ecg_timeseries.csv with {n_patients} patients...")
    
    all_records = []
    n_normal = n_patients // 2
    n_abnormal = n_patients - n_normal
    
    # Normal ECGs
    for patient_id in range(1, n_normal + 1):
        time_steps = np.random.randint(time_steps_range[0], time_steps_range[1])
        heart_rate = np.random.randint(60, 100)
        ecg_signal = generate_normal_ecg(time_steps, sampling_rate, heart_rate)
        
        for step in range(time_steps):
            all_records.append({
                'patient_id': patient_id,
                'time_step': step,
                'ecg_signal': ecg_signal[step],
                'label': 0
            })
    
    # Abnormal ECGs
    for patient_id in range(n_normal + 1, n_patients + 1):
        time_steps = np.random.randint(time_steps_range[0], time_steps_range[1])
        abnormality = np.random.choice(['arrhythmia', 'ischemia', 'hypertrophy'])
        ecg_signal = generate_abnormal_ecg(time_steps, sampling_rate, abnormality)
        
        for step in range(time_steps):
            all_records.append({
                'patient_id': patient_id,
                'time_step': step,
                'ecg_signal': ecg_signal[step],
                'label': 1
            })
    
    df = pd.DataFrame(all_records)
    
    print(f"✅ Generated {len(df)} time-series records for {n_patients} patients")
    print(f"📊 Normal: {n_normal} patients, Abnormal: {n_abnormal} patients")
    
    return df


if __name__ == "__main__":
    from pathlib import Path
    
    ecg_df = generate_ecg_timeseries_data(n_patients=250, random_seed=42)
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "ecg_timeseries.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ecg_df.to_csv(output_path, index=False)
    print(f"💾 Saved to: {output_path}")