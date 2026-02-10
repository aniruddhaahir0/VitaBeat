"""
VitaBeat Dataset Generator
Main script to generate all synthetic datasets
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation.clinical_data import generate_heart_processed_data
from src.data_generation.cardio_data import generate_cardio_base_data, generate_cardiac_failure_processed
from src.data_generation.ecg_data import generate_ecg_timeseries_data


def generate_all_datasets(output_dir="raw", random_seed=42):
    """Generate all VitaBeat datasets."""
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🫀 VITABEAT DATASET GENERATION")
    print("=" * 70)
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"🎲 Random seed: {random_seed}")
    print("=" * 70)
    
    # 1. Generate heart_processed.csv
    print("\n[1/4] Generating Clinical Heart Disease Dataset...")
    print("-" * 70)
    heart_df = generate_heart_processed_data(n_samples=1000, random_seed=random_seed)
    heart_path = output_path / "heart_processed.csv"
    heart_df.to_csv(heart_path, index=False)
    print(f"✅ Saved: {heart_path}")
    
    # 2. Generate cardio_base.csv
    print("\n[2/4] Generating Large-Scale Cardiovascular Dataset...")
    print("-" * 70)
    cardio_df = generate_cardio_base_data(n_samples=70000, random_seed=random_seed)
    cardio_path = output_path / "cardio_base.csv"
    cardio_df.to_csv(cardio_path, index=False)
    print(f"✅ Saved: {cardio_path}")
    
    # 3. Generate cardiac_failure_processed.csv
    print("\n[3/4] Processing Cardiac Failure Dataset...")
    print("-" * 70)
    processed_df = generate_cardiac_failure_processed(cardio_df, random_seed=random_seed)
    processed_path = output_path / "cardiac_failure_processed.csv"
    processed_df.to_csv(processed_path, index=False)
    print(f"✅ Saved: {processed_path}")
    
    # 4. Generate ecg_timeseries.csv
    print("\n[4/4] Generating ECG Time-Series Dataset...")
    print("-" * 70)
    ecg_df = generate_ecg_timeseries_data(
        n_patients=250,
        time_steps_range=(300, 500),
        sampling_rate=250,
        random_seed=random_seed
    )
    ecg_path = output_path / "ecg_timeseries.csv"
    ecg_df.to_csv(ecg_path, index=False)
    print(f"✅ Saved: {ecg_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📊 Dataset Summary:")
    print(f"   1. heart_processed.csv           : {len(heart_df):>8,} rows")
    print(f"   2. cardio_base.csv               : {len(cardio_df):>8,} rows")
    print(f"   3. cardiac_failure_processed.csv : {len(processed_df):>8,} rows")
    print(f"   4. ecg_timeseries.csv            : {len(ecg_df):>8,} rows")
    print(f"\n   Total records generated          : {len(heart_df) + len(cardio_df) + len(processed_df) + len(ecg_df):>8,}")
    print("\n🚀 Next steps:")
    print("   python src/COMPLETE_PIPELINE.py")
    print("=" * 70)
    
    return {
        'heart_processed': heart_df,
        'cardio_base': cardio_df,
        'cardiac_failure_processed': processed_df,
        'ecg_timeseries': ecg_df
    }


if __name__ == "__main__":
    datasets = generate_all_datasets(output_dir="raw", random_seed=42)
    print("\n✨ VitaBeat datasets are ready for analysis!")