"""
VitaBeat - Complete ML Pipeline Starter Code
This file contains all the code you need to build the complete system
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, roc_curve, confusion_matrix,
                              classification_report)

# Import calibration metrics from correct location
try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
except ImportError:
    # For older sklearn versions
    try:
        from sklearn.metrics import calibration_curve, brier_score_loss
    except ImportError:
        print("⚠️  Calibration functions not available in this sklearn version")
        calibration_curve = None
        brier_score_loss = None

import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP not installed. Install with: pip install shap")

# Set random seeds
np.random.seed(42)

# Set plot style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# 1. DATA LOADING & EDA
# ============================================================================

def load_datasets():
    """Load all VitaBeat datasets"""
    print("\n" + "="*80)
    print("📂 LOADING DATASETS")
    print("="*80)
    
    try:
        heart_df = pd.read_csv('data/raw/heart_processed.csv')
        print(f"✅ Loaded heart_processed.csv: {heart_df.shape}")
    except FileNotFoundError:
        print("❌ heart_processed.csv not found. Run: python data/generate_datasets.py")
        return None, None, None, None
    
    try:
        cardio_df = pd.read_csv('data/raw/cardio_base.csv')
        print(f"✅ Loaded cardio_base.csv: {cardio_df.shape}")
    except FileNotFoundError:
        cardio_df = None
        print("⚠️  cardio_base.csv not found")
    
    try:
        processed_df = pd.read_csv('data/raw/cardiac_failure_processed.csv')
        print(f"✅ Loaded cardiac_failure_processed.csv: {processed_df.shape}")
    except FileNotFoundError:
        processed_df = None
        print("⚠️  cardiac_failure_processed.csv not found")
    
    try:
        ecg_df = pd.read_csv('data/raw/ecg_timeseries.csv')
        print(f"✅ Loaded ecg_timeseries.csv: {ecg_df.shape}")
    except FileNotFoundError:
        ecg_df = None
        print("⚠️  ecg_timeseries.csv not found")
    
    return heart_df, cardio_df, processed_df, ecg_df


def perform_eda(df, dataset_name="Dataset"):
    """Comprehensive Exploratory Data Analysis"""
    print(f"\n{'='*80}")
    print(f"📊 EDA: {dataset_name}")
    print(f"{'='*80}")
    
    # Basic info
    print(f"\n📋 Shape: {df.shape}")
    print(f"\n🔍 Columns: {list(df.columns)}")
    print(f"\n📈 Missing Values:\n{df.isnull().sum().sum()} total missing values")
    print(f"\n📊 Statistical Summary:")
    print(df.describe())
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=0.5)
    plt.title(f'{dataset_name} - Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{dataset_name}_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved correlation heatmap")
    
    # Target distribution (if present)
    target_col = None
    if 'HeartDisease' in df.columns:
        target_col = 'HeartDisease'
    elif 'cardio' in df.columns:
        target_col = 'cardio'
    
    if target_col:
        plt.figure(figsize=(8, 6))
        counts = df[target_col].value_counts()
        plt.bar(['Healthy', 'Disease'], counts.values, color=['skyblue', 'salmon'])
        plt.title(f'{dataset_name} - Target Distribution', fontsize=14, fontweight='bold')
        plt.xlabel(target_col)
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(counts.values):
            plt.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'outputs/plots/{dataset_name}_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved target distribution")
        
        print(f"\n🎯 Class Balance:")
        print(df[target_col].value_counts(normalize=True))


def visualize_ecg_samples(ecg_df, n_samples=4):
    """Visualize sample ECG waveforms"""
    if ecg_df is None:
        print("⚠️  ECG data not available")
        return
    
    print(f"\n{'='*80}")
    print(f"💓 ECG VISUALIZATION")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Get patient IDs (2 normal, 2 abnormal)
    normal_ids = ecg_df[ecg_df['label'] == 0]['patient_id'].unique()[:2]
    abnormal_ids = ecg_df[ecg_df['label'] == 1]['patient_id'].unique()[:2]
    
    patient_ids = list(normal_ids) + list(abnormal_ids)
    
    for idx, patient_id in enumerate(patient_ids):
        patient_data = ecg_df[ecg_df['patient_id'] == patient_id]
        label = patient_data['label'].iloc[0]
        label_text = "Normal" if label == 0 else "Abnormal"
        
        axes[idx].plot(patient_data['time_step'], patient_data['ecg_signal'], 
                       color='blue' if label == 0 else 'red', linewidth=0.8, alpha=0.8)
        axes[idx].set_title(f'Patient {patient_id} - {label_text}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time Step')
        axes[idx].set_ylabel('ECG Signal (mV)')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Sample ECG Waveforms', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('outputs/plots/ecg_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved ECG sample visualization")


# ============================================================================
# 2. TABULAR ML MODELS
# ============================================================================

def train_tabular_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple tabular models"""
    
    print(f"\n{'='*80}")
    print(f"🤖 TRAINING TABULAR MODELS")
    print(f"{'='*80}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    models = {}
    
    # 1. Logistic Regression
    print("\n🔵 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    results['Logistic Regression'] = evaluate_model(lr, X_test_scaled, y_test, "Logistic_Regression")
    models['lr'] = lr
    
    # 2. Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results['Random Forest'] = evaluate_model(rf, X_test, y_test, "Random_Forest")
    models['rf'] = rf
    
    # 3. XGBoost (if available)
    if HAS_XGBOOST:
        print("\n🚀 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                       random_state=42, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        models['xgb'] = xgb_model
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 MODEL COMPARISON")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())
    
    return results, models, scaler


def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n📊 {model_name.replace('_', ' ')} Results:")
    for metric, value in metrics.items():
        print(f"   {metric.capitalize():12s}: {value:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'Disease'],
                yticklabels=['Healthy', 'Disease'])
    plt.title(f'{model_name.replace("_", " ")} - Confusion Matrix', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name.replace("_", " ")} - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


# ============================================================================
# 3. CARDIOVASCULAR RISK SCORING (0-100)
# ============================================================================

def calculate_risk_score(probabilities):
    """
    Convert prediction probabilities to 0-100 risk score
    
    Risk Categories:
    - Low Risk: 0-30
    - Moderate Risk: 30-70
    - High Risk: 70-100
    """
    risk_scores = probabilities * 100
    
    risk_categories = []
    for score in risk_scores:
        if score < 30:
            risk_categories.append('Low Risk')
        elif score < 70:
            risk_categories.append('Moderate Risk')
        else:
            risk_categories.append('High Risk')
    
    return risk_scores, risk_categories


def demonstrate_risk_scoring(model, X_test, feature_names, n_samples=5):
    """Demonstrate risk scoring on sample patients"""
    
    print(f"\n{'='*80}")
    print(f"🫀 VITABEAT RISK ASSESSMENT - Sample Patients")
    print(f"{'='*80}")
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    risk_scores, risk_categories = calculate_risk_score(y_proba)
    
    # Select sample patients
    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\nPatient #{i}")
        print("-" * 40)
        print(f"Risk Score: {risk_scores[idx]:.1f}/100")
        print(f"Risk Category: {risk_categories[idx]}")
        print(f"Clinical Features:")
        
        # Show top features
        if hasattr(X_test, 'iloc'):
            patient_features = X_test.iloc[idx]
            for feat_name, feat_val in list(zip(feature_names, patient_features))[:5]:
                print(f"   - {feat_name}: {feat_val:.2f}")
        else:
            print(f"   [Feature values: {X_test[idx][:5]}]")


# ============================================================================
# 4. ECG DEEP LEARNING MODEL (CNN) - SIMPLIFIED
# ============================================================================

def prepare_ecg_data(ecg_df, target_length=400):
    """Prepare ECG data for deep learning"""
    if ecg_df is None:
        print("⚠️  ECG data not available")
        return None, None, None, None
    
    print(f"\n{'='*80}")
    print(f"🔧 PREPARING ECG DATA")
    print(f"{'='*80}")
    
    # Group by patient
    patient_sequences = []
    labels = []
    
    for patient_id in ecg_df['patient_id'].unique():
        patient_data = ecg_df[ecg_df['patient_id'] == patient_id]
        sequence = patient_data['ecg_signal'].values
        label = patient_data['label'].iloc[0]
        
        # Pad/truncate to fixed length
        if len(sequence) < target_length:
            sequence = np.pad(sequence, (0, target_length - len(sequence)), 'constant')
        else:
            sequence = sequence[:target_length]
        
        patient_sequences.append(sequence)
        labels.append(label)
    
    X = np.array(patient_sequences).reshape(-1, target_length, 1)
    y = np.array(labels)
    
    print(f"✅ Prepared {len(X)} ECG sequences")
    print(f"   Shape: {X.shape}")
    print(f"   Labels: {np.bincount(y)}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ============================================================================
# 5. EXPLAINABILITY (SHAP) - OPTIONAL
# ============================================================================

def explain_with_shap(model, X_train, X_test, feature_names):
    """Generate SHAP explanations"""
    
    if not HAS_SHAP:
        print("\n⚠️  SHAP not available. Install with: pip install shap")
        return None, None
    
    print(f"\n{'='*80}")
    print(f"🔍 GENERATING SHAP EXPLANATIONS")
    print(f"{'='*80}")
    
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values (use subset for speed)
        sample_size = min(100, len(X_test))
        if hasattr(X_test, 'iloc'):
            X_sample = X_test.iloc[:sample_size]
        else:
            X_sample = X_test[:sample_size]
        
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('outputs/plots/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved SHAP summary plot")
        
        # Feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\n📊 Top 10 Most Important Features (SHAP):")
        print(importance_df.head(10).to_string(index=False))
        
        return shap_values, importance_df
        
    except Exception as e:
        print(f"⚠️  SHAP analysis failed: {e}")
        return None, None


# ============================================================================
# 6. CALIBRATION & RELIABILITY
# ============================================================================

def plot_calibration_curve_safe(y_true, y_proba, model_name="Model"):
    """Plot calibration curve with error handling"""
    
    if calibration_curve is None or brier_score_loss is None:
        print("\n⚠️  Calibration functions not available in this sklearn version")
        return None
    
    print(f"\n{'='*80}")
    print(f"📏 CALIBRATION ANALYSIS")
    print(f"{'='*80}")
    
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        brier_score = brier_score_loss(y_true, y_proba)
        
        plt.figure(figsize=(8, 8))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
                 label=model_name, linewidth=2, markersize=8)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title(f'Calibration Curve - {model_name}\nBrier Score: {brier_score:.4f}', 
                  fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'outputs/plots/{model_name}_calibration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Calibration Metrics for {model_name}:")
        print(f"   Brier Score: {brier_score:.4f} (lower is better)")
        print("✅ Saved calibration curve")
        
        return brier_score
    except Exception as e:
        print(f"⚠️  Calibration plot failed: {e}")
        return None


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline execution"""
    
    print("=" * 80)
    print("🫀 VITABEAT - COMPLETE ML PIPELINE")
    print("=" * 80)
    
    # Create output directories
    import os
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    # ========== LOAD DATA ==========
    heart_df, cardio_df, processed_df, ecg_df = load_datasets()
    
    if heart_df is None:
        print("\n❌ Required datasets not found. Please run: python data/generate_datasets.py")
        return
    
    # ========== EDA ==========
    perform_eda(heart_df, "Heart_Disease")
    
    if ecg_df is not None:
        visualize_ecg_samples(ecg_df)
    
    # ========== PREPARE TABULAR DATA ==========
    print(f"\n{'='*80}")
    print(f"🔧 PREPARING TABULAR DATA")
    print(f"{'='*80}")
    
    X = heart_df.drop('HeartDisease', axis=1)
    y = heart_df['HeartDisease']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train set: {X_train.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    # ========== TRAIN TABULAR MODELS ==========
    results, models, scaler = train_tabular_models(X_train, X_test, y_train, y_test)
    
    # ========== RISK SCORING ==========
    best_model = models.get('xgb') or models.get('rf') or models.get('lr')
    demonstrate_risk_scoring(best_model, X_test, feature_names, n_samples=5)
    
    # ========== SHAP EXPLANATIONS ==========
    if 'rf' in models and HAS_SHAP:
        explain_with_shap(models['rf'], X_train, X_test, feature_names)
    
    # ========== CALIBRATION ==========
    if 'xgb' in models:
        y_proba_xgb = models['xgb'].predict_proba(X_test)[:, 1]
        plot_calibration_curve_safe(y_test, y_proba_xgb, "XGBoost")
    elif 'rf' in models:
        y_proba_rf = models['rf'].predict_proba(X_test)[:, 1]
        plot_calibration_curve_safe(y_test, y_proba_rf, "Random_Forest")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 80)
    print("✅ VITABEAT PIPELINE COMPLETE!")
    print("=" * 80)
    print("\n📁 Output Summary:")
    print("   - Plots saved in: outputs/plots/")
    print("   - Models ready for deployment")
    print("\n🎯 Next Steps:")
    print("   1. Review generated plots in outputs/plots/")
    print("   2. Check model performance metrics above")
    print("   3. Prepare presentation with results")
    print("\n🏆 VitaBeat is ready for demo!")
    print("=" * 80)


if __name__ == "__main__":
    main()