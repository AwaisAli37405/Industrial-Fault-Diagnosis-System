import numpy as np
import pandas as pd
import os
from balance_data import balance_classes
from train_model import train_mlp, plot_confusion_matrix

def create_synthetic_mafaulda_data(n_files=100, n_samples=10000):
    """Create synthetic data to test the pipeline."""
    if not os.path.exists("data/synthetic/normal"):
        os.makedirs("data/synthetic/normal")
    if not os.path.exists("data/synthetic/imbalance"):
        os.makedirs("data/synthetic/imbalance")
        
    for i in range(n_files):
        # 8 channels: tachometer, 6 accelerometer, 1 microphone
        data = np.random.randn(n_samples, 8)
        # Add some patterns for 'normal'
        if i < n_files // 2:
            data[:, 0] = np.sin(np.linspace(0, 50 * np.pi, n_samples)) # Tachometer pulse
            filepath = f"data/synthetic/normal/file_{i}.csv"
        else:
            data[:, 0] = np.sin(np.linspace(0, 45 * np.pi, n_samples)) # Different freq
            filepath = f"data/synthetic/imbalance/file_{i}.csv"
            
        pd.DataFrame(data).to_csv(filepath, index=False, header=False)

def run_mock_pipeline():
    print("Creating synthetic data...")
    create_synthetic_mafaulda_data(n_files=20)
    
    # We'll use our existing extraction logic but mock the directory structure
    from feature_extractor import process_all_files
    
    print("Processing synthetic files...")
    # Temporarily point to synthetic data
    X, y = process_all_files("data/synthetic")
    
    print(f"Features shape: {X.shape}, labels shape: {y.shape}")
    
    print("\nTraining WITHOOUT SMOTE...")
    mlp_no_smote, cm_no_smote, acc_no_smote = train_mlp(X, y)
    
    print("\nApplying SMOTE...")
    # Use our balance logic
    X_res, y_res, scaler = balance_classes(X, y)
    
    print("\nTraining WITH SMOTE...")
    mlp_smote, cm_smote, acc_smote = train_mlp(X_res, y_res)
    
    print(f"\nMock Pipeline Success! Acc with SMOTE: {acc_smote:.2%}")

if __name__ == "__main__":
    run_mock_pipeline()
