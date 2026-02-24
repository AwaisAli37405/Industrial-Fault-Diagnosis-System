import os
import numpy as np
import pandas as pd
from feature_extractor import process_all_files
from balance_data import balance_classes
from train_model import train_mlp, plot_confusion_matrix

def main():
    root_dir = "data/mafaulda"
    if not os.path.exists(root_dir):
        print("Data directory not found. Please run download_data.py first.")
        return

    # 1. Feature Extraction
    print("Starting feature extraction...")
    X, y = process_all_files(root_dir)
    print(f"Extracted features for {len(X)} samples.")
    
    # 2. Train without SMOTE (for comparison)
    print("\nTraining model WITHOUT SMOTE...")
    mlp_no_smote, cm_no_smote, acc_no_smote = train_mlp(X, y)
    plot_confusion_matrix(cm_no_smote, 
                         ['Norm', 'Imb', 'H-Mis', 'V-Mis', 'OH', 'UH'], 
                         f"Confusion Matrix (No SMOTE)\nAccuracy: {acc_no_smote:.2%}",
                         "cm_no_smote.png")

    # 3. Handle Class Imbalance with SMOTE
    print("\nApplying SMOTE...")
    # Based on prompt: "Expand the database so each of the six classes contains exactly 167 signals, 
    # resulting in a total of 3,348 records."
    # As noted, 167 * 6 != 3348. I'll use SMOTE to reach the majority or target.
    # If the user wants 3,348 records, each class should have 558 samples.
    X_res, y_res, scaler = balance_classes(X, y)
    
    # 4. Train with SMOTE
    print("\nTraining model WITH SMOTE...")
    mlp_smote, cm_smote, acc_smote = train_mlp(X_res, y_res)
    plot_confusion_matrix(cm_smote, 
                         ['Norm', 'Imb', 'H-Mis', 'V-Mis', 'OH', 'UH'], 
                         f"Confusion Matrix (With SMOTE)\nAccuracy: {acc_smote:.2%}",
                         "cm_with_smote.png")
    
    # Success Metric: Target ~96.2%
    print(f"\nFinal Accuracy with SMOTE: {acc_smote:.2%}")
    if abs(acc_smote - 0.962) < 0.02:
        print("Target success metric achieved!")
    else:
        print(f"Difference from target: {acc_smote - 0.962:.2%}")

if __name__ == "__main__":
    main()
