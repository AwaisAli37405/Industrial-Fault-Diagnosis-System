import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis

def calculate_entropy(signal, bins=100):
    """Estimate entropy using histogram approach."""
    # Using np.histogram for efficiency
    p, _ = np.histogram(signal, bins=bins, density=True)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def extract_features_from_file(filepath):
    """Extract 25 features from a single MAFAULDA CSV file."""
    # Data columns: 1 tachometer, 6 accelerometer, 1 microphone
    # Total 8 signals.
    try:
        data = pd.read_csv(filepath, header=None).values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Step 1: Rotation Frequency (fr) from Tachometer (Column 0)
    tach_signal = data[:, 0]
    sampling_rate = 50000
    n = len(tach_signal)
    
    # We use DFT to find the dominant frequency in the tachometer signal
    # The tachometer usually gives pulses per revolution.
    # In MAFAULDA, it's often 1 pulse per revolution or similar.
    yf = fft(tach_signal - np.mean(tach_signal))
    xf = fftfreq(n, 1 / sampling_rate)
    idx = np.argmax(np.abs(yf[:n//2]))
    fr = np.abs(xf[idx])
    
    features = [fr]
    
    # Step 2: Mean, Entropy, Kurtosis for each of the 8 signals
    for i in range(8):
        signal = data[:, i]
        m = np.mean(signal)
        k = kurtosis(signal)
        e = calculate_entropy(signal)
        features.extend([m, e, k])
        
    return np.array(features)

def process_all_files(root_dir):
    """Process all 1,951 files and return features and labels."""
    all_features = []
    labels = []
    
    categories = {
        'normal': 0,
        'imbalance': 1,
        'horizontal-misalignment': 2,
        'vertical-misalignment': 3,
        'overhang': 4,
        'underhang': 5
    }
    
    # Mapping based on MAFAULDA directory structure
    dir_map = {
        'normal': 'normal',
        'imbalance': 'imbalance',
        'horizontal-misalignment': 'horizontal-misalignment',
        'vertical-misalignment': 'vertical-misalignment',
        'overhang': 'bearing/overhang',
        'underhang': 'bearing/underhang'
    }
    
    for cat_name, label in categories.items():
        cat_dir = os.path.join(root_dir, dir_map[cat_name])
        if not os.path.exists(cat_dir):
            print(f"Warning: Directory {cat_dir} not found.")
            continue
            
        for root, dirs, files in os.walk(cat_dir):
            for file in files:
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    feat = extract_features_from_file(filepath)
                    if feat is not None:
                        all_features.append(feat)
                        labels.append(label)
                        
    return np.array(all_features), np.array(labels)

if __name__ == "__main__":
    # Test on a single file if available
    test_file = "data/mafaulda/normal/12.288.csv"
    if os.path.exists(test_file):
        f = extract_features_from_file(test_file)
        print(f"Features: {f}")
        print(f"Total features: {len(f)}")
