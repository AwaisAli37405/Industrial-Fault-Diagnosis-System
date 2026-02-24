import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_mlp(X, y):
    """
    Train an MLP with 25-10-6 architecture.
    - Input: 25 neurons
    - Hidden: 1 layer with 10 neurons
    - Output: 6 neurons
    - Activation: tanh
    - Split: 70/10/20
    """
    # 70% train, 30% temp (which will be split into 10% validation and 20% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # Of the 30% temp: 1/3 is 10%, 2/3 is 20%.
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # MLP Architecture: 25-10-6
    # scikit-learn's MLPClassifier automatically sets output layer based on classes
    mlp = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation='tanh',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(mlp, X_train, y_train, cv=kf)
    print(f"CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Final training on train+val or just train? 
    # Usually, we tune on val and then re-train.
    mlp.fit(X_train, y_train)
    
    # Evaluation
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")
    
    return mlp, cm, acc

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Test with dummy data
    X = np.random.rand(1002, 25)
    y = np.random.randint(0, 6, 1002)
    train_mlp(X, y)
