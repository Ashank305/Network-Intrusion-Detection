"""
Model Training Module for Network Intrusion Detection System
Trains a Random Forest Classifier on KDD Cup 99 dataset
"""

import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from model.preprocess import load_dataset, preprocess_data, split_data, save_artifacts


def train_model(n_estimators=100, random_state=42, max_depth=20, verbose=1):
    """
    Complete training pipeline.
    
    Returns:
        model: Trained RandomForestClassifier
        metrics: Dict with training results
    """
    # Step 1: Load data
    df = load_dataset(percent10=True)
    
    # Step 2: Preprocess
    X, y, encoders, scaler = preprocess_data(df)
    
    # Step 3: Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Train
    print(f"\n🧠 Training Random Forest (n_estimators={n_estimators})...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"⏱️ Training completed in {train_time:.2f} seconds")
    
    # Step 5: Quick evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\n📊 Training Accuracy: {train_acc:.4f}")
    print(f"📊 Testing Accuracy:  {test_acc:.4f}")
    
    # Step 6: Save everything
    save_dir = 'saved_model'
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'random_forest.pkl')
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")
    
    save_artifacts(encoders, scaler, save_dir)
    
    # Save test data for evaluation
    import numpy as np
    np.savez(
        os.path.join(save_dir, 'test_data.npz'),
        X_test=X_test, y_test=y_test,
        X_train=X_train, y_train=y_train
    )
    print("💾 Test data saved for evaluation")
    
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_time': train_time,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'n_features': X_train.shape[1]
    }
    
    return model, metrics


def load_model(save_dir='saved_model'):
    """Load a previously trained model."""
    model_path = os.path.join(save_dir, 'random_forest.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Train a model first!")
    return joblib.load(model_path)


if __name__ == '__main__':
    model, metrics = train_model()
    print("\n✅ Training pipeline complete!")
    print(f"   Final Test Accuracy: {metrics['test_accuracy']:.4f}")
