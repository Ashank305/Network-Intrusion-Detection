"""
Data Preprocessing Module for Network Intrusion Detection System
Uses KDD Cup 1999 dataset from sklearn
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# KDD Cup 99 column names (41 features + label)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Attack type mapping for multi-class
ATTACK_CATEGORY_MAP = {
    b'normal.': 'Normal',
    b'back.': 'DoS', b'land.': 'DoS', b'neptune.': 'DoS', b'pod.': 'DoS',
    b'smurf.': 'DoS', b'teardrop.': 'DoS',
    b'ipsweep.': 'Probe', b'nmap.': 'Probe', b'portsweep.': 'Probe',
    b'satan.': 'Probe',
    b'ftp_write.': 'R2L', b'guess_passwd.': 'R2L', b'imap.': 'R2L',
    b'multihop.': 'R2L', b'phf.': 'R2L', b'spy.': 'R2L',
    b'warezclient.': 'R2L', b'warezmaster.': 'R2L',
    b'buffer_overflow.': 'U2R', b'loadmodule.': 'U2R', b'perl.': 'U2R',
    b'rootkit.': 'U2R'
}


def load_dataset(subset='SA', percent10=True):
    """
    Load KDD Cup 99 dataset from sklearn.
    
    Args:
        subset: 'SA' for HTTP subset, None for full (use percent10 for speed)
        percent10: If True, load 10% subset (faster for training)
    
    Returns:
        DataFrame with all features and labels
    """
    print("📥 Loading KDD Cup 99 dataset...")
    data = fetch_kddcup99(subset=None, percent10=percent10)
    
    df = pd.DataFrame(data.data, columns=COLUMN_NAMES)
    df['label'] = data.target
    
    # Decode byte strings to regular strings
    for col in CATEGORICAL_COLS:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_binary_label(df):
    """Convert multi-class labels to binary: Normal (0) vs Attack (1)"""
    df = df.copy()
    df['binary_label'] = df['label'].apply(
        lambda x: 0 if x == b'normal.' else 1
    )
    return df


def create_multiclass_label(df):
    """Map attack types to categories: Normal, DoS, Probe, R2L, U2R"""
    df = df.copy()
    df['attack_category'] = df['label'].map(ATTACK_CATEGORY_MAP).fillna('Unknown')
    return df


def preprocess_data(df, fit_encoders=True, encoders=None, scaler=None):
    """
    Full preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        fit_encoders: If True, fit new encoders. If False, use provided ones.
        encoders: Dict of pre-fitted LabelEncoders (for prediction)
        scaler: Pre-fitted StandardScaler (for prediction)
    
    Returns:
        X: Feature matrix (scaled)
        y: Target labels
        encoders: Dict of fitted LabelEncoders
        scaler: Fitted StandardScaler
    """
    df = df.copy()
    
    # Create binary labels
    df = create_binary_label(df)
    
    # Encode categorical features
    if fit_encoders:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            # Handle unseen categories gracefully
            df[col] = df[col].astype(str)
            known = set(encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known else encoders[col].classes_[0])
            df[col] = encoders[col].transform(df[col])
    
    # Separate features and target
    feature_cols = [c for c in COLUMN_NAMES]
    X = df[feature_cols].values.astype(np.float64)
    y = df['binary_label'].values
    
    # Scale features
    if fit_encoders:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, encoders, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def save_artifacts(encoders, scaler, save_dir='saved_model'):
    """Save encoders and scaler for later use."""
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(encoders, os.path.join(save_dir, 'encoders.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    print(f"💾 Artifacts saved to {save_dir}/")


def load_artifacts(save_dir='saved_model'):
    """Load saved encoders and scaler."""
    encoders = joblib.load(os.path.join(save_dir, 'encoders.pkl'))
    scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
    return encoders, scaler


if __name__ == '__main__':
    df = load_dataset()
    X, y, encoders, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_artifacts(encoders, scaler)
    print("✅ Preprocessing complete!")
