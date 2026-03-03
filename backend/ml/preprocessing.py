"""
=============================================================================
AI4BCancer - Unified Preprocessing Pipeline
=============================================================================
Replicates the exact preprocessing from the repository's Preprocessing.py:
  1. Load Wisconsin Breast Cancer CSV data
  2. Drop 'id' and 'Unnamed: 32' columns
  3. Encode diagnosis: B=0, M=1 (LabelEncoder)
  4. Drop NaN rows
  5. StandardScaler on 30 numeric features
  
Also provides inference-time preprocessing for single samples.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants: The 30 features expected by all models in the repository
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

CLASS_NAMES = ["Benign", "Malignant"]

# Paths for saved artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "..", "Dataset", "data.csv")


def load_and_preprocess_dataset(file_path: str = None):
    """
    Load the Wisconsin Breast Cancer dataset and apply the exact same
    preprocessing pipeline used across all models in the repository.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    if file_path is None:
        file_path = DATASET_PATH

    # Step 1: Load CSV
    df = pd.read_csv(file_path)

    # Step 2: Drop unnecessary columns (matches repo Preprocessing.py)
    cols_to_drop = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Step 3: Encode diagnosis column (B=0, M=1)
    label_encoder = LabelEncoder()
    df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])

    # Step 4: Drop rows with any null values
    df = df.dropna()

    # Step 5: Split features and target
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # Step 6: Train/test split (same parameters as repo: 80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 7: StandardScaler (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the fitted scaler for inference
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, FEATURE_NAMES


def preprocess_single_sample(features: dict, scaler: StandardScaler = None):
    """
    Preprocess a single sample (30 features) for inference.
    
    Args:
        features: dict mapping feature name -> value for all 30 features
        scaler: fitted StandardScaler (loaded from disk if None)
    
    Returns:
        np.ndarray of shape (1, 30) — scaled feature vector
    
    Raises:
        ValueError: if any required feature is missing
    """
    # Load scaler from disk if not provided
    if scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. Run train.py first."
            )
        scaler = joblib.load(SCALER_PATH)

    # Validate all 30 features are present
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Build ordered feature vector
    feature_vector = np.array(
        [[features[f] for f in FEATURE_NAMES]], dtype=np.float64
    )

    # Apply the same scaling used during training
    scaled = scaler.transform(feature_vector)
    return scaled


def load_scaler() -> StandardScaler:
    """Load the pre-fitted StandardScaler from disk."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run train.py first."
        )
    return joblib.load(SCALER_PATH)
