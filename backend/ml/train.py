"""
=============================================================================
AI4BCancer - Model Training Script
=============================================================================
Trains all sub-models in the hybrid ensemble and saves weights to disk.
Run this once before starting the web application:

    cd backend
    python -m ml.train
=============================================================================
"""

import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing import load_and_preprocess_dataset, MODELS_DIR
from ml.hybrid_model import HybridModel


def main():
    """Train the hybrid ensemble model and save all weights."""
    print("=" * 60)
    print("  AI4BCancer - Training Hybrid Ensemble Model")
    print("=" * 60)

    # Step 1: Load and preprocess dataset
    print("\n[Step 1] Loading and preprocessing dataset...")
    start = time.time()
    X_train, X_test, y_train, y_test, scaler, feature_names = (
        load_and_preprocess_dataset()
    )
    print(f"  Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"  Features: {len(feature_names)}")
    print(f"  Scaler saved to: {os.path.join(MODELS_DIR, 'scaler.joblib')}")
    print(f"  Time: {time.time() - start:.1f}s")

    # Step 2: Train hybrid ensemble
    print("\n[Step 2] Training all ensemble sub-models...")
    model = HybridModel()
    start = time.time()
    model.train(X_train, y_train.values if hasattr(y_train, 'values') else y_train)
    print(f"  Total training time: {time.time() - start:.1f}s")

    # Step 3: Evaluate on test set
    print("\n[Step 3] Evaluating models on test set...")
    results = model.evaluate(
        X_test, y_test.values if hasattr(y_test, 'values') else y_test
    )

    print("\n  Individual Model Accuracies:")
    print("  " + "-" * 40)
    for name, acc in results.items():
        if name not in ("ensemble", "ensemble_report"):
            print(f"  {name:25s}: {acc:.4f} ({acc*100:.2f}%)")

    print(f"\n  {'ENSEMBLE':25s}: {results['ensemble']:.4f} ({results['ensemble']*100:.2f}%)")
    print(f"\n  Classification Report:\n{results['ensemble_report']}")

    # Step 4: Save training data summary for explainability
    import numpy as np
    import joblib
    training_data_path = os.path.join(MODELS_DIR, "training_data_sample.joblib")
    joblib.dump(X_train, training_data_path)
    print(f"\n[Step 4] Training data sample saved for XAI: {training_data_path}")

    print("\n" + "=" * 60)
    print("  Training complete! All models saved to:", MODELS_DIR)
    print("  You can now start the server: uvicorn main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
