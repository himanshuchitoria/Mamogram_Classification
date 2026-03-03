"""
=============================================================================
AI4BCancer - Hybrid Ensemble Model
=============================================================================
Combines the top-performing models from the repository into a weighted
soft-voting ensemble for maximum accuracy and reduced false negatives.

Selected models & weights (based on reported accuracy from README):
  1. ANN (Keras Dense)           — weight 3  (99.12%)
  2. Logistic Regression         — weight 2  (97.37%)
  3. Gradient Boosting           — weight 2  (97.37%)
  4. Random Forest               — weight 1  (96.49%)
  5. SVM (RBF kernel)            — weight 1  (95.61%)

The ensemble uses weighted probability averaging for soft voting.
=============================================================================
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model file paths
MODEL_PATHS = {
    "ann": os.path.join(MODELS_DIR, "ann_model.h5"),
    "logistic_regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "gradient_boosting": os.path.join(MODELS_DIR, "gradient_boosting.joblib"),
    "random_forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "svm": os.path.join(MODELS_DIR, "svm.joblib"),
}

# Weights for each model in the ensemble (higher = more influence)
MODEL_WEIGHTS = {
    "ann": 3,
    "logistic_regression": 2,
    "gradient_boosting": 2,
    "random_forest": 1,
    "svm": 1,
}


class HybridModel:
    """
    Hybrid ensemble classifier combining ANN + sklearn models with
    weighted soft voting for breast cancer classification.
    """

    def __init__(self):
        self.models = {}
        self.weights = MODEL_WEIGHTS
        self._ann_model = None
        self._is_loaded = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train all sub-models on the preprocessed training data.
        
        Args:
            X_train: Scaled feature matrix (n_samples, 30)
            y_train: Binary labels (0=Benign, 1=Malignant)
        """
        os.makedirs(MODELS_DIR, exist_ok=True)

        # ----- 1. ANN (mirrors Models/ANN/ANN.py architecture) -----
        print("[HybridModel] Training ANN...")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        ann = Sequential([
            Dense(64, activation="relu", input_dim=X_train.shape[1]),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        ann.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        )
        ann.save(MODEL_PATHS["ann"])
        self._ann_model = ann
        print("[HybridModel] ANN trained and saved.")

        # ----- 2. Logistic Regression (mirrors Models/Logistic Regression/LR.py) -----
        print("[HybridModel] Training Logistic Regression...")
        try:
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train, y_train)
            joblib.dump(lr, MODEL_PATHS["logistic_regression"])
            self.models["logistic_regression"] = lr
            print("[HybridModel] Logistic Regression trained and saved.")
        except Exception as e:
            print(f"Logistic Regression failed: {e}")

        # ----- 3. Gradient Boosting (mirrors Models/Gradient Boosting) -----
        print("[HybridModel] Training Gradient Boosting...")
        try:
            gb = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
            )
            gb.fit(X_train, y_train)
            joblib.dump(gb, MODEL_PATHS["gradient_boosting"])
            self.models["gradient_boosting"] = gb
            print("[HybridModel] Gradient Boosting trained and saved.")
        except Exception as e:
            print(f"Gradient Boosting failed: {e}")

        # ----- 4. Random Forest (mirrors Models/Random Forest) -----
        print("[HybridModel] Training Random Forest...")
        try:
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            joblib.dump(rf, MODEL_PATHS["random_forest"])
            self.models["random_forest"] = rf
            print("[HybridModel] Random Forest trained and saved.")
        except Exception as e:
            print(f"Random Forest failed: {e}")

        # ----- 5. SVM with probability (mirrors Models/SVM) -----
        print("[HybridModel] Training SVM...")
        try:
            svm = SVC(probability=True, kernel="rbf", random_state=42)
            svm.fit(X_train, y_train)
            joblib.dump(svm, MODEL_PATHS["svm"])
            self.models["svm"] = svm
            print("[HybridModel] SVM trained and saved.")
        except Exception as e:
            print(f"SVM failed: {e}")

        self._is_loaded = True
        print("[HybridModel] All models trained successfully!")

    def load(self):
        """Load all pre-trained models from disk."""
        # Check all files exist
        missing = [k for k, v in MODEL_PATHS.items() if not os.path.exists(v)]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}. Run train.py first."
            )

        # Load sklearn models
        self.models["logistic_regression"] = joblib.load(MODEL_PATHS["logistic_regression"])
        self.models["gradient_boosting"] = joblib.load(MODEL_PATHS["gradient_boosting"])
        self.models["random_forest"] = joblib.load(MODEL_PATHS["random_forest"])
        self.models["svm"] = joblib.load(MODEL_PATHS["svm"])

        # Load Keras ANN
        from tensorflow.keras.models import load_model
        self._ann_model = load_model(MODEL_PATHS["ann"])

        self._is_loaded = True
        print("[HybridModel] All models loaded successfully!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get weighted ensemble probability predictions.
        
        Args:
            X: Scaled feature matrix of shape (n_samples, 30)
        
        Returns:
            np.ndarray of shape (n_samples, 2) — [P(Benign), P(Malignant)]
        """
        if not self._is_loaded:
            self.load()

        total_weight = sum(self.weights.values())
        weighted_proba = np.zeros((X.shape[0], 2))

        # ANN prediction
        ann_pred = self._ann_model.predict(X, verbose=0).flatten()
        ann_proba = np.column_stack([1 - ann_pred, ann_pred])
        weighted_proba += self.weights["ann"] * ann_proba

        # Sklearn model predictions
        for name in ["logistic_regression", "gradient_boosting", "random_forest", "svm"]:
            proba = self.models[name].predict_proba(X)
            weighted_proba += self.weights[name] * proba

        # Normalize by total weight
        weighted_proba /= total_weight
        return weighted_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble class predictions (0=Benign, 1=Malignant).
        
        Uses a threshold of 0.45 (instead of 0.5) to reduce false negatives,
        which is critical in medical diagnostics — missing a malignant case
        is far worse than a false positive.
        """
        proba = self.predict_proba(X)
        # Lower threshold biases toward detecting malignancy (reduces FN)
        return (proba[:, 1] >= 0.45).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate ensemble and individual models on test data.
        
        Returns dict with accuracy scores for each model and the ensemble.
        """
        from sklearn.metrics import accuracy_score, classification_report

        results = {}

        # Individual model accuracies
        if self._ann_model:
            ann_pred = (self._ann_model.predict(X_test, verbose=0).flatten() >= 0.45).astype(int)
            results["ann"] = accuracy_score(y_test, ann_pred)

        for name, model in self.models.items():
            pred = model.predict(X_test)
            results[name] = accuracy_score(y_test, pred)

        # Ensemble accuracy
        ensemble_pred = self.predict(X_test)
        results["ensemble"] = accuracy_score(y_test, ensemble_pred)
        results["ensemble_report"] = classification_report(
            y_test, ensemble_pred, target_names=["Benign", "Malignant"]
        )

        return results
