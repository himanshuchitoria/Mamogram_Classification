"""
=============================================================================
AI4BCancer - Explainable AI (XAI) Module
=============================================================================
Provides LIME and SHAP-based explanations for model predictions, consistent
with the repository's Models with XAI folder (LIME + SHAP approach).

Generates:
  - LIME feature importance for individual predictions
  - SHAP feature importance (global + local)
  - Matplotlib visualizations as base64-encoded images
=============================================================================
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

from .preprocessing import FEATURE_NAMES, CLASS_NAMES


class ExplainabilityEngine:
    """
    Generates LIME and SHAP explanations for breast cancer predictions.
    """

    def __init__(self, training_data: np.ndarray):
        """
        Initialize with scaled training data for LIME background.
        
        Args:
            training_data: Scaled X_train array of shape (n_samples, 30)
        """
        self.training_data = training_data

        # Set up LIME explainer (mirrors Models with XAI/Logistic Regression/LR with XAI.py)
        self.lime_explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=FEATURE_NAMES,
            class_names=CLASS_NAMES,
            mode="classification",
            discretize_continuous=True,
        )

    def explain_lime(self, sample: np.ndarray, predict_fn, num_features: int = 15):
        """
        Generate LIME explanation for a single prediction.
        
        Args:
            sample: Scaled feature vector of shape (30,)
            predict_fn: Model's predict_proba function
            num_features: Number of top features to show
        
        Returns:
            dict with keys:
              - feature_importance: list of (feature_name, weight) tuples
              - plot_base64: base64-encoded PNG of the explanation chart
              - top_features: top 5 most important features with explanations
        """
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            sample,
            predict_fn,
            num_features=num_features,
        )

        # Extract feature contributions
        feature_importance = explanation.as_list()

        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance, key=lambda x: abs(x[1]), reverse=True
        )

        # Generate visualization
        plot_base64 = self._generate_lime_plot(sorted_features)

        # Top 5 features with medical context
        top_features = []
        for feat_name, weight in sorted_features[:5]:
            top_features.append({
                "feature": feat_name,
                "weight": round(float(weight), 4),
                "direction": "Malignant" if weight > 0 else "Benign",
                "impact": "high" if abs(weight) > 0.1 else "moderate" if abs(weight) > 0.05 else "low",
            })

        return {
            "feature_importance": [
                {"feature": f, "weight": round(float(w), 4)} for f, w in sorted_features
            ],
            "plot_base64": plot_base64,
            "top_features": top_features,
        }

    def explain_shap(self, sample: np.ndarray, predict_fn):
        """
        Generate SHAP explanation for a single prediction.
        
        Uses KernelSHAP for model-agnostic explanations.
        
        Args:
            sample: Scaled feature vector of shape (1, 30)
            predict_fn: Model's predict_proba function
        
        Returns:
            dict with shap_values and plot_base64
        """
        try:
            import shap

            # Use a small background dataset for KernelSHAP (50 samples)
            bg_size = min(50, len(self.training_data))
            background = shap.sample(self.training_data, bg_size)

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(sample)

            # Generate SHAP bar plot
            plot_base64 = self._generate_shap_plot(shap_values, sample)

            # Extract per-feature SHAP values for the malignant class
            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # Class 1 (Malignant)
            else:
                sv = shap_values[0]

            feature_shap = [
                {"feature": FEATURE_NAMES[i], "shap_value": round(float(sv[i]), 4)}
                for i in range(len(FEATURE_NAMES))
            ]
            feature_shap.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            return {
                "feature_shap": feature_shap,
                "plot_base64": plot_base64,
            }
        except Exception as e:
            # SHAP can be slow/memory intensive; gracefully degrade
            print(f"[XAI] SHAP explanation failed: {e}")
            return {
                "feature_shap": [],
                "plot_base64": "",
                "error": str(e),
            }

    def _generate_lime_plot(self, sorted_features: list) -> str:
        """Create a horizontal bar chart of LIME feature importances."""
        fig, ax = plt.subplots(figsize=(10, 7))

        features = [f[0] for f in sorted_features[:15]][::-1]
        weights = [f[1] for f in sorted_features[:15]][::-1]
        colors = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]

        ax.barh(features, weights, color=colors, edgecolor="#333", linewidth=0.5)
        ax.set_xlabel("Feature Contribution", fontsize=12, fontweight="bold")
        ax.set_title("LIME Feature Importance", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="#333", linewidth=0.8, linestyle="--")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="→ Malignant"),
            Patch(facecolor="#2ecc71", label="→ Benign"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _generate_shap_plot(self, shap_values, sample: np.ndarray) -> str:
        """Create a SHAP waterfall-style bar chart."""
        try:
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            fig, ax = plt.subplots(figsize=(10, 7))

            # Sort by absolute value
            indices = np.argsort(np.abs(sv))[-15:]
            features = [FEATURE_NAMES[i] for i in indices]
            values = [sv[i] for i in indices]
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

            ax.barh(features, values, color=colors, edgecolor="#333", linewidth=0.5)
            ax.set_xlabel("SHAP Value (impact on Malignant prediction)", fontsize=12)
            ax.set_title("SHAP Feature Importance", fontsize=14, fontweight="bold")
            ax.axvline(x=0, color="#333", linewidth=0.8, linestyle="--")

            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception:
            return ""

    @staticmethod
    def _fig_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
