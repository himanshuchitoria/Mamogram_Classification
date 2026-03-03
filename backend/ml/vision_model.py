"""
=============================================================================
AI4BCancer - True Deep Learning Vision Model
=============================================================================
Implements a state-of-the-art Convolutional Neural Network (CNN) specifically
designed to classify raw mammogram images, fully bypassing the tabular models
that were designed strictly for FNA cell nuclei data.

Architecture:
- Base: DenseNet121 (pretrained on ImageNet)
  Dense connections excel at extracting complex texture/margin features in 
  medical imaging and mitigating vanishing gradients.
- Head: GlobalAveragePooling2D -> Dropout(0.5) -> Dense(1, Sigmoid)

Note: For true clinical accuracy, this model must be fine-tuned on a robust
dataset of local mammogram images (not just `data.csv`).
=============================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Tuple

# Define the local paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISION_MODEL_PATH = os.path.join(MODELS_DIR, "vision_densenet.keras")

# DenseNet expects 224x224 RGB images
IMG_SIZE = (224, 224)


class VisionModel:
    """
    A true CNN model for mammogram classification.
    """
    def __init__(self):
        self._model = None
        self._is_loaded = False
        self.img_size = IMG_SIZE

    def build_model(self, learning_rate: float = 1e-4) -> None:
        """
        Constructs the DenseNet121 transfer learning architecture.
        """
        print("[VisionModel] Constructing DenseNet121 architecture...")
        # 1. Base Pre-trained Model (Freeze weights initially for transfer learning)
        base_model = DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        base_model.trainable = False  # Freeze convolutional base

        # 2. Classification Head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)  # Prevent overfitting
        predictions = Dense(1, activation='sigmoid')(x) # Binary classification (Benign=0, Malig=1)
        
        # 3. Compile Model
        self._model = Model(inputs=base_model.input, outputs=predictions)
        self._model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self._is_loaded = True
        print("[VisionModel] Model constructed and compiled.")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocesses raw image bytes into the exact tensor format
        expected by DenseNet121.
        """
        import cv2

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode uploaded image. Ensure it is valid.")

        # Resize to exactly 224x224
        img_resized = cv2.resize(img, self.img_size)

        # Convert to float and apply DenseNet specific prep preprocessing
        # Usually it expects inputs scaled to [0, 1] or bounded differently based on tf.keras.applications.densenet.preprocess_input
        # We'll use the official preprocess function:
        img_float = np.expand_dims(img_resized, axis=0) # Add batch dimension -> (1, 224, 224, 3)
        img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_float)
        
        return img_preprocessed

    def load(self) -> bool:
        """
        Attempts to load a fine-tuned model from disk.
        If it doesn't exist, it builds the generic pre-trained architecture.
        """
        if os.path.exists(VISION_MODEL_PATH):
            try:
                self._model = load_model(VISION_MODEL_PATH)
                self._is_loaded = True
                print("[VisionModel] Successfully loaded fine-tuned model weights.")
                return True
            except Exception as e:
                print(f"[VisionModel] Failed to load model weights: {e}")
                self.build_model()
                return False
        else:
            print("[VisionModel] No fine-tuned weights found on disk. Building generic untrained architecture.")
            self.build_model()
            return False

    def predict_proba(self, image_tensor: np.ndarray) -> np.ndarray:
        """
        Returns the [P(Benign), P(Malignant)] probabilities.
        """
        if not self._is_loaded:
            raise RuntimeError("VisionModel is not loaded.")
            
        # Neural net returns a single sigmoid probability P(Malignant)
        p_malignant = self._model.predict(image_tensor, verbose=0)[0][0]
        
        # We dynamically cast it to the [P(Benign), P(Malignant)] structure expected by the API
        p_benign = 1.0 - p_malignant
        
        return np.array([[p_benign, p_malignant]])

    def save(self) -> None:
        """Saves current weights to disk."""
        if self._model:
            os.makedirs(MODELS_DIR, exist_ok=True)
            self._model.save(VISION_MODEL_PATH)
            print(f"[VisionModel] Saved to {VISION_MODEL_PATH}")
