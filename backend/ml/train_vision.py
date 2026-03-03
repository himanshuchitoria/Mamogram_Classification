"""
=============================================================================
AI4BCancer - Vision Model Training Script
=============================================================================
Trains the DenseNet121 Deep Learning vision pipeline on local mammogram images.

Expected Directory Structure:
Dataset/
  Images/
    Benign/      <-- Put benign mammograms here
    Malignant/   <-- Put malignant mammograms here

This script uses data augmentation (flips, rotations, zoom) to train a robust
medical imaging classifier even with limited data.
=============================================================================
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.vision_model import VisionModel, IMG_SIZE, VISION_MODEL_PATH

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Dataset", "Images")

def main():
    print("=" * 60)
    print("  AI4BCancer - Training Vision Model (DenseNet121)")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"\n[ERROR] Image dataset directory not found at: {DATA_DIR}")
        print("Please create 'Dataset/Images/Benign' and 'Dataset/Images/Malignant'")
        print("and populate them with mammogram images before running this script.")
        sys.exit(1)

    # 1. Initialize the Vision Model
    vm = VisionModel()
    vm.build_model(learning_rate=1e-4)
    model = vm._model

    # 2. Setup Data Generators with Medical Augmentation
    print("\n[Step 1] Initializing Data Generators...")
    
    # Mammograms don't have color, but DenseNet expects 3 channels. 
    # The generator will duplicate the grayscale channel to RGB.
    # We apply robust morphological augmentations common in medical imaging.
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.densenet.preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True, # Breasts can be imaged in different orientations (CC/MLO)
        validation_split=0.2 # 80/20 train/val split
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary', # [0]=Benign, [1]=Malignant
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )

    if train_generator.samples == 0:
        print("\n[ERROR] No images found in dataset folders.")
        sys.exit(1)

    print(f"Class mapping: {train_generator.class_indices}")

    # 3. Setup Callbacks
    os.makedirs(os.path.dirname(VISION_MODEL_PATH), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=VISION_MODEL_PATH,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 4. Train the Model
    print("\n[Step 2] Training Model (Transfer Learning Head)...")
    epochs = 30
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    # 5. Fine-tuning Phase (Unfreeze top convolutional blocks)
    print("\n[Step 3] Fine-tuning the Base Model...")
    # Unfreeze the base model
    model.layers[1].trainable = True
    
    # Freeze the first 100 layers (generic features), unfreeze the rest (specific textures)
    for layer in model.layers[1].layers[:100]:
        layer.trainable = False

    # Recompile with a MUCH lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Continue training
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20, # 20 more epochs of fine-tuning
        callbacks=callbacks
    )

    print("\n" + "=" * 60)
    print("  Vision Model Training Complete!")
    print(f"  Best weights saved to: {VISION_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
