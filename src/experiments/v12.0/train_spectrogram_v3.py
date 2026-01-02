# Save this as train_spectrogram_v3.py
# This script trains our new 2D-CNN model (v6)
# on the final spectrogram dataset (tfrecord_dataset_v3_spectrogram).

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import glob
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Data ---
    TRAIN_TFRECORD_PATH = "tfrecord_dataset_v3_spectrogram/train/*.tfrecord"
    VAL_TFRECORD_PATH = "tfrecord_dataset_v3_spectrogram/val/*.tfrecord"
    
    # --- Model Hyperparameters (from our data conversion) ---
    # These shapes are based on the STFT parameters in the conversion script
    # Input shape: (n_freqs, n_times, n_channels)
    # (101, 99, 12)
    INPUT_SHAPE = (101, 101, 12)
    NUM_TARGETS = 7 # 7 Virtual Motors
    
    # --- Training ---
    BATCH_SIZE = 16 
    EPOCHS = 150    
    
    # --- From Log File ---
    TOTAL_TRAIN_SEQS = 1903
    TOTAL_VAL_SEQS = 475

# --- 1. DATA LOADER ---

def parse_spectrogram_tfrecord(example_proto):
    """
    Parses the TFRecord examples created by convert_to_spectrogram.py.
    """
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        'target_vector': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    # Spectrogram was saved as (12, 101, 99) -> (C, H, W)
    spectrogram = tf.io.parse_tensor(parsed['spectrogram'], out_type=tf.float32)
    
    # Target was saved as (7,)
    target = tf.io.parse_tensor(parsed['target_vector'], out_type=tf.float32)
    
    # *** CRITICAL: Transpose the spectrogram ***
    # Keras Conv2D expects (Height, Width, Channels)
    # We convert (C, H, W) -> (H, W, C)
    # (12, 101, 99) -> (101, 99, 12)
    spectrogram = tf.transpose(spectrogram, (1, 2, 0))
    
    # Set the shapes explicitly
    spectrogram = tf.reshape(spectrogram, Config.INPUT_SHAPE)
    target = tf.reshape(target, [Config.NUM_TARGETS])
    
    return spectrogram, target

def create_dataset(tfrecord_pattern: str, batch_size: int, shuffle=True):
    """Creates a high-performance TensorFlow dataset from TFRecord files."""
    
    file_list = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    dataset = file_list.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000) # Buffer size matches our dataset
        
    dataset = dataset.map(parse_spectrogram_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    if shuffle:
        dataset = dataset.repeat() # Repeat indefinitely for training
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. MODEL ARCHITECTURE (v6) ---

def create_v6_cnn_spectrogram(model_name="v6_cnn_spectrogram"):
    """
    Model v6: A 2D-CNN for End-to-End Spectrogram Regression.
    This is a Seq2Vec model: (5-sec image) -> (1 final prediction)
    """
    inputs = layers.Input(shape=Config.INPUT_SHAPE, name="emg_spectrogram_input")
    
    # 1. CNN Feature Extraction Block
    # We treat the (101, 99, 12) input as an "image"
    
    # Initial "stem"
    x = layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 2. Fully-Connected Prediction Head
    # Flatten the 2D features into a 1D vector
    x = layers.Flatten()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Final output layer
    outputs = layers.Dense(Config.NUM_TARGETS, activation='sigmoid', name="motor_output_vector")(x)
    # We use 'sigmoid' because our targets are all normalized to 0-1
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Start with a low LR
        loss='mse', # Mean Squared Error
        metrics=['mae'] # Mean Absolute Error
    )
    return model

# --- 3. TRAINING & EVALUATION ---

def setup_experiment_directory(experiment_name):
    """Creates a unique directory for storing all artifacts of a training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def train_model(model_creator, model_name, train_dataset, val_dataset, train_steps, val_steps):
    """The main function to train a single model and save all results."""
    print(f"\n{'='*60}\n🚀 STARTING TRAINING FOR {model_name}\n{'='*60}")
    
    exp_dir = setup_experiment_directory(f"series_{model_name}")
    model = model_creator()
    model.summary()
    
    # Callbacks
    best_model_path = os.path.join(exp_dir, "models", f"best_{model_name}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
    ]

    print(f"\nTraining {model_name}...")
    start_time = datetime.datetime.now()

    history = model.fit(
        train_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    
    # Load best model and get its validation MAE
    best_val_mae = min(history.history['val_mae'])
    
    # Save results
    results = {
        'model_version': model_name,
        'best_validation_mae': float(best_val_mae),
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_mae']) + 1),
        'training_time_seconds': training_time.total_seconds(),
        'model_architecture': json.loads(model.to_json())
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create and save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE'); plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f'{model_name}_training_history.png'))
    plt.close()

    print(f"\n🎯 {model_name} - FINAL RESULTS:")
    print(f"   Best Validation MAE: {results['best_validation_mae']:.4f}")
    print(f"   Training Time: {str(training_time).split('.')[0]}")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"💾 Results and artifacts saved to: {exp_dir}")
    return results

# --- 4. MAIN EXECUTION ---
def main():
    print("🚀 STARTING SPECTROGRAM MODEL TRAINING (PHASE 3b)")
    print("=============================================================")
    
    # Find the data we just created
    train_files = glob.glob(Config.TRAIN_TFRECORD_PATH)
    val_files = glob.glob(Config.VAL_TFRECORD_PATH)
    
    if not train_files or not val_files:
        print(f"❌ FATAL: Could not find TFRecord files.")
        print(f"   Checked for train: {Config.TRAIN_TFRECORD_PATH}")
        print(f"   Checked for val: {Config.VAL_TFRECORD_PATH}")
        return
    
    # Calculate steps per epoch
    train_steps = Config.TOTAL_TRAIN_SEQS // Config.BATCH_SIZE
    val_steps = Config.TOTAL_VAL_SEQS // Config.BATCH_SIZE
    
    if train_steps == 0: train_steps = 1
    if val_steps == 0: val_steps = 1

    print(f"   Found {Config.TOTAL_TRAIN_SEQS} train sequences and {Config.TOTAL_VAL_SEQS} val sequences.")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Steps/Epoch: {train_steps}")
    print(f"   Validation Steps/Epoch: {val_steps}")

    # Create the datasets
    train_dataset = create_dataset(Config.TRAIN_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(Config.VAL_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=False)

    # Train the single v6 model
    results = train_model(
        create_v6_cnn_spectrogram, 
        "v6_cnn_spectrogram", 
        train_dataset, 
        val_dataset, 
        train_steps, 
        val_steps
    )

    print(f"\n{'='*70}\n📊 END-TO-END EXPERIMENT COMPLETE 📊\n{'='*70}")
    
    print(f"{'Model':<25} {'Best Val MAE':<15}")
    print('-'*70)
    print(f"{results['model_version']:<25} {results['best_validation_mae']:.4f}")
    
    print(f"\n   Recall: Our 'Sequential' (v4) model had a Val MAE of 0.1149")
    
    if results['best_validation_mae'] < 0.1149:
        print(f"\n   🏆🏆🏆 NEW CHAMPION! The End-to-End model is the best one so far! 🏆🏆🏆")
    else:
        print(f"\n   ... The Sequential (v4) model remains the champion.")


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()