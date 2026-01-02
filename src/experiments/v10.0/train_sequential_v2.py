# Save this as train_sequential_v2.py
# This script trains our new sequential models (v4, v5)
# on the final TFRecord dataset.

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
    TRAIN_TFRECORD_PATH = "tfrecord_dataset_v2/train/*.tfrecord"
    VAL_TFRECORD_PATH = "tfrecord_dataset_v2/val/*.tfrecord"
    
    # --- Model Hyperparameters (from our data conversion) ---
    SEQUENCE_LENGTH = 50 # 50 steps (5 seconds)
    NUM_FEATURES = 48    # 48 Hudgins' features (12 channels * 4)
    NUM_TARGETS = 7      # 7 Virtual Motors
    
    # --- Training ---
    BATCH_SIZE = 16 # Small dataset, so small batch size
    EPOCHS = 150    # Let's train for a while; EarlyStopping will find the best
    
# --- 1. DATA LOADER ---

def parse_sequence_tfrecord(example_proto):
    """
    Parses the TFRecord examples created by convert_to_tfrecord_v2.py.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'targets': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    features = tf.io.parse_tensor(parsed['features'], out_type=tf.float32)
    targets = tf.io.parse_tensor(parsed['targets'], out_type=tf.float32)
    
    # Set the shapes explicitly
    features = tf.reshape(features, [Config.SEQUENCE_LENGTH, Config.NUM_FEATURES])
    targets = tf.reshape(targets, [Config.SEQUENCE_LENGTH, Config.NUM_TARGETS])
    
    return features, targets

def create_dataset(tfrecord_pattern: str, batch_size: int, shuffle=True):
    """Creates a high-performance TensorFlow dataset from TFRecord files."""
    
    file_list = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    dataset = file_list.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        # Shuffle *before* mapping for better performance
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.map(parse_sequence_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    if shuffle:
        # Repeat indefinitely for training
        dataset = dataset.repeat()
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. MODEL ARCHITECTURES ---

def create_v4_lstm_baseline(model_name="v4_lstm_baseline"):
    """
    Model v4: A simple LSTM-only model. This is our sequential baseline.
    This is a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # LSTM layer that processes the sequence.
    # return_sequences=True is ESSENTIAL for Seq2Seq. It outputs
    # a prediction for *every* of the 50 time steps.
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    
    # A final Dense layer, wrapped in TimeDistributed.
    # This applies the *same* dense layer to *each* of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear') # 'linear' for regression
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse', # Mean Squared Error is standard for regression
        metrics=['mae'] # Mean Absolute Error is easier to interpret
    )
    return model

def create_v5_cnn_lstm(model_name="v5_cnn_lstm"):
    """
    Model v5: The SOTA-informed RCNN (CNN-LSTM) Hybrid.
    This is also a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # 1. CNN Feature Extraction Block
    # 1D Conv layers to find local patterns (motifs) in the 50-step sequence.
    # We use 'causal' padding to prevent the model from "cheating" by
    # looking at future time steps to predict the present.
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(x)
    
    # 2. LSTM Temporal Processing Block
    # The LSTM layer learns the long-term relationships between the
    # features extracted by the CNN.
    # We must use return_sequences=True.
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.4)(x)
    
    # 3. TimeDistributed Prediction Head
    # Applies a Dense layer to each of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear')
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
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
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1), # 25 patience
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
    print("🚀 STARTING SEQUENTIAL MODEL TRAINING (PHASE 2d)")
    print("=============================================================")
    
    # Find the data we just created
    train_files = glob.glob(Config.TRAIN_TFRECORD_PATH)
    val_files = glob.glob(Config.VAL_TFRECORD_PATH)
    
    if not train_files or not val_files:
        print(f"❌ FATAL: Could not find TFRecord files.")
        print(f"   Checked for train: {Config.TRAIN_TFRECORD_PATH}")
        print(f"   Checked for val: {Config.VAL_TFRECORD_PATH}")
        return

    # Get total number of samples from our log file
    # This is a bit of a hack, but it's the easiest way
    # In a real-world scenario, we'd read this from metadata.json
    total_train_seqs = 405
    total_val_seqs = 71
    
    # Calculate steps per epoch
    train_steps = total_train_seqs // Config.BATCH_SIZE
    val_steps = total_val_seqs // Config.BATCH_SIZE
    
    # Ensure steps are at least 1
    if train_steps == 0: train_steps = 1
    if val_steps == 0: val_steps = 1

    print(f"   Found {total_train_seqs} train sequences and {total_val_seqs} val sequences.")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Steps/Epoch: {train_steps}")
    print(f"   Validation Steps/Epoch: {val_steps}")

    # Create the datasets
    train_dataset = create_dataset(Config.TRAIN_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(Config.VAL_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=False)

    # Define models to train
    models_to_train = {
        'v4_lstm_baseline': create_v4_lstm_baseline,
        'v5_cnn_lstm': create_v5_cnn_lstm,
    }
    
    all_results = []
    
    for model_name, model_creator in models_to_train.items():
        results = train_model(
            model_creator, 
            model_name, 
            train_dataset, 
            val_dataset, 
            train_steps, 
            val_steps
        )
        all_results.append(results)

    print(f"\n{'='*70}\n📊 SEQUENTIAL EXPERIMENTS COMPLETE 📊\n{'='*70}")
    
    # Print summary table
    print(f"{'Model':<20} {'Best Val MAE':<15} {'Best Epoch':<12}")
    print('-'*70)
    for res in sorted(all_results, key=lambda x: x['best_validation_mae']):
        print(f"{res['model_version']:<20} {res['best_validation_mae']:.4f}{'':<11} {res['best_epoch']:<12}")
    
    best_model_results = min(all_results, key=lambda x: x['best_validation_mae'])
    print(f"\n🏆 BEST SEQUENTIAL MODEL: {best_model_results['model_version']} (Val MAE: {best_model_results['best_validation_mae']:.4f})")
    print(f"\n   Recall: Our 'Point-in-Time' (v2) model had a Test MAE of 0.1610")
    print(f"   ... We will need to run a final test, but this validation MAE is our new benchmark.")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()