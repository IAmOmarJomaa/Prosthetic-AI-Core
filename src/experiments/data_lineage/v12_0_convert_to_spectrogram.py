# Save this as convert_to_spectrogram.py
# This script converts raw 2000Hz CSVs into a sequential
# dataset of 2D Spectrograms (images) and normalized 7-motor targets.

import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Input Files ---
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    # --- Output ---
    OUTPUT_DIR = "tfrecord_dataset_v3_spectrogram"
    TARGET_SCALER_FILE = "target_scaler.pkl" # We'll re-save our Min/Max scaler
    
    # --- Signal Processing ---
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # --- Sequencing ---
    # We want to match the 5-second movement, as per the dataset guide
    SEQUENCE_S = 5.0
    SEQUENCE_SAMPLES = int(FS * SEQUENCE_S) # 5.0s * 2000 Hz = 10,000 samples
    
    # --- Spectrogram (STFT) Parameters ---
    # We'll use the same 100ms (200-sample) window as our 10Hz features
    STFT_WINDOW_SAMPLES = 200
    # We'll overlap windows by 50% for a smoother image
    STFT_OVERLAP_SAMPLES = 100 
    
    # --- Balancing & TFRecord ---
    SEQUENCES_PER_SHARD = 200 # Spectrograms are large
    VALIDATION_SPLIT = 0.20 # 20% for validation

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def create_virtual_motors(glove_data: np.ndarray) -> np.ndarray:
    """
    Converts a (N, 22) glove array into a (N, 7) motor array.
    """
    # This is our known, fact-based mapping
    VIRTUAL_MOTORS: Dict[str, List[int]] = {
        "motor_thumb_flex": [2, 3], # glove_3, glove_4 (0-indexed)
        "motor_index_flex": [5, 6, 7],
        "motor_middle_flex": [9, 10, 11],
        "motor_ring_flex": [13, 14, 15],
        "motor_pinky_flex": [17, 18, 19],
        "motor_thumb_abduct": [1], # glove_2
        "motor_wrist_flex": [20]  # glove_21
    }
    
    # Create the new 7-motor array
    motor_data = np.zeros((glove_data.shape[0], 7), dtype=np.float32)
    for i, (motor_name, sensor_indices) in enumerate(VIRTUAL_MOTORS.items()):
        motor_data[:, i] = glove_data[:, sensor_indices].mean(axis=1)
        
    return motor_data

def normalize_motors(motor_data: np.ndarray, norm_params_file: str) -> np.ndarray:
    """
    Applies Min-Max (0-1) normalization to the 7 motor columns.
    """
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    normalized_data = np.zeros_like(motor_data)
    
    # This list *must* match the order in VIRTUAL_MOTORS
    motor_names = [
        "motor_thumb_flex", "motor_index_flex", "motor_middle_flex",
        "motor_ring_flex", "motor_pinky_flex", "motor_thumb_abduct", "motor_wrist_flex"
    ]
    
    for i, motor_name in enumerate(motor_names):
        p = params[motor_name]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        if motor_range == 0: motor_range = 1.0 
        
        normalized_data[:, i] = (motor_data[:, i] - motor_min) / motor_range
    
    # Clip to our target 0-1 range
    return np.clip(normalized_data, 0.0, 1.0).astype(np.float32)

def compute_spectrograms(emg_chunk: np.ndarray) -> np.ndarray:
    """
    Converts a (10000, 12) EMG chunk into a (12, 101, 99) spectrogram tensor.
    """
    n_channels = emg_chunk.shape[1]
    all_spectrograms = []
    
    for i in range(n_channels):
        f, t, Sxx = signal.stft(
            emg_chunk[:, i],
            fs=Config.FS,
            nperseg=Config.STFT_WINDOW_SAMPLES,
            noverlap=Config.STFT_OVERLAP_SAMPLES
        )
        # Sxx shape is (n_freqs, n_times), e.g., (101, 99)
        # We use np.abs() to get magnitude and add a small epsilon for log stability
        Sxx_mag = np.abs(Sxx)
        all_spectrograms.append(Sxx_mag)

    # Stack into a (12, 101, 99) tensor
    # We use log-magnitude, which is standard for audio/EMG "images"
    # This compresses the dynamic range
    spectrogram_stack = np.stack(all_spectrograms, axis=0)
    return np.log(spectrogram_stack + 1e-6).astype(np.float32)

def make_tfexample(spectrogram: np.ndarray, target_vector: np.ndarray, label: int) -> tf.train.Example:
    """Serializes a (12, 101, 99) spectrogram and (7,) target vector."""
    
    spectrogram_bytes = tf.io.serialize_tensor(spectrogram).numpy()
    target_bytes = tf.io.serialize_tensor(target_vector).numpy()
    
    feature_dict = {
        'spectrogram': tf.train.Feature(bytes_list=tf.train.BytesList(value=[spectrogram_bytes])),
        'target_vector': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING SPECTROGRAM CONVERSION (PHASE 3a)")
    print("=============================================================")
    
    # ==================== EDIT THIS LIST ====================
    # This should be the FULL list of your raw data files
    FILES_TO_PROCESS = [
        r"D:/DB\\DB2\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S12_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S13_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S14_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S15_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S16_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S17_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S18_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S19_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S20_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S21_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S22_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S23_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S24_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S25_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S26_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S27_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S28_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S29_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S30_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S31_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S32_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S33_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S34_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S35_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S36_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S37_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S38_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S39_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S40_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB2\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S12_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S13_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S14_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S15_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S16_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S17_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S18_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S19_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S20_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S21_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S22_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S23_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S24_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S25_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S26_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S27_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S28_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S29_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S30_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S31_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S32_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S33_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S34_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S35_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S36_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S37_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S38_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S39_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S40_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S9_E2_A1.csv",
        r"D:/DB\\DB3\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB3\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S9_E2_A1.csv"
    ]
    # ========================================================
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_sequences = [] # This will be a list of (spectrogram, target_vector, label)
    
    print(f"Starting processing for {len(FILES_TO_PROCESS)} files...")
    
    for fp in tqdm(FILES_TO_PROCESS, desc="Processing Files", unit="file"):
        try:
            use_cols = [f"emg_{i}" for i in range(1, 13)] + [f"glove_{i}" for i in range(1, 23)] + ["restimulus"]
            df = pd.read_csv(fp, usecols=lambda c: c in use_cols)
            
            emg_cols = [f"emg_{i}" for i in range(1, 13)]
            glove_cols = [f"glove_{i}" for i in range(1, 23)]
            
            emg_data = df[emg_cols].values
            glove_data = df[glove_cols].values
            labels = df['restimulus'].values.astype(int)
            
            # Filter the raw EMG data
            for i in range(emg_data.shape[1]):
                emg_data[:, i] = signal.filtfilt(b_notch, a_notch, emg_data[:, i])
                emg_data[:, i] = signal.filtfilt(b_band, a_band, emg_data[:, i])
            
            # Find boundaries of "pure" chunks
            boundaries = np.where(np.diff(labels) != 0)[0] + 1
            chunk_starts = np.concatenate(([0], boundaries))
            chunk_stops = np.concatenate((boundaries, [len(labels)]))
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                n_samples = stop - start
                chunk_label = labels[start]
                
                # How many 5-second (10,000-sample) sequences can we make from this chunk?
                n_sequences = n_samples // Config.SEQUENCE_SAMPLES
                
                if n_sequences == 0: continue
                    
                # Process all *full* non-overlapping 5-second sequences
                for i in range(n_sequences):
                    seq_start = start + (i * Config.SEQUENCE_SAMPLES)
                    seq_stop = seq_start + Config.SEQUENCE_SAMPLES
                    
                    emg_chunk = emg_data[seq_start:seq_stop]
                    glove_chunk = glove_data[seq_start:seq_stop]
                    
                    # 1. Create the Spectrogram "Image"
                    spectrogram = compute_spectrograms(emg_chunk)
                    
                    # 2. Create the Target Vector
                    # We only care about the hand pose at the *end* of the 5s movement
                    # We'll average the *last 100ms* of glove data for a stable target
                    target_glove_window = glove_chunk[-Config.STFT_WINDOW_SAMPLES:]
                    # Convert (200, 22) -> (22,) by averaging
                    target_glove_vector = target_glove_window.mean(axis=0)
                    # Convert (22,) -> (7,) by grouping
                    target_motor_vector = create_virtual_motors(target_glove_vector.reshape(1, -1))[0]
                    
                    # Add this (input, output, label) pair to our big list
                    all_sequences.append((spectrogram, target_motor_vector, chunk_label))

        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process file {fp}. Error: {e}")
            continue

    print(f"\n... Found {len(all_sequences)} total 5-second sequences.")
    
    # --- 4. Balance the Sequences ---
    print("--- 2. Balancing Sequences (Stratified Sampling) ---")
    
    # Put our sequences into a DataFrame to balance
    seq_df = pd.DataFrame(all_sequences, columns=['spectrogram', 'target_vector', 'label'])
    
    rest_df = seq_df[seq_df['label'] == 0]
    move_df = seq_df[seq_df['label'] != 0]
    
    if rest_df.empty or move_df.empty:
        print("❌ FATAL: Dataset is missing 'rest' or 'movement' sequences.")
        return

    print(f"   ... Found {len(rest_df)} 'rest' sequences and {len(move_df)} 'movement' sequences.")
    
    # Find minority class
    min_move_count = move_df.groupby('label').size().min()
    min_class_count = min(len(rest_df), min_move_count)
    
    print(f"   ... Minority class has {min_class_count} sequences. Balancing to this number.")
    
    all_balanced_dfs = []
    
    # Sample from "Rest"
    all_balanced_dfs.append(rest_df.sample(n=min_class_count, random_state=42))
    
    # Sample from each movement
    movement_labels = move_df['label'].unique()
    for label in movement_labels:
        move_label_df = move_df[move_df['label'] == label]
        all_balanced_dfs.append(move_label_df.sample(n=min_class_count, random_state=42, replace=True)) # Oversample if needed

    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} sequences.")

    # --- 5. Normalize Targets and Save TFRecords ---
    print("--- 3. Normalizing Targets and Saving to TFRecord ---")
    
    # Stack all target vectors for normalization
    target_vectors_stacked = np.stack(balanced_df['target_vector'].values)
    
    # Normalize the 7-motor targets using our JSON file
    normalized_targets = normalize_motors(target_vectors_stacked, Config.NORMALIZATION_PARAMS_FILE)
    
    # Put them back in the DataFrame
    balanced_df['target_vector'] = list(normalized_targets)
    
    # Save the normalization params file we used
    joblib.dump(json.load(open(Config.NORMALIZATION_PARAMS_FILE)), os.path.join(Config.OUTPUT_DIR, Config.TARGET_SCALER_FILE))

    # --- 6. Write TFRecords ---
    train_dir = os.path.join(Config.OUTPUT_DIR, "train")
    val_dir = os.path.join(Config.OUTPUT_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    val_size = int(len(balanced_df) * Config.VALIDATION_SPLIT)
    val_df = balanced_df.iloc[:val_size]
    train_df = balanced_df.iloc[val_size:]
    
    print(f"   ... Splitting into {len(train_df)} train and {len(val_df)} val sequences.")
    
    # --- Write Shards ---
    for split_name, df_split in [('train', train_df), ('val', val_df)]:
        shard_count = 0
        writer = None
        for i, (idx, row) in enumerate(tqdm(df_split.iterrows(), total=len(df_split), desc=f"   ... Writing {split_name} shards")):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(Config.OUTPUT_DIR, split_name, f'data_{shard_count:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_count += 1
            
            example = make_tfexample(row['spectrogram'], row['target_vector'], row['label'])
            writer.write(example.SerializeToString())
        if writer: writer.close()
        print(f"   ... Wrote {len(df_split)} {split_name} sequences to {shard_count} shards.")

    # --- FINAL METADATA ---
    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_balanced_sequences": len(balanced_df),
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 SPECTROGRAM CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {len(balanced_df):,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("\n   ✅ THE DATASET IS NOW READY FOR TRAINING (PHASE 3b).")
    print("=" * 60)

if __name__ == "__main__":
    main()