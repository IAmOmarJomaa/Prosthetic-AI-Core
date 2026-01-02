# Save this as convert_to_tfrecord.py
# This is the FINAL data conversion script (Phase 2c).
# It converts the 10Hz CSV into a balanced, normalized, sequential TFRecord dataset.

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Input Files ---
    DATA_CSV_FILE = "10hz_feature_data.csv"
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    # --- Output ---
    OUTPUT_DIR = "tfrecord_dataset_v2"
    FEATURE_SCALER_FILE = "feature_scaler.pkl"
    TARGET_SCALER_FILE = "target_scaler.pkl" # We save the Min/Max as a scaler
    
    # --- Clustering & Balancing ---
    NUM_MOVEMENT_CLUSTERS = 6
    
    # --- Sequencing ---
    SEQUENCE_LENGTH = 50 # 50 samples * 100ms/sample = 5 seconds
    
    # --- TFRecord ---
    SEQUENCES_PER_SHARD = 500
    VALIDATION_SPLIT = 0.15 # 15% of our balanced sequences will be for validation

# --- 🛠️ HELPER FUNCTIONS ---

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the 10Hz CSV data."""
    print(f"Loading 10Hz data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        raise FileNotFoundError
    
    df = pd.read_csv(file_path)
    print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
    return df

def create_virtual_motors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Creates the 7 Virtual Motor columns based on the sdata201453.pdf paper."""
    print("--- 1. Creating 7 Virtual Motors ---")
    
    # Sensor map based on sdata201453.pdf
    VIRTUAL_MOTORS: Dict[str, List[str]] = {
        "motor_thumb_flex": [f"glove_{i}" for i in [3, 4]],
        "motor_index_flex": [f"glove_{i}" for i in [6, 7, 8]],
        "motor_middle_flex": [f"glove_{i}" for i in [10, 11, 12]],
        "motor_ring_flex": [f"glove_{i}" for i in [14, 15, 16]],
        "motor_pinky_flex": [f"glove_{i}" for i in [18, 19, 20]],
        "motor_thumb_abduct": ["glove_2"],
        "motor_wrist_flex": ["glove_21"]
    }
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
    
    # Get the input feature column names (feat_1 to feat_48)
    feature_cols = [f'feat_{i+1}' for i in range(48)]
    
    print(f"   ... Created {len(motor_cols)} motor columns.")
    return df, feature_cols, motor_cols

def normalize_data(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], norm_params_file: str, out_dir: str) -> pd.DataFrame:
    """
    Applies Min-Max normalization to motors (targets) and Z-Score to features (inputs).
    """
    print("--- 2. Normalizing Data ---")
    
    # --- A. Normalize Motors (Targets) ---
    print(f"   ... Applying Min-Max (0-1) normalization to {len(m_cols)} motors...")
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    for motor in m_cols:
        p = params[motor]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        
        # Ensure range is not zero to avoid division by zero
        if motor_range == 0: motor_range = 1.0 
        
        # Apply normalization and clip between 0 and 1
        df[motor] = (df[motor] - motor_min) / motor_range
        df[motor] = np.clip(df[motor], 0.0, 1.0)
    
    # Save the Min/Max params as our "target scaler" for the model
    # We will need to "un-normalize" the model's output later
    joblib.dump(params, os.path.join(out_dir, Config.TARGET_SCALER_FILE))

    # --- B. Normalize Features (Inputs) ---
    print(f"   ... Applying Z-Score normalization to {len(f_cols)} features...")
    feature_data = df[f_cols].values
    feature_scaler = StandardScaler()
    
    # Fit and transform the features
    features_scaled = feature_scaler.fit_transform(feature_data)
    
    # Save the scaler for the model to use on new data
    joblib.dump(feature_scaler, os.path.join(out_dir, Config.FEATURE_SCALER_FILE))
    
    # Put the normalized features back into the DataFrame
    df[f_cols] = features_scaled
    print("   ... Normalization complete. Scalers saved.")
    return df

def balance_data(df: pd.DataFrame, m_cols: List[str]) -> pd.DataFrame:
    """
    Performs stratified sampling using 'rest' + K-Means movement clusters.
    """
    print("--- 3. Balancing Data (Stratified Sampling) ---")
    
    # Separate Rest and Movement data
    rest_df = df[df['restimulus'] == 0].copy()
    move_df = df[df['restimulus'] != 0].copy()
    
    if rest_df.empty or move_df.empty:
        print("   ⚠️ Warning: Missing rest or movement data. Balancing may be skewed.")
        return df # Return original df if we can't balance

    print(f"   ... Found {len(rest_df)} 'rest' samples and {len(move_df)} 'movement' samples.")
    
    # --- A. Cluster Movements ---
    print(f"   ... Finding {Config.NUM_MOVEMENT_CLUSTERS} movement archetypes (K-Means)...")
    # We cluster on the *normalized* motor data
    kmeans = KMeans(n_clusters=Config.NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_df['cluster'] = kmeans.fit_predict(move_df[m_cols])
    
    # --- B. Stratified Sampling ---
    all_balanced_dfs = []
    
    # Find the count of each cluster
    cluster_counts = move_df['cluster'].value_counts()
    
    # Find the "minority class" (smallest movement cluster)
    min_move_count = cluster_counts.min()
    print(f"   ... Minority movement cluster has {min_move_count} samples.")
    
    # Sample from movement clusters
    for i in range(Config.NUM_MOVEMENT_CLUSTERS):
        cluster_df = move_df[move_df['cluster'] == i]
        all_balanced_dfs.append(cluster_df.sample(n=min_move_count, random_state=42, replace=False))
        
    # Sample from "Rest"
    # We will take an equal amount of "rest" samples
    rest_sample_size = min(len(rest_df), min_move_count)
    print(f"   ... Sampling {rest_sample_size} 'rest' samples.")
    all_balanced_dfs.append(rest_df.sample(n=rest_sample_size, random_state=42, replace=False))
    
    # Combine all balanced data
    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} samples.")
    return balanced_df

def write_sequences_to_tfrecord(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], output_dir: str):
    """
    Builds sequences from the balanced/normalized 10Hz data and saves to TFRecord.
    """
    print("--- 4. Creating Sequences & Saving to TFRecord ---")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Group by the 'group_id' to prevent sequences from crossing boundaries
    grouped = df.groupby('group_id')
    
    train_writer = None
    val_writer = None
    shard_counts = {'train': 0, 'val': 0}
    seq_counts = {'train': 0, 'val': 0}

    # Iterate over each "pure" chunk
    for group_id, group_df in tqdm(grouped, desc="   ... Processing groups into sequences", unit="group"):
        if len(group_df) < Config.SEQUENCE_LENGTH:
            continue # This chunk is too short to make a sequence
            
        features = group_df[f_cols].values
        targets = group_df[m_cols].values.astype(np.float32) # Ensure targets are float32
        
        # Slide our 50-sample window over this pure chunk
        for i in range(len(group_df) - Config.SEQUENCE_LENGTH + 1):
            
            # --- Assign to Train or Validation ---
            # We use the group_id to make the split. This ensures
            # ALL sequences from one chunk go to *either* train *or* val.
            # This is a robust way to prevent data leakage.
            if group_id % 100 < (Config.VALIDATION_SPLIT * 100):
                split = 'val'
            else:
                split = 'train'

            # --- Handle Shard Rollover ---
            if seq_counts[split] % Config.SEQUENCES_PER_SHARD == 0:
                if split == 'train' and train_writer: train_writer.close()
                if split == 'val' and val_writer: val_writer.close()
                
                path = os.path.join(output_dir, split, f'data_{shard_counts[split]:04d}.tfrecord')
                if split == 'train':
                    train_writer = tf.io.TFRecordWriter(path)
                else:
                    val_writer = tf.io.TFRecordWriter(path)
                shard_counts[split] += 1
            
            # Extract the 50-sample sequences
            seq_features = features[i : i + Config.SEQUENCE_LENGTH]
            seq_targets = targets[i : i + Config.SEQUENCE_LENGTH]
            
            # Serialize and create the TFExample
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(seq_features)),
                'targets': _bytes_feature(tf.io.serialize_tensor(seq_targets))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Write to the correct file
            if split == 'train':
                train_writer.write(example.SerializeToString())
            else:
                val_writer.write(example.SerializeToString())
            
            seq_counts[split] += 1

    if train_writer: train_writer.close()
    if val_writer: val_writer.close()
    
    print(f"\n   ... Sequencing complete.")
    print(f"   ... Wrote {seq_counts['train']:,} training sequences to {shard_counts['train']} shards.")
    print(f"   ... Wrote {seq_counts['val']:,} validation sequences to {shard_counts['val']} shards.")
    return seq_counts['train'] + seq_counts['val']

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING FINAL CONVERSION (PHASE 2c)")
    print("=============================================================")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load 10Hz CSV
    df = load_data(Config.DATA_CSV_FILE)
    if df is None: return

    # 2. Create 7 Virtual Motors
    df, feature_cols, motor_cols = create_virtual_motors(df)
    
    # 3. Normalize Inputs (Z-Score) and Outputs (Min-Max)
    df = normalize_data(df, feature_cols, motor_cols, Config.NORMALIZATION_PARAMS_FILE, Config.OUTPUT_DIR)

    # 4. Balance the data using Stratified Sampling
    balanced_df = balance_data(df, motor_cols)
    
    # 5. Create Sequences and Save to TFRecord
    total_seqs = write_sequences_to_tfrecord(balanced_df, feature_cols, motor_cols, Config.OUTPUT_DIR)

    # --- FINAL METADATA ---
    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_balanced_10hz_samples": len(balanced_df),
        "total_sequences_written": total_seqs,
        "feature_scaler": Config.FEATURE_SCALER_FILE,
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 FINAL CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {total_seqs:,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("   Scalers saved: feature_scaler.pkl, target_scaler.pkl")
    print("\n   ✅ THE DATASET IS NOW 100% READY FOR TRAINING.")
    print("=" * 60)

if __name__ == "__main__":
    main()