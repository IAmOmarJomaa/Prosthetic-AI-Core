# Save this as convert_to_tfrecord_v2.py
# This is the FINAL data conversion script (Phase 2c).
# It converts the 10Hz CSV into a balanced, normalized, sequential TFRecord dataset.
# *** VERSION 2.1: Includes the float32 cast fix ***

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
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    DATA_CSV_FILE = "10hz_feature_data.csv"
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    OUTPUT_DIR = "tfrecord_dataset_v2"
    FEATURE_SCALER_FILE = "feature_scaler.pkl"
    TARGET_SCALER_FILE = "target_scaler.pkl" 
    
    # MANDATORY FIX: Explicitly define the correct dimensions
    NUM_TARGETS = 12 # 12 Virtual Motors
    
    NUM_MOVEMENT_CLUSTERS = 12 # 12 movement archetypes
    SEQUENCE_LENGTH = 50 # 5 seconds
    
    SEQUENCES_PER_SHARD = 500
    VALIDATION_SPLIT = 0.15 

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
    """Creates the 12 Virtual Motor columns based on the report's final grouping."""
    print("--- 1. Creating 12 Virtual Motors (Targets) ---")
    
    # MANDATORY FIX: Updated to 12-motor grouping (must match analyze_10hz_data.py)
    VIRTUAL_MOTORS: Dict[str, List[str]] = {
        # A. Primary Flexion/Abduction (7 Motors)
        "motor_thumb_flex": ["glove_3", "glove_4"],
        "motor_index_flex": ["glove_5", "glove_6", "glove_7"],
        "motor_middle_flex": ["glove_8", "glove_9", "glove_10"],
        "motor_ring_flex": ["glove_12", "glove_13", "glove_14"],
        "motor_pinky_flex": ["glove_16", "glove_17", "glove_18"],
        "motor_thumb_abduct": ["glove_2"],
        "motor_wrist_flex_ext": ["glove_21"], 
        
        # B. Secondary Adduction/Waving (5 Motors - Enhanced)
        "motor_index_adduct": ["glove_11"],
        "motor_middle_adduct": ["glove_15"],
        "motor_ring_adduct": ["glove_19"],
        "motor_pinky_adduct": ["glove_20"],
        "motor_hand_waving": ["glove_1", "glove_22"]
    }
    
    # Check for dimensional mismatch
    if len(VIRTUAL_MOTORS) != Config.NUM_TARGETS:
         print(f"❌ ERROR: Motor count ({len(VIRTUAL_MOTORS)}) does not match Config.NUM_TARGETS ({Config.NUM_TARGETS}).")
         raise ValueError("Dimensional Mismatch in VIRTUAL_MOTORS setup.")
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
    
    feature_cols = [f'feat_{i+1}' for i in range(48)]
    return df, feature_cols, motor_cols

def normalize_data(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], norm_params_file: str, out_dir: str) -> pd.DataFrame:
    """
    Applies Min-Max to motors (targets) and Z-Score to features (inputs).
    This is applied to the *entire* 10Hz dataframe.
    """
    print("--- 2. Normalizing All 10Hz Data ---")
    
    # --- A. Normalize Motors (Targets) ---
    # The print statement is now correct: it prints the actual number of motors being processed
    print(f"   ... Applying Min-Max (0-1) to {len(m_cols)} motors...") 
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    for motor in m_cols:
        p = params[motor]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        if motor_range == 0: motor_range = 1.0 
        
        df[motor] = (df[motor] - motor_min) / motor_range
        df[motor] = np.clip(df[motor], 0.0, 1.0)
    
    joblib.dump(params, os.path.join(out_dir, Config.TARGET_SCALER_FILE))

    # --- B. Normalize Features (Inputs) ---
    print(f"   ... Applying Z-Score to {len(f_cols)} features...")
    feature_data = df[f_cols].values
    feature_scaler = StandardScaler()
    
    # Fit and transform the features
    # Save as float32 to prevent type mismatch error
    features_scaled = feature_scaler.fit_transform(feature_data).astype(np.float32)
    
    joblib.dump(feature_scaler, os.path.join(out_dir, Config.FEATURE_SCALER_FILE))
    
    # Put the normalized features back into the DataFrame
    df[f_cols] = features_scaled
    print("   ... Normalization complete. Scalers saved.")
    return df

def create_all_sequences(df: pd.DataFrame, f_cols: List[str], m_cols: List[str]) -> pd.DataFrame:
    """
    Builds all possible sequences from the full, normalized 10Hz dataframe.
    """
    print("--- 3. Creating All Possible Sequences ---")
    
    all_sequences = [] # This will be a list of dicts
    
    # Group by the 'group_id' to prevent sequences from crossing boundaries
    grouped = df.groupby('group_id')
    
    for group_id, group_df in tqdm(grouped, desc="   ... Processing groups", unit="group"):
        if len(group_df) < Config.SEQUENCE_LENGTH:
            continue # This chunk is too short
            
        # ==================== THE FIX (Alternative Location) ====================
        # We ensure features are float32 before sequencing.
        # This is the one-line fix.
        features = group_df[f_cols].values.astype(np.float32) 
        # ======================================================================
        
        targets = group_df[m_cols].values.astype(np.float32)
        labels = group_df['restimulus'].values
        
        # Slide our 50-sample window over this pure chunk
        for i in range(len(group_df) - Config.SEQUENCE_LENGTH + 1):
            
            seq_label = labels[i + Config.SEQUENCE_LENGTH - 1]
            seq_features = features[i : i + Config.SEQUENCE_LENGTH]
            seq_targets = targets[i : i + Config.SEQUENCE_LENGTH]
            
            all_sequences.append({
                "features": seq_features,
                "targets": seq_targets,
                "label": seq_label
            })

    print(f"   ... Found {len(all_sequences):,} total valid sequences.")
    return pd.DataFrame(all_sequences)

def balance_sequences(seq_df: pd.DataFrame, m_cols: List[str]) -> pd.DataFrame:
    """
    Performs stratified sampling on the SEQUENCES, not the 10Hz data.
    """
    print("--- 4. Balancing Sequences (Stratified Sampling) ---")
    
    rest_seq_df = seq_df[seq_df['label'] == 0].copy()
    move_seq_df = seq_df[seq_df['label'] != 0].copy()
    
    if rest_seq_df.empty or move_seq_df.empty:
        print("   ⚠️ Warning: Missing rest or movement sequences. Balancing may be skewed.")
        return seq_df

    print(f"   ... Found {len(rest_seq_df)} 'rest' sequences and {len(move_seq_df)} 'movement' sequences.")
    
    # The number of movement clusters must match the number of movement motors (12)
    # The stratified sampling will now reflect the 13 classes (12 movements + 1 rest)
    print(f"   ... Finding {Config.NUM_MOVEMENT_CLUSTERS} movement archetypes (K-Means)...") 
    
    avg_poses = np.array([seq.mean(axis=0) for seq in move_seq_df['targets']])
    
    kmeans = KMeans(n_clusters=Config.NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_seq_df['cluster'] = kmeans.fit_predict(avg_poses)
    
    all_balanced_dfs = []
    
    cluster_counts = move_seq_df['cluster'].value_counts()
    
    min_move_count = cluster_counts.min()
    print(f"   ... Minority movement cluster has {min_move_count} sequences.")
    
    for i in range(Config.NUM_MOVEMENT_CLUSTERS):
        cluster_df = move_seq_df[move_seq_df['cluster'] == i]
        all_balanced_dfs.append(cluster_df.sample(n=min_move_count, random_state=42, replace=False))
        
    # The rest class is balanced against the size of the smallest movement cluster
    rest_sample_size = min(len(rest_seq_df), min_move_count) 
    print(f"   ... Sampling {rest_sample_size} 'rest' sequences.")
    all_balanced_dfs.append(rest_seq_df.sample(n=rest_sample_size, random_state=42, replace=False))
    
    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} sequences.")
    return balanced_df

def write_sequences_to_tfrecord(seq_df: pd.DataFrame, output_dir: str):
    """
    Saves the final, balanced DataFrame of sequences to TFRecord files.
    """
    print("--- 5. Saving Sequences to TFRecord ---")
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    val_size = int(len(seq_df) * Config.VALIDATION_SPLIT)
    val_df = seq_df.iloc[:val_size]
    train_df = seq_df.iloc[val_size:]
    
    print(f"   ... Splitting into {len(train_df)} train and {len(val_df)} val sequences.")
    
    shard_counts = {'train': 0, 'val': 0}

    # --- Write Training Data ---
    with tqdm(total=len(train_df), desc="   ... Writing train shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(train_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(train_dir, f'data_{shard_counts["train"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['train'] += 1
            
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    # --- Write Validation Data ---
    with tqdm(total=len(val_df), desc="   ... Writing val shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(val_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(val_dir, f'data_{shard_counts["val"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['val'] += 1
                
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    print(f"\n   ... Wrote {len(train_df):,} training sequences to {shard_counts['train']} shards.")
    print(f"   ... Wrote {len(val_df):,} validation sequences to {shard_counts['val']} shards.")
    return len(seq_df)

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING FINAL CONVERSION (PHASE 2c) - V2 (Corrected Order)")
    print("=============================================================")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = load_data(Config.DATA_CSV_FILE)
    if df is None: return

    df, feature_cols, motor_cols = create_virtual_motors(df)
    
    df = normalize_data(df, feature_cols, motor_cols, Config.NORMALIZATION_PARAMS_FILE, Config.OUTPUT_DIR)

    all_sequences_df = create_all_sequences(df, feature_cols, motor_cols)
    
    balanced_seq_df = balance_sequences(all_sequences_df, motor_cols)
    
    total_seqs = write_sequences_to_tfrecord(balanced_seq_df, Config.OUTPUT_DIR)

    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_unbalanced_sequences": len(all_sequences_df),
        "total_balanced_sequences": total_seqs,
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