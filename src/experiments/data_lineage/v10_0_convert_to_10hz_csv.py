# Save this as convert_to_10hz_csv.py
# This script converts raw 2000Hz CSVs into a single 10Hz feature-engineered CSV
# for Exploratory Data Analysis (EDA), with verbose logging.

import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # We will define the file list in the main() function.
    OUTPUT_CSV_PATH = "10hz_feature_data.csv" # The SINGLE output CSV
    
    # Signal Processing
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # Feature Engineering (Downsampling)
    WINDOW_MS = 100 # 100ms window
    WINDOW_SAMPLES = int(FS * (WINDOW_MS / 1000.0)) # = 200 samples

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def calculate_hudgins_features(window: np.ndarray) -> np.ndarray:
    """
    Calculates the 4 Hudgins' features (RMS, WL, ZC, SSC) for a 12-channel window.
    Input window shape: (200, 12)
    Output feature vector shape: (48,)
    """
    n_samples, n_channels = window.shape
    features = []
    
    # 1. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2, axis=0))
    features.append(rms)
    
    # 2. Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    features.append(wl)
    
    # 3. Zero Crossings (ZC) (with a small threshold for noise)
    zc = np.sum((window[:-1] * window[1:] < 0) & (np.abs(window[:-1] - window[1:]) > 1e-6), axis=0)
    features.append(zc)
    
    # 4. Slope Sign Changes (SSC) (with a small threshold for noise)
    ssc = np.sum((window[1:-1] - window[:-2]) * (window[1:-1] - window[2:]) > 1e-6, axis=0)
    features.append(ssc)
    
    return np.concatenate(features, axis=0).astype(np.float32)

# --- 1. DATA GATHERING (2000Hz -> 10Hz) ---
def build_10hz_list_of_rows(file_list: List[str]) -> List[Dict]:
    """
    Processes all raw CSVs, extracts features, and returns a list of rows
    (as dictionaries) for the 10Hz data.
    """
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_rows = [] # List to hold all our new 10Hz rows (as dicts)
    group_id_counter = 0 # This is the critical ID for preventing sequence breaks
    
    print("Processing Raw Files (2000Hz -> 10Hz)...")
    
    for file_index, fp in enumerate(tqdm(file_list, desc="Overall Progress", unit="file", total=len(file_list))):
        print(f"\nProcessing File {file_index+1}/{len(file_list)}: {os.path.basename(fp)}")
        try:
            # Load all necessary columns
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
            
            print(f"   ... Found {len(chunk_starts)} 'pure' movement/rest chunks.")
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                if start >= stop: continue
                
                chunk_emg = emg_data[start:stop]
                chunk_glove = glove_data[start:stop]
                chunk_label = labels[start] # All labels in this chunk are the same
                
                n_samples = len(chunk_emg)
                # Use non-overlapping windows, discard partial remainder
                n_windows = n_samples // Config.WINDOW_SAMPLES 
                remainder = n_samples % Config.WINDOW_SAMPLES
                
                if n_windows == 0:
                    print(f"   LOG: Discarding chunk (rows {start}-{stop}, label {chunk_label}) - too short ({n_samples} samples) for one window.")
                    continue
                
                # *** YOUR REQUESTED LOGGING ***
                if remainder > 0:
                    print(f"   LOG: Discarding last {remainder} samples from chunk (rows {start}-{stop}, label {chunk_label}) to ensure pure windows.")
                
                # Process all *full* non-overlapping windows in this chunk
                for i in range(n_windows):
                    win_start = start + (i * Config.WINDOW_SAMPLES)
                    win_stop = win_start + Config.WINDOW_SAMPLES
                    
                    # We reference the *original* dataframe slices
                    emg_window = emg_data[win_start:win_stop]
                    glove_window = glove_data[win_start:win_stop]
                    
                    features_48 = calculate_hudgins_features(emg_window)
                    target_22 = glove_window[-1, :].astype(np.float32)
                    
                    # Create a dictionary for this 10Hz row
                    row_data = {
                        'group_id': group_id_counter, # This ID marks the 'pure' chunk
                        'restimulus': chunk_label
                    }
                    # Add features
                    for f_idx in range(48):
                        row_data[f'feat_{f_idx+1}'] = features_48[f_idx] # Use 1-indexing
                    # Add targets
                    for t_idx in range(22):
                        row_data[f'glove_{t_idx+1}'] = target_22[t_idx] # Use 1-indexing
                        
                    all_rows.append(row_data)
                
                group_id_counter += 1 # Increment for each new, pure chunk
                
        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process file {fp}. Error: {e}")
            continue
            
    print(f"\n... 10Hz data extraction complete. Found {len(all_rows)} total 10Hz samples.")
    return all_rows

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING CONVERSION TO 10Hz CSV (PHASE 2a)")
    print("=============================================================")
    
    # ==================== EDIT THIS LIST ====================
    # Paste your file paths here for the test run.
    # Use 'r' before the string to handle Windows backslashes '\'.
    FILES_TO_PROCESS = [
        r"D:\stage\v5.0\data\DB2\s1\E1_A1.csv",
        r"DS:\stage\v5.0\data\DB2\s1\E2_A1.csv",
        r"D:\stage\v5.0\data\DB2\s2\E1_A1.csv"
        # Add more files here when you are ready for the full run
    ]
    # ========================================================
    
    print(f"Starting processing for {len(FILES_TO_PROCESS)} manually specified files...")

    # --- STEP 1: Process 2000Hz -> 10Hz ---
    list_of_rows = build_10hz_list_of_rows(FILES_TO_PROCESS)
    
    if not list_of_rows:
        print("❌ FATAL: No 10Hz data could be extracted. Check file paths and data.")
        return

    # --- STEP 2: Convert list of dicts to DataFrame and save ---
    print(f"\n... Converting to DataFrame and saving to {Config.OUTPUT_CSV_PATH}...")
    try:
        final_df = pd.DataFrame(list_of_rows)
        # We will use CSV as you requested, not Parquet.
        final_df.to_csv(Config.OUTPUT_CSV_PATH, index=False)
        print(f"   ... Saved as {Config.OUTPUT_CSV_PATH}.")

    except Exception as e:
        print(f"❌ FATAL: Failed to save DataFrame to CSV. Error: {e}")
        return

    print("\n" + "=" * 60)
    print(f"🎉 CONVERSION TO 10Hz CSV COMPLETE!")
    print(f"   Total 10Hz samples created: {len(list_of_rows):,}")
    print(f"   Output file: {Config.OUTPUT_CSV_PATH}")
    print("\n   ✅ YOU CAN NOW OPEN THIS CSV FILE TO SEE THE DATA.")
    print("\nNext step: Run the 'Phase 2b' analysis script on this new file.")
    print("=" * 60)

if __name__ == "__main__":
    main()