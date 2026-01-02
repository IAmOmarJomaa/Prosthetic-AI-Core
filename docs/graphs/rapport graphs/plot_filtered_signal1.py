# Save this as plot_notch_filter_effect.py
# This script specifically isolates and visualizes the effect of *only*
# the 50Hz notch filter on a small slice of the raw signal.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv" # Use the same file path as before

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

# --- ⚙️ TIME SLICE TO PLOT (ZOOMED IN) ---
# We'll find a "noisy" section, often during rest
START_SAMPLE = 30000 
# We'll plot a tiny 0.2-second slice (400 samples)
DURATION_SAMPLES = 400 

# --- ⚙️ FILTER PARAMETERS ---
FS = 2000.0 # 2000 Hz
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
# We are *NOT* using the bandpass filter in this script

# --- 🛠️ HELPER FUNCTION: Filter Design ---
def design_notch_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Creates *only* the 50Hz notch filter."""
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs)
    return b_notch, a_notch

# --- Main Plotting Function ---
def plot_filter_comparison(file_path):
    print(f"Loading raw data for {CHANNEL_TO_PLOT} from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the single channel we want to inspect
        df = pd.read_csv(file_path, usecols=[CHANNEL_TO_PLOT])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal
    signal_raw = df[CHANNEL_TO_PLOT].values
    
    # 3. Create the filtered signal (NOTCH ONLY)
    print(f"   ... Applying ONLY the 50Hz notch filter...")
    b_notch, a_notch = design_notch_filter(FS)
    signal_notched = signal.filtfilt(b_notch, a_notch, signal_raw)
    print("   ... Filtering complete.")

    # 4. Get the *slice* of both signals
    stop_sample = START_SAMPLE + DURATION_SAMPLES
    raw_slice = signal_raw[START_SAMPLE:stop_sample]
    notched_slice = signal_notched[START_SAMPLE:stop_sample]
    
    # Create a time axis in milliseconds
    time_axis_ms = (np.arange(len(raw_slice)) / FS) * 1000.0 

    # 5. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # --- Plot 1: Raw Signal (should show 50Hz "fuzz") ---
    ax1.plot(time_axis_ms, raw_slice, label="Raw Signal", color='r', alpha=0.9)
    ax1.set_title(f"BEFORE: Raw {CHANNEL_TO_PLOT} (Note the 50Hz 'fuzz')", fontsize=14)
    ax1.set_ylabel("Raw EMG Value")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Notched Signal (should be smooth) ---
    ax2.plot(time_axis_ms, notched_slice, label="Notch Filtered Signal", color='b')
    ax2.set_title(f"AFTER: {CHANNEL_TO_PLOT} with 50Hz Hum Removed", fontsize=14)
    ax2.set_ylabel("Filtered EMG Value")
    ax2.set_xlabel(f"Time (milliseconds) - A {DURATION_SAMPLES/FS:.2f} second slice", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = "notch_filter_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... This plot *only* shows the removal of the 50Hz hum.")

if __name__ == "__main__":
    plot_filter_comparison(RAW_FILE_TO_INSPECT)