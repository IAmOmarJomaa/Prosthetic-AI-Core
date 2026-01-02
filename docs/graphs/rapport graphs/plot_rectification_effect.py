# Save this as plot_rectification_effect.py
# This script visualizes the effect of Rectification (absolute value)
# on a filtered and Z-scored signal.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

# --- ⚙️ TIME SLICE TO PLOT (ZOOMED IN) ---
# We'll find a "noisy" section, often during rest
START_SAMPLE = 30000 
# We'll plot a 0.5-second slice (1000 samples) to see the wave clearly
DURATION_SAMPLES = 1000 

# --- ⚙️ FILTER PARAMETERS ---
FS = 2000.0 # 2000 Hz
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
BAND_LOW = 20.0
BAND_HIGH = 450.0
BUTTER_ORDER = 4

# --- 🛠️ HELPER FUNCTION: Filter Design ---
def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = BAND_LOW / nyq
    high = BAND_HIGH / nyq
    b_band, a_band = signal.butter(BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

# --- Main Plotting Function ---
def plot_rectification_comparison(file_path):
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
    signal_raw = df[CHANNEL_TO_PLOT].values.reshape(-1, 1) # Reshape for scaler
    
    # 3. Create the filtered signal
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    
    signal_filtered = signal.filtfilt(b_notch, a_notch, signal_raw, axis=0)
    signal_filtered = signal.filtfilt(b_band, a_band, signal_filtered, axis=0)

    # 4. Create the Z-scored signal ("BEFORE" data)
    print("   ... Applying Z-Score (StandardScaler)...")
    scaler = StandardScaler()
    signal_zscored = scaler.fit_transform(signal_filtered)
    print("   ... Z-Score complete.")
    
    # 5. Create the Rectified signal ("AFTER" data)
    print("   ... Applying Rectification (Absolute Value)...")
    signal_rectified = np.abs(signal_zscored)

    # 6. Get the *slice* of both signals for plotting
    stop_sample = START_SAMPLE + DURATION_SAMPLES
    zscored_slice = signal_zscored[START_SAMPLE:stop_sample]
    rectified_slice = signal_rectified[START_SAMPLE:stop_sample]
    
    # Create a time axis in milliseconds
    time_axis_ms = (np.arange(len(zscored_slice)) / FS) * 1000.0 

    # 7. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.suptitle("Effect of Rectification (Absolute Value)", fontsize=16)
    
    # --- Plot 1: Z-Scored Signal (Before) ---
    ax1.plot(time_axis_ms, zscored_slice, label="Z-Scored Signal", color='b')
    ax1.axhline(0, color='k', linestyle='--', label="Zero line")
    ax1.set_title(f"BEFORE: Z-Scored {CHANNEL_TO_PLOT} (Positive and Negative Values)", fontsize=14)
    ax1.set_ylabel("Normalized Value (Std. Dev)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Rectified Signal (After) ---
    ax2.plot(time_axis_ms, rectified_slice, label="Rectified Signal", color='g')
    ax2.axhline(0, color='k', linestyle='--', label="Zero line")
    ax2.set_title(f"AFTER: Rectified {CHANNEL_TO_PLOT} (All Non-Negative)", fontsize=14)
    ax2.set_ylabel("Normalized Value (Absolute)")
    ax2.set_xlabel(f"Time (milliseconds) - A {DURATION_SAMPLES/FS:.2f} second slice", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "rectification_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... The top plot shows the wave centered at 0. The bottom plot shows all negative parts 'flipped' to positive.")

if __name__ == "__main__":
    plot_rectification_comparison(RAW_FILE_TO_INSPECT)