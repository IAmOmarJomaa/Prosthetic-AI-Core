# Save this as plot_zscore_effect.py
# This script visualizes the statistical distribution of the 12 EMG channels
# *before* and *after* Z-Score normalization.

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

# The 12 channels we will analyze
EMG_COLS = [f"emg_{i}" for i in range(1, 13)]

# We'll use a large chunk of data for a good statistical sample
# Let's take 50,000 samples (25 seconds) from a movement section
START_SAMPLE = 30000 
SAMPLE_COUNT = 50000

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
def plot_zscore_comparison(file_path):
    print(f"Loading raw data from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load *only* the EMG data
        df = pd.read_csv(file_path, usecols=EMG_COLS)
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal
    signal_raw = df.values
    
    # 3. Create the filtered signal ("BEFORE" data)
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    
    signal_filtered = np.zeros_like(signal_raw)
    for i in range(signal_raw.shape[1]): # Filter each channel
        signal_filtered[:, i] = signal.filtfilt(b_notch, a_notch, signal_raw[:, i])
        signal_filtered[:, i] = signal.filtfilt(b_band, a_band, signal_filtered[:, i])
    print("   ... Filtering complete.")

    # 4. Create the Z-scored signal ("AFTER" data)
    print("   ... Applying Z-Score (StandardScaler)...")
    scaler = StandardScaler()
    signal_zscored = scaler.fit_transform(signal_filtered)
    print("   ... Z-Score complete.")

    # 5. Get the *slice* of both signals for plotting
    stop_sample = START_SAMPLE + SAMPLE_COUNT
    filtered_slice = signal_filtered[START_SAMPLE:stop_sample]
    zscored_slice = signal_zscored[START_SAMPLE:stop_sample]
    
    # 6. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.suptitle("Effect of Z-Score Normalization on EMG Channel Distributions", fontsize=16)
    
    # --- Plot 1: BEFORE Z-Score ---
    ax1.set_title("BEFORE: Filtered Data (12 Channels)")
    ax1.set_xlabel("Signal Value (Amplitude)")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)
    # This will plot 12 *different* density curves
    for i in range(filtered_slice.shape[1]):
        sns.kdeplot(filtered_slice[:, i], ax=ax1, label=f'emg_{i+1}', warn_singular=False)
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    # This plot should look "messy" with 12 different peaks and widths

    # --- Plot 2: AFTER Z-Score ---
    ax2.set_title("AFTER: Z-Scored Data (12 Channels)")
    ax2.set_xlabel("Normalized Value (Standard Deviations)")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)
    # This will plot 12 *identical* density curves
    for i in range(zscored_slice.shape[1]):
        sns.kdeplot(zscored_slice[:, i], ax=ax2, warn_singular=False)
    # This plot should show one single, clean bell curve
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "zscore_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... The 'BEFORE' plot shows 12 different distributions.")
    print("   ... The 'AFTER' plot should show one single, centered distribution (a perfect bell curve).")

if __name__ == "__main__":
    sns.set_theme() # Use Seaborn's nice default styling
    plot_zscore_comparison(RAW_FILE_TO_INSPECT)