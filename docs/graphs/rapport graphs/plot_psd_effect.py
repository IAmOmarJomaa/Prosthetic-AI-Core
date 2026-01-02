# Save this as plot_psd_effect.py
# This script visualizes the *frequency domain* (PSD)
# to show the effect of the filters.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

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
def plot_psd_comparison(file_path):
    print(f"Loading raw data for {CHANNEL_TO_PLOT} from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the single channel we want to inspect
        df = pd.read_csv(file_path, usecols=[CHANNEL_TO_PLOT, 'restimulus'])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal *during movement* (more signal to see)
    signal_raw = df[df['restimulus'] != 0][CHANNEL_TO_PLOT].values
    if len(signal_raw) == 0:
        print("   ⚠️ Warning: No movement data found, using 'rest' data.")
        signal_raw = df[CHANNEL_TO_PLOT].values

    # 3. Create the filtered signal
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    signal_filtered = signal.filtfilt(b_notch, a_notch, signal_raw)
    signal_filtered = signal.filtfilt(b_band, a_band, signal_filtered)
    print("   ... Filtering complete.")

    # 4. Calculate the Power Spectral Density (PSD) for both signals
    # This is the "frequency graph" you were asking for.
    print("   ... Calculating Power Spectral Density (PSD)...")
    # Using signal.welch() is a standard method to get a clean PSD
    f_raw, Pxx_raw = signal.welch(signal_raw, FS, nperseg=2048)
    f_filtered, Pxx_filtered = signal.welch(signal_filtered, FS, nperseg=2048)

    # 5. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.suptitle("Filter Effect in the Frequency Domain (Power Spectral Density)", fontsize=16)
    
    # --- Plot 1: Raw Signal PSD ---
    ax1.semilogy(f_raw, Pxx_raw, label="Raw Signal", color='r', alpha=0.8)
    ax1.set_title(f"BEFORE: Raw {CHANNEL_TO_PLOT} (Note the 50Hz spike)")
    ax1.set_ylabel("Power / Frequency (dB/Hz) - Log Scale")
    ax1.axvline(NOTCH_FREQ, color='k', linestyle='--', label="50Hz Hum")
    ax1.axvspan(0, BAND_LOW, color='gray', alpha=0.3, label="Low-freq Noise")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Filtered Signal PSD ---
    ax2.semilogy(f_filtered, Pxx_filtered, label="Filtered Signal", color='b')
    ax2.set_title(f"AFTER: {CHANNEL_TO_PLOT} (Noise successfully removed)")
    ax2.set_ylabel("Power / Frequency (dB/Hz) - Log Scale")
    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.axvline(NOTCH_FREQ, color='k', linestyle='--', label="50Hz Hum (Removed)")
    ax2.axvspan(0, BAND_LOW, color='gray', alpha=0.3, label="Low-freq Noise (Removed)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Limit x-axis to 0-500Hz to see the details
    ax2.set_xlim(0, 500)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "psd_filter_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... This plot *proves* the filters are working. Look for the 50Hz spike in the top plot and see it's gone in the bottom plot.")

if __name__ == "__main__":
    plot_psd_comparison(RAW_FILE_TO_INSPECT)