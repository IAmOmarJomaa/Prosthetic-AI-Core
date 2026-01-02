# Save this as plot_raw_glove_targets.py
# This script visualizes the 22 raw, uncalibrated glove sensors
# to demonstrate the problems of redundancy and non-zero rest.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import signal
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The 22 glove channels we will analyze
GLOVE_COLS = [f"glove_{i}" for i in range(1, 23)]
LABEL_COL = "restimulus"

# --- ⚙️ TIME SLICE TO PLOT ---
# We'll find a slice that includes both rest and movement
START_SAMPLE = 25000 
DURATION_SAMPLES = 16000 # 8 seconds (should cover a 5s move + 3s rest)

# --- Main Plotting Function ---
def plot_raw_glove(file_path):
    print(f"Loading raw glove data from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the glove and label columns
        df = pd.read_csv(file_path, usecols=GLOVE_COLS + [LABEL_COL])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the *slice* of the data
    stop_sample = START_SAMPLE + DURATION_SAMPLES
    data_slice = df.iloc[START_SAMPLE:stop_sample].copy()
    
    # Create a time axis in seconds
    time_axis_sec = (np.arange(len(data_slice)) / 2000.0)

    # 3. Create the plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]} # Make top plot bigger
    )
    plt.suptitle("Raw, Uncalibrated 22-Channel Glove Data (Phase 1 Target)", fontsize=16)
    
    # --- Plot 1: The 22 Glove Sensors ("Spaghetti Plot") ---
    ax1.set_title("Glove Sensors (Raw, Uncalibrated)")
    ax1.set_ylabel("Raw Sensor Value")
    ax1.grid(True, alpha=0.3)
    
    # Plot all 22 lines with some transparency
    for col in GLOVE_COLS:
        ax1.plot(time_axis_sec, data_slice[col], alpha=0.6)
    
    # Highlight two key problems
    ax1.annotate(
        "Problem 1: 'Rest is Not Zero'\n(Sensors rest at different, uncalibrated values)",
        xy=(0.5, np.mean(data_slice[GLOVE_COLS].iloc[100])), # Point to a rest section
        xytext=(1.5, -50), # Text location
        arrowprops=dict(facecolor='black', shrink=0.05),
        ha='left', va='center'
    )
    ax1.annotate(
        "Problem 2: Redundancy\n(Large groups of sensors move together)",
        xy=(3.5, np.mean(data_slice[GLOVE_COLS].iloc[7000])), # Point to a movement section
        xytext=(4.5, 200), # Text location
        arrowprops=dict(facecolor='black', shrink=0.05),
        ha='left', va='center'
    )

    # --- Plot 2: The Ground Truth Label ---
    ax2.plot(time_axis_sec, data_slice[LABEL_COL], label="Ground Truth Label (restimulus)", color='r', drawstyle='steps-post')
    ax2.set_title("Movement Label")
    ax2.set_ylabel("Movement ID")
    ax2.set_xlabel(f"Time (seconds)", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "raw_glove_target_visualization.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... This plot shows the 'messy' 22-sensor data we fed to the v2 model.")

if __name__ == "__main__":
    plot_raw_glove(RAW_FILE_TO_INSPECT)