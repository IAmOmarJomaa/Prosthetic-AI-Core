# Save this as plot_filtered_signal.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# We will plot a 10-second slice (20,000 samples)
START_SAMPLE = 20000 # Start at 10 seconds
END_SAMPLE = 40000   # End at 20 seconds

# The columns we want to inspect
EMG_COLS = [f"emg_{i}" for i in range(1, 13)]
GLOVE_COLS = [f"glove_{i}" for i in range(1, 23)]
LABEL_COL = "restimulus"

# A representative glove sensor to plot (glove_11 is middle finger)
GLOVE_TO_PLOT = "glove_11"

# --- ⚙️ FILTER PARAMETERS (Copied from our conversion script) ---
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
def plot_filtered_data(file_path):
    print(f"Loading raw data from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        print("Please update the RAW_FILE_TO_INSPECT variable in this script.")
        return

    try:
        # 1. Load only the columns we need
        use_cols = EMG_COLS + GLOVE_COLS + [LABEL_COL]
        df = pd.read_csv(file_path, usecols=lambda c: c in use_cols)
        print(f"   ... Loaded {len(df):,} total 2000Hz samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # --- NEW: Apply Filters ---
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    
    # Create a new DataFrame for the filtered data
    # This is faster and avoids Pandas warnings
    emg_filtered = pd.DataFrame(index=df.index)
    for col in EMG_COLS:
        sig = df[col].values # Get raw signal
        sig = signal.filtfilt(b_notch, a_notch, sig) # Apply notch filter
        sig = signal.filtfilt(b_band, a_band, sig)   # Apply bandpass filter
        emg_filtered[col] = sig
    print("   ... Filtering complete.")

    # 2. Create an "EMG Envelope" *from the filtered data*
    df['emg_envelope'] = emg_filtered[EMG_COLS].abs().mean(axis=1)
    df['emg_envelope'] = df['emg_envelope'].rolling(window=200, min_periods=1, center=True).mean()

    # 3. Get the 10-second slice of data we want to plot
    data_slice = df.iloc[START_SAMPLE:END_SAMPLE]
    time_axis = np.arange(len(data_slice)) / 2000.0 # Time in seconds

    # 4. Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # --- Plot 1: EMG Signal (Now Filtered) ---
    ax1.plot(time_axis, data_slice['emg_envelope'], label="FILTERED EMG Envelope (12-Ch Avg)", color='b')
    ax1.set_title("FILTERED EMG Signal Envelope (Muscle Activity)", fontsize=14)
    ax1.set_ylabel("Mean Absolute Value")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Glove Signal (Unchanged) ---
    ax2.plot(time_axis, data_slice[GLOVE_TO_PLOT], label=f"{GLOVE_TO_PLOT} (Raw)", color='g')
    ax2.set_title(f"Glove Sensor (Example: {GLOVE_TO_PLOT})", fontsize=14)
    ax2.set_ylabel("Raw Sensor Value")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Restimulus Label (Unchanged) ---
    ax3.plot(time_axis, data_slice[LABEL_COL], label="Movement Label", color='r', drawstyle='steps-post')
    ax3.set_title("Ground Truth Label (restimulus)", fontsize=14)
    ax3.set_ylabel("Movement ID")
    ax3.set_xlabel("Time (seconds)", fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "filtered_signal_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... You can now compare this to 'raw_signal_visualization.png'.")

if __name__ == "__main__":
    plot_filtered_data(RAW_FILE_TO_INSPECT)