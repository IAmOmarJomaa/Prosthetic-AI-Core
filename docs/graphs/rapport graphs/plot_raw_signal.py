# Save this as plot_raw_signal.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

# --- Main Plotting Function ---
def plot_raw_data(file_path):
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

    # 2. Create an "EMG Envelope"
    # We calculate the mean absolute value of all 12 channels
    # and then smooth it with a rolling window to see the "intent"
    df['emg_envelope'] = df[EMG_COLS].abs().mean(axis=1)
    df['emg_envelope'] = df['emg_envelope'].rolling(window=200, min_periods=1, center=True).mean()

    # 3. Get the 10-second slice of data we want to plot
    data_slice = df.iloc[START_SAMPLE:END_SAMPLE]
    time_axis = np.arange(len(data_slice)) / 2000.0 # Time in seconds

    # 4. Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # --- Plot 1: EMG Signal ---
    ax1.plot(time_axis, data_slice['emg_envelope'], label="EMG Envelope (12-Ch Avg)", color='b')
    ax1.set_title("EMG Signal Envelope (Muscle Activity)", fontsize=14)
    ax1.set_ylabel("Mean Absolute Value")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Glove Signal ---
    ax2.plot(time_axis, data_slice[GLOVE_TO_PLOT], label=f"{GLOVE_TO_PLOT} (Raw)", color='g')
    ax2.set_title(f"Glove Sensor (Example: {GLOVE_TO_PLOT})", fontsize=14)
    ax2.set_ylabel("Raw Sensor Value")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Restimulus Label ---
    ax3.plot(time_axis, data_slice[LABEL_COL], label="Movement Label", color='r', drawstyle='steps-post')
    ax3.set_title("Ground Truth Label (restimulus)", fontsize=14)
    ax3.set_ylabel("Movement ID")
    ax3.set_xlabel("Time (seconds)", fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "raw_signal_visualization.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... You can now open this PNG file to see your raw data.")

if __name__ == "__main__":
    plot_raw_data(RAW_FILE_TO_INSPECT)