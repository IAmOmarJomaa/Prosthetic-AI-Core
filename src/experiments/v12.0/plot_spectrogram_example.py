# Save this as plot_spectrogram_example.py
# Generates the Spectrogram Image visualization required for the Phase 3 section (Figure 3.14)

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os

# --- ⚙️ CONFIGURATION ---
# IMPORTANT: Use a short file path for fast plotting
EXAMPLE_FILE_PATH = r"S1_E1_A1.csv" # CRITICAL: Change this path to a real, accessible CSV
START_TIME_S = 10.0 # Start 5-second segment at this timestamp (10 seconds)

# Spectrogram Parameters (MUST match convert_to_spectrogram.py)
FS = 2000.0 
SEQUENCE_S = 5.0
SEQUENCE_SAMPLES = int(FS * SEQUENCE_S) # 10,000 samples
STFT_WINDOW_SAMPLES = 200
STFT_OVERLAP_SAMPLES = 100 
NUM_CHANNELS = 12

# Filter Design (MUST match convert_to_spectrogram.py)
def design_filters(fs: float):
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs)
    nyq = 0.5 * fs
    low = 20.0 / nyq
    high = 450.0 / nyq
    b_band, a_band = signal.butter(4, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def compute_and_plot_spectrograms(emg_chunk: np.ndarray, file_name: str):
    """
    Computes log-magnitude spectrograms for all 12 channels and plots a 4x3 grid.
    """
    all_spectrograms = []
    
    for i in range(NUM_CHANNELS):
        f, t, Sxx = signal.stft(
            emg_chunk[:, i],
            fs=FS,
            nperseg=STFT_WINDOW_SAMPLES,
            noverlap=STFT_OVERLAP_SAMPLES
        )
        Sxx_mag = np.abs(Sxx)
        all_spectrograms.append(np.log(Sxx_mag + 1e-6))
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('Spectrogram Transformation for 5-Second Segment (Phase 3 Input)', fontsize=14)
    
    # Find global max/min for consistent color mapping
    vmin = min(np.min(spec) for spec in all_spectrograms)
    vmax = max(np.max(spec) for spec in all_spectrograms)

    for i, ax in enumerate(axes.flat):
        if i < NUM_CHANNELS:
            im = ax.pcolormesh(t, f, all_spectrograms[i], shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Ch {i+1}', fontsize=8)
            ax.set_ylim([0, 500])
        else:
            ax.set_visible(False)
            
    # Final cleanup
    fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize=12)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Log Magnitude (dB)')
    
    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])
    plt.savefig(file_name, dpi=150)
    plt.close()
    print(f"   ✅ Spectrogram visualization saved to: {file_name}")

# --- MAIN EXECUTION ---
def main():
    if not os.path.exists(EXAMPLE_FILE_PATH):
        print(f"❌ ERROR: Example file not found at {EXAMPLE_FILE_PATH}. Please update the path.")
        return

    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    
    # Load raw data
    df = pd.read_csv(EXAMPLE_FILE_PATH, usecols=[f"emg_{i}" for i in range(1, 13)])
    emg_data = df.values
    
    # Calculate indices for the 5s chunk
    start_index = int(START_TIME_S * FS)
    end_index = start_index + SEQUENCE_SAMPLES
    
    if end_index > len(emg_data):
        print(f"❌ ERROR: File too short for 5s sequence starting at {START_TIME_S}s.")
        return

    emg_chunk = emg_data[start_index:end_index].copy()
    
    # Filter the chunk
    for i in range(emg_chunk.shape[1]):
        emg_chunk[:, i] = signal.filtfilt(b_notch, a_notch, emg_chunk[:, i])
        emg_chunk[:, i] = signal.filtfilt(b_band, a_band, emg_chunk[:, i])
        
    compute_and_plot_spectrograms(emg_chunk, "spectrogram_transformation.png")
    
if __name__ == "__main__":
    main()