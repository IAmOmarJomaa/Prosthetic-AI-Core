import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration for the Mock Plot ---
FS = 2000  # Original Sample Rate (Hz)
WINDOW_SIZE = 200  # 100ms window (2000 samples/s * 0.1s)
TIME_STEP = 1 / FS
N_FULL_CHUNKS = 2  # Two full chunks for each state
N_PARTIAL_CHUNK = 150  # 150 samples in the partial, discarded chunk

# --- Mock Data Generation ---
# 1. State 1 (e.g., Rest)
N1 = N_FULL_CHUNKS * WINDOW_SIZE
emg1 = np.random.normal(0, 0.5, N1) + 2  # Mean 2, low variance
# 2. Partial Discarded Chunk
N_partial = N_PARTIAL_CHUNK
emg_partial = np.random.normal(0, 0.5, N_partial) + 2
# 3. State 2 (e.g., Movement 1)
N2 = N_FULL_CHUNKS * WINDOW_SIZE
emg2 = np.random.normal(0, 2.0, N2) + 10  # Mean 10, high variance

# Combine all data
emg_full = np.concatenate([emg1, emg_partial, emg2])
N_total = len(emg_full)
time_sec = np.arange(N_total) * TIME_STEP

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})

# --- AXIS 1: Raw Signal and Downsampling Process ---
ax1.set_title(r"Boundary-Aware Downsampling Process ($\mathbf{2000 \ Hz \to 10 \ Hz}$) at Movement Transition", fontsize=14, pad=10)
ax1.plot(time_sec, emg_full, label='Filtered EMG Signal (Channel 0)', color='#1f77b4', linewidth=0.8)
ax1.set_ylabel('Amplitude (Z-Score)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-5, 15)

# 1. Draw the "Pure Chunk" boundary
N1_idx = N1
boundary_time = time_sec[N1_idx-1] + TIME_STEP
ax1.axvline(boundary_time, color='r', linestyle='--', alpha=0.7, label='Movement ID Boundary (restimulus change)')
ax1.text(boundary_time - 0.005, 14, 'Boundary', color='r', rotation=0, ha='right', fontsize=10)

# 2. Draw the 100ms Windows and highlight the discarded one
# Full Windows in State 1
for i in range(N_FULL_CHUNKS):
    start_time = i * WINDOW_SIZE * TIME_STEP
    end_time = (i + 1) * WINDOW_SIZE * TIME_STEP
    ax1.axvline(start_time, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(end_time, color='gray', linestyle=':', alpha=0.5)
    ax1.text((start_time + end_time) / 2, 12.5, f'Win {i+1}', ha='center', fontsize=9, color='k')

# Partial Discarded Window
start_partial_time = N1 * TIME_STEP
end_partial_time = (N1 + N_PARTIAL_CHUNK) * TIME_STEP
ax1.axvspan(start_partial_time, end_partial_time, color='red', alpha=0.1, label='Discarded Window')
ax1.text((start_partial_time + end_partial_time) / 2, 13.5, 'Discarded\nRemainder', ha='center', fontsize=9, color='r', weight='bold')

# Full Windows in State 2
end_partial_idx = N1 + N_PARTIAL_CHUNK
for i in range(N_FULL_CHUNKS):
    start_time = (end_partial_idx + i * WINDOW_SIZE) * TIME_STEP
    end_time = (end_partial_idx + (i + 1) * WINDOW_SIZE) * TIME_STEP
    ax1.axvline(start_time, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(end_time, color='gray', linestyle=':', alpha=0.5)
    ax1.text((start_time + end_time) / 2, 12.5, f'Win {i+3}', ha='center', fontsize=9, color='k')

# 3. Add the restimulus labels
ax1.text(time_sec[N1_idx // 2], 14, r'State A ($\mathbf{restimulus = 0}$)', ha='center', fontsize=12, color='green', weight='bold')
ax1.text((time_sec[end_partial_idx] + time_sec[-1]) / 2, 14, r'State B ($\mathbf{restimulus = 5}$)', ha='center', fontsize=12, color='darkorange', weight='bold')


# --- AXIS 2: Final 10Hz Feature Data ---
# Simulate the 10Hz feature data (RMS for simplicity)
rms_vals = [np.mean(emg_full[0:200]**2)**0.5, np.mean(emg_full[200:400]**2)**0.5, 
            np.mean(emg_full[550:750]**2)**0.5, np.mean(emg_full[750:950]**2)**0.5]
rms_time = [time_sec[100], time_sec[300], time_sec[650], time_sec[850]]
group_ids = ['A-1', 'A-2', 'B-1', 'B-2']
group_colors = ['green', 'green', 'darkorange', 'darkorange']

# Plot the 10Hz RMS Feature and Group IDs
ax2.scatter(rms_time, rms_vals, 
            s=100, marker='D', color=group_colors, 
            zorder=3, label='10Hz Feature Point (RMS)')

ax2.set_ylabel('Feature Value', fontsize=12)
ax2.set_xlabel(r'Time (seconds)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_yticks([])
ax2.set_ylim(0, 8)

# Custom Group ID Labels
for i in range(len(rms_time)):
    ax2.text(rms_time[i], rms_vals[i] + 0.5, group_ids[i], ha='center', va='bottom', fontsize=10, weight='bold', color='k')

# Add Discarded Label for clarity on the 10Hz sequence
center_discarded_time = (start_partial_time + end_partial_time) / 2
ax2.axvspan(start_partial_time, end_partial_time, color='red', alpha=0.1)
ax2.text(center_discarded_time, 0.5, 'Discarded Period\n(No 10 Hz Sample)', ha='center', va='bottom', fontsize=9, color='r', weight='bold')

plt.tight_layout()
plt.show()