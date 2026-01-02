# Proportional Myoelectric Control Core (v10.0)
**High-Fidelity 12-DOF Prosthetic Control via Temporal Feature-Engineered LSTMs**

## 🧬 Project Overview: The "Why"
This Research & Development initiative bridges the gap between affordable, discrete gesture classification and high-cost, continuous proportional control systems. By leveraging deep learning and advanced signal processing, this system predicts the continuous intent (0.0 to 1.0) for **12 Virtual Degrees of Freedom** using non-invasive 12-channel surface electromyography (sEMG).



### 🏗️ Engineering Architecture
The project scientifically validated three competing hypotheses using the CRISP-DM methodology:

1.  **Phase 1 (Static Baseline):** Instantaneous sEMG snapshot processing using Non-negative Matrix Factorization (NMF).
2.  **Phase 2 (The Champion):** Sequential modeling utilizing **Hudgins' Time-Domain features** (RMS, WL, ZC, SSC) fed into a **Stacked LSTM**.
3.  **Phase 3 (The Challenger):** End-to-End deep learning using **log-magnitude STFT Spectrograms** and a **2D-CNN**.

---

## 🛠️ Repository Directory & Logic Map
*Permanent technical documentation for the `PROSTHETIC_AI_CORE` structure:*

### 1. `src/data_pipeline/` (The Engine)
**`convert_to_10hz_csv.py`**: Implements **Boundary-Aware Downsampling**. It segments pure movement chunks based on the `restimulus` ground truth, discarding transition noise to ensure 100% label purity per 100ms window.
**`convert_to_tfrecord_v2.py`**: Efficiently serializes high-frequency data into binary **TFRecords** for optimized GPU throughput.
**`convert_to_spectrogram.py`**: Generates (12, 101, 101) time-frequency representations (STFT) for Phase 3 exploration.

### 2. `src/analysis/` (The Brain)
**`analyze_movement_clusters.py`**: Employs **K-Means (k=12)** and **PCA** to identify movement archetypes for stratified dataset balancing.
**`analyze_10hz_data.py`**: Critical EDA that established "Rest is Not Zero," leading to the **Rest-Based Min-Max Normalization** strategy.

### 3. `src/models/` (The Architectures)
**`train_sequential_v2.py`**: The **Stacked LSTM ($v4$)** training coreIt utilizes 128-unit layers and a `TimeDistributed` head to achieve the project-best **MAE of 0.1149**.
**`train_spectrogram_v3.py`**: A high-capacity **2D-CNN Vision Pipeline** ($4.8M$ parameters) tested for automatic feature discovery.

### 4. `config/` (Production Artifacts)
**`motor_normalization_params.json`**: The digital twin parameters for the 12 Virtual Motors, enabling real-world hardware mapping (0.0-1.0 range).

---

## 📈 Performance Scorecard
| Rank | Model | Architecture | MAE | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 | **$v4_{lstm}$ (Phase 2)** | **Stacked LSTM** | **0.1149** | **+28.6%**  |
| 🥈 | $v6_{cnn}$ (Phase 3) | 2D-CNN Spectro | 0.1312 | +18.5%  |
| 🥉 | $v2_{ffn}$ (Phase 1) | Static Baseline | 0.1610 | Baseline  |



---
