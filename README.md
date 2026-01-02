# Proportional Myoelectric Control Core (v10.0)
**High-Fidelity 12-DOF Prosthetic Control via Temporal Feature-Engineered LSTMs**

## 🧬 Project Overview: The "Why"
[cite_start]This Research & Development initiative bridges the gap between affordable, discrete gesture classification and high-cost, continuous proportional control systems[cite: 19, 96, 1122]. [cite_start]By leveraging deep learning and advanced signal processing, this system predicts the continuous intent (0.0 to 1.0) for **12 Virtual Degrees of Freedom** using non-invasive 12-channel surface electromyography (sEMG)[cite: 64, 97, 698].



### 🏗️ Engineering Architecture
[cite_start]The project scientifically validated three competing hypotheses using the CRISP-DM methodology[cite: 20, 332, 1058]:

1.  [cite_start]**Phase 1 (Static Baseline):** Instantaneous sEMG snapshot processing using Non-negative Matrix Factorization (NMF)[cite: 112, 444, 513].
2.  [cite_start]**Phase 2 (The Champion):** Sequential modeling utilizing **Hudgins' Time-Domain features** (RMS, WL, ZC, SSC) fed into a **Stacked LSTM**[cite: 20, 122, 659, 1092].
3.  [cite_start]**Phase 3 (The Challenger):** End-to-End deep learning using **log-magnitude STFT Spectrograms** and a **2D-CNN**[cite: 20, 134, 793, 1019].

---

## 🛠️ Repository Directory & Logic Map
*Permanent technical documentation for the `PROSTHETIC_AI_CORE` structure:*

### 1. `src/data_pipeline/` (The Engine)
* [cite_start]**`convert_to_10hz_csv.py`**: Implements **Boundary-Aware Downsampling**[cite: 602]. [cite_start]It segments pure movement chunks based on the `restimulus` ground truth, discarding transition noise to ensure 100% label purity per 100ms window[cite: 614, 657].
* [cite_start]**`convert_to_tfrecord_v2.py`**: Efficiently serializes high-frequency data into binary **TFRecords** for optimized GPU throughput[cite: 582, 584].
* [cite_start]**`convert_to_spectrogram.py`**: Generates (12, 101, 101) time-frequency representations (STFT) for Phase 3 exploration[cite: 793, 795].

### 2. `src/analysis/` (The Brain)
* [cite_start]**`analyze_movement_clusters.py`**: Employs **K-Means (k=12)** and **PCA** to identify movement archetypes for stratified dataset balancing[cite: 748, 783, 784].
* [cite_start]**`analyze_10hz_data.py`**: Critical EDA that established "Rest is Not Zero," leading to the **Rest-Based Min-Max Normalization** strategy[cite: 364, 738, 739].

### 3. `src/models/` (The Architectures)
* [cite_start]**`train_sequential_v2.py`**: The **Stacked LSTM ($v4$)** training core[cite: 968]. [cite_start]It utilizes 128-unit layers and a `TimeDistributed` head to achieve the project-best **MAE of 0.1149**[cite: 973, 1092, 1129].
* [cite_start]**`train_spectrogram_v3.py`**: A high-capacity **2D-CNN Vision Pipeline** ($4.8M$ parameters) tested for automatic feature discovery[cite: 1019, 1042].

### 4. `config/` (Production Artifacts)
* [cite_start]**`motor_normalization_params.json`**: The digital twin parameters for the 12 Virtual Motors, enabling real-world hardware mapping (0.0-1.0 range)[cite: 368, 744, 1207].

---

## 📈 Performance Scorecard
| Rank | Model | Architecture | MAE | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 | **$v4_{lstm}$ (Phase 2)** | **Stacked LSTM** | **0.1149** | [cite_start]**+28.6%** [cite: 1111] |
| 🥈 | $v6_{cnn}$ (Phase 3) | 2D-CNN Spectro | 0.1312 | [cite_start]+18.5% [cite: 1111] |
| 🥉 | $v2_{ffn}$ (Phase 1) | Static Baseline | 0.1610 | [cite_start]Baseline [cite: 1111] |



---

## 💬 Interview Quick-Reference (The "Aha!" Moment)
**Question:** *"Why did the smaller LSTM beat the massive 4.8M parameter CNN?"*
[cite_start]**Answer:** *"It was a victory of **Data Efficiency over Raw Capacity**[cite: 1104, 1114]. In biomedical signal processing with finite datasets, smart feature engineering (Hudgins' set) provides a low-noise manifold that allows Recurrent Networks to focus entirely on temporal dependencies[cite: 324, 1114]. The CNN, despite its power, began to overfit on high-dimensional noise, whereas the LSTM generalized the underlying muscle firing patterns significantly better[cite: 1105, 1131]."*

---
[cite_start]**Developed as a Dedicated R&D Initiative at SYNC[cite: 13, 83].**