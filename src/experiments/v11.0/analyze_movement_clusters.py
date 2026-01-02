# Save this as analyze_movement_clusters.py
# This script generates the final, corrected PCA cluster plot for Figure 3.13.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# --- ⚙️ CONFIGURATION ---
DATA_FILE = "10hz_feature_data.csv" # CRITICAL: Must be the final CSV file
NUM_GLOVE_SENSORS = 22
# CRITICAL FIX: The report now uses 12 movement archetypes (12 clusters + 1 rest class)
NUM_MOVEMENT_CLUSTERS = 12 

# --- 1. LOAD AND PREPARE THE DATASET ---
def load_and_prepare_data(file_path):
    """Loads and scales the glove data for clustering."""
    print(f"Loading data from {file_path} for clustering...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return None, None
    
    df = pd.read_csv(file_path)
    
    # Use only movement samples (restimulus != 0)
    move_df = df[df['restimulus'] != 0].copy()
    glove_cols = [f'glove_{i}' for i in range(1, NUM_GLOVE_SENSORS + 1)]

    if move_df.empty:
        print("❌ FATAL: No movement data found in CSV.")
        return None, None

    glove_data = move_df[glove_cols]
    
    # Normalize the data (important for PCA/KMeans)
    scaler = StandardScaler()
    glove_data_scaled = scaler.fit_transform(glove_data)
    
    return move_df, glove_data_scaled

# --- 2. MOVEMENT CLUSTER ANALYSIS (PCA + K-Means) ---
def analyze_movement_clusters(move_df, glove_data_scaled):
    """
    Uses PCA and K-Means to find the 12 movement archetypes.
    """
    print(f"\n--- 🔬 Generating Figure 3.13 (k={NUM_MOVEMENT_CLUSTERS} Clustering) ---")
    
    # 1. Apply PCA to reduce from 22 dimensions to 2
    pca = PCA(n_components=2)
    glove_pca = pca.fit_transform(glove_data_scaled)
    
    print(f"   ... PCA Explained Variance (2 components): {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # 2. Apply K-Means clustering to find the archetypal movements
    print(f"   ... Applying K-Means to find {NUM_MOVEMENT_CLUSTERS} movement groups...")
    kmeans = KMeans(n_clusters=NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_df['cluster'] = kmeans.fit_predict(glove_data_scaled)
    
    # 3. Plot the clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=glove_pca[:, 0], 
        y=glove_pca[:, 1], 
        hue=move_df['cluster'],
        palette=sns.color_palette("hsv", n_colors=NUM_MOVEMENT_CLUSTERS), # Use enough colors for 12 clusters
        alpha=0.5, 
        s=10,
        legend='full'
    )
    plt.title(f"Movement Groups (PCA + K-Means Clustering, k={NUM_MOVEMENT_CLUSTERS})", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    output_file = "movement_clusters_pca.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Corrected cluster plot saved to: {output_file}")
    
# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING CLUSTER PLOT GENERATION (PHASE 2b)")
    print("=============================================================")
    
    move_df, glove_data_scaled = load_and_prepare_data(DATA_FILE)
    
    if move_df is not None:
        analyze_movement_clusters(move_df, glove_data_scaled)
        
    print("\n" + "=" * 60)
    print("🎉 PLOTTING COMPLETE!")
    print("   Ensure the new 'movement_clusters_pca.png' is used in Figure 3.13.")
    print("=" * 60)

if __name__ == "__main__":
    main()