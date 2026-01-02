import os
import pandas as pd
import numpy as np

# Glove column definitions (same as your TFRecord code)
GLOVE_DB7 = [f"glove_{i}" for i in range(1,19)]
GLOVE_DB23 = ["glove_1","glove_2","glove_3","glove_4","glove_5","glove_6","glove_8","glove_9","glove_11","glove_12","glove_13","glove_15","glove_16","glove_17","glove_19","glove_20","glove_21","glove_22"]

CHUNK_SIZE = 50000  # Larger = faster (adjust based on RAM)

def get_glove_cols(path):
    if "DB7" in path: return GLOVE_DB7
    if "DB2" in path or "DB3" in path: return GLOVE_DB23
    return None

def scan_file_fast(filepath):
    try:
        cols = get_glove_cols(filepath)
        if not cols: return

        for chunk in pd.read_csv(filepath, usecols=cols, chunksize=CHUNK_SIZE):
            # Check INF — vectorized, fast
            inf_cols = chunk.columns[np.isinf(chunk).any()].tolist()
            if inf_cols:
                print(f"💥 INF in {os.path.basename(filepath)} → columns: {inf_cols}")

            # Check NaN — vectorized, fast
            nan_cols = chunk.columns[chunk.isnull().any()].tolist()
            if nan_cols:
                print(f"🕳️  NaN in {os.path.basename(filepath)} → columns: {nan_cols}")

    except Exception:
        pass  # Silent on file read errors (assume not our problem)

# === MAIN ===
FILE_LIST_CSV = r"D:\stage\v5.0\logs\map\train_map.csv"
df = pd.read_csv(FILE_LIST_CSV)
for f in df['filename']:
    if os.path.exists(f):
        scan_file_fast(f)