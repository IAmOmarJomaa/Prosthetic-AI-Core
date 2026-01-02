# Save this as analyze_10hz_data.py
# This script loads the 10Hz feature data, creates new "Virtual Motor"
# columns, and calculates their true Min (rest) and Max (movement) values.

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List

# --- ⚙️ CONFIGURATION ---
# This is the 10Hz CSV file you just created
DATA_FILE = "10hz_feature_data.csv" 

# 1. DEFINE VIRTUAL MOTOR GROUPINGS
# This is based on the *actual* sensor map from the paper (sdata201453.pdf)
# We use the 1-indexed column names from your new CSV.
VIRTUAL_MOTORS: Dict[str, List[str]] = {
    
    # motor_thumb_flex: Average of the 2 thumb flexion sensors
    "motor_thumb_flex": [f"glove_{i}" for i in [3, 4]],
    
    # motor_index_flex: Average of the 3 index finger flexion sensors
    "motor_index_flex": [f"glove_{i}" for i in [6, 7, 8]],
    
    # motor_middle_flex: Average of the 3 middle finger flexion sensors
    "motor_middle_flex": [f"glove_{i}" for i in [10, 11, 12]],
    
    # motor_ring_flex: Average of the 3 ring finger flexion sensors
    "motor_ring_flex": [f"glove_{i}" for i in [14, 15, 16]],
    
    # motor_pinky_flex: Average of the 3 pinky finger flexion sensors
    "motor_pinky_flex": [f"glove_{i}" for i in [18, 19, 20]],
    
    # motor_thumb_abduct: The single sensor for thumb spread
    "motor_thumb_abduct": ["glove_2"],
    
    # motor_wrist_flex: The single sensor for wrist flexion/extension
    "motor_wrist_flex": ["glove_21"]
    
    # Note: We are ignoring roll, side-to-side wrist, and finger spreading
    # to create a simpler, more robust 7-motor model.
}

# --- 1. LOAD THE DATASET ---
def load_data(file_path):
    """Loads the 10Hz data from the CSV file."""
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        print("Please ensure the '10hz_feature_data.csv' file is in this directory.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
        return df
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return None

# --- 2. CREATE VIRTUAL MOTOR COLUMNS ---
def create_virtual_motors(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Averages the raw sensor columns to create the new "motor" columns.
    """
    print("\n--- 🔬 1. Creating 7 Virtual Motors ---")
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        print(f"   ... Creating '{motor_name}' from sensor(s): {sensor_list}")
        # Calculate the mean of the sensors in the list, row by row
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
        
    return df, motor_cols

# --- 3. ANALYZE MOTOR MIN/MAX VALUES ---
def analyze_motor_ranges(df: pd.DataFrame, motor_cols: List[str]):
    """
    Finds the "Min" (average rest) and "Max" (99th percentile movement)
    for each new virtual motor.
    """
    print("\n--- 🔬 2. Analyzing Motor Ranges ---")
    
    # Separate Rest and Movement data
    rest_df = df[df['restimulus'] == 0]
    move_df = df[df['restimulus'] != 0]
    
    if rest_df.empty or move_df.empty:
        print(f"   ⚠️ Warning: Data sample is missing 'rest' (found {len(rest_df)}) or 'movement' (found {len(move_df)}) data.")
        print("   ... Analysis will be incomplete. Please use a larger data sample for the final run.")
        # We can still proceed with what we have for this test run
    
    # Calculate "Min" (the average value at rest)
    rest_mins = rest_df[motor_cols].mean()
    
    # Calculate "Max" (the 99th percentile of movement)
    # Using 99th percentile is more robust to outliers than a hard .max()
    move_maxs = move_df[motor_cols].quantile(0.99)
    
    # Combine into a final report
    analysis_report = pd.DataFrame({
        "Rest_Value_Min": rest_mins,
        "Move_Value_Max_99pct": move_maxs
    })
    
    # Add a "Range" column for our information
    analysis_report["Usable_Range"] = analysis_report["Move_Value_Max_99pct"] - analysis_report["Rest_Value_Min"]

    print("\n✅ ANALYSIS COMPLETE. These are the values for our 0-to-1 normalization:\n")
    print(analysis_report.to_string(float_format="%.2f"))
    
    # Save this report to a file for our next script
    report_file = "motor_normalization_params.json"
    analysis_report.to_json(report_file, orient='index', indent=2)
    print(f"\n   ... Analysis saved to: {report_file}")


# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING VIRTUAL MOTOR ANALYSIS (PHASE 2b)")
    print("=" * 60)
    
    df = load_data(DATA_FILE)
    
    if df is not None:
        df, motor_cols = create_virtual_motors(df)
        analyze_motor_ranges(df, motor_cols)
        
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE!")
        print(f"   Please review the printed table and the '{report_file}' file.")
        print("   These Min/Max values are what we will use to build our final, normalized TFRecord dataset.")
        print("=" * 60)

if __name__ == "__main__":
    main()