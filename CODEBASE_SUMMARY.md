# Project Source Code Summary

## File: migrate_to_prod.py
**Path:** `C:\stage\stage\migrate_to_prod.py`

```python
import shutil
from pathlib import Path

# Paths
ROOT = Path(r"C:\stage\stage")
CORE = ROOT / "PROSTHETIC_AI_CORE"

def sync_research_proof():
    # Target only the high-value research results
    for v in ["v10.0", "v11.0", "v12.0"]:
        v_path = ROOT / v
        if not v_path.exists(): continue
        
        print(f"--- Processing {v} ---")
        
        # 1. Grab ALL unique scripts (even the experimental ones)
        for py_file in v_path.glob("*.py"):
            dest = CORE / "src" / "experiments" / v
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, dest / py_file.name)
            
        # 2. Grab the final datasets if they aren't too huge for your disk
        for tf_folder in v_path.glob("tfrecord_dataset*"):
            dest = CORE / "data" / "processed" / v
            if not dest.exists():
                print(f"Moving Dataset from {v}...")
                shutil.copytree(tf_folder, dest)

    print("✅ Research history and processed datasets synced to CORE.")

if __name__ == "__main__":
    sync_research_proof()
```

---

## File: DB\creation\create_train.py
**Path:** `C:\stage\stage\DB\creation\create_train.py`

```python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import NMF
import tensorflow as tf
import joblib

# Configuration
FILE_LIST_CSV = r"train_map.csv"
TFRecord_OUTPUT_DIR = r"F:\DB\tfrecords"
LOG_OUTPUT = r"F:\DB\processing_log.csv"
ERROR_LOG = r"F:\DB\processing_errors.log"
CHUNK_SIZE = 10000
DATA_DTYPE = 'float32'
EXAMPLES_PER_TFRECORD = 50000  # Number of examples per TFRecord file

# Define the correct base path for your files
CORRECT_BASE_PATH = r"E:\DB"

# Define default values for each glove sensor based on typical rest positions
DEFAULT_GLOVE_VALUES = {
    "glove_1": 0.0, "glove_2": 0.0, "glove_3": 0.0, "glove_4": 0.0, "glove_5": 0.0,
    "glove_6": 0.0, "glove_7": 0.0, "glove_8": 0.0, "glove_9": 0.0, "glove_10": 0.0,
    "glove_11": 0.0, "glove_12": 0.0, "glove_13": 0.0, "glove_14": 0.0, "glove_15": 0.0,
    "glove_16": 0.0, "glove_17": 0.0, "glove_18": 0.0, "glove_19": 0.0, "glove_20": 0.0,
    "glove_21": 0.0, "glove_22": 0.0
}

# Define glove column mappings for different databases
GLOVE_COLUMNS_DB7 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_7", "glove_8", "glove_9", "glove_10",
    "glove_11", "glove_12", "glove_13", "glove_14", "glove_15",
    "glove_16", "glove_17", "glove_18"
]

GLOVE_COLUMNS_DB23 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_8", "glove_9", "glove_11", "glove_12",
    "glove_13", "glove_15", "glove_16", "glove_17", "glove_19",
    "glove_20", "glove_21", "glove_22"
]

def correct_file_path(original_path):
    """Correct the file path by updating the base path"""
    if "DB2" in original_path or "DB3" in original_path or "DB7" in original_path:
        parts = original_path.split("DB\\", 1)
        if len(parts) > 1:
            relative_path = parts[1]
            return os.path.join(CORRECT_BASE_PATH, relative_path)
    return original_path

def get_db_type(file_path):
    """Determine DB type from path."""
    if "DB7" in file_path:
        return "DB7"
    elif "DB2" in file_path or "DB3" in file_path:
        return "DB23"
    else:
        raise ValueError(f"Unknown DB type in path: {file_path}")

def get_glove_columns(db_type):
    """Return glove column names based on database type."""
    if db_type == "DB7":
        return GLOVE_COLUMNS_DB7
    elif db_type == "DB23":
        return GLOVE_COLUMNS_DB23
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

def load_nmf_model(model_path):
    return joblib.load(model_path)

def create_tf_example(nmf_features, glove_features):
    """Create a TF Example from NMF and glove features."""
    feature_dict = {}
    
    # Add NMF features
    for i in range(nmf_features.shape[1]):
        feature_dict[f'nmf_{i}'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=nmf_features[:, i]))
    
    # Add glove features
    for i in range(glove_features.shape[1]):
        feature_dict[f'glove_{i}'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=glove_features[:, i]))
    
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def process_file_with_nmf(file_path, nmf_model, glove_columns, chunk_size=CHUNK_SIZE):
    """
    Process a single CSV file by applying NMF to the first 12 columns and
    extracting specified glove columns.
    
    Returns:
    - List of data chunks (each is a tuple of nmf_features and glove_features)
    - Number of rows processed
    - Boolean indicating if null values were found
    """
    try:
        # Get the total number of rows for progress tracking
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
        
        # Initialize variables
        data_chunks = []
        has_nulls = False
        rows_processed = 0
        
        # Read file in chunks
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size, dtype=np.float32), 
                         total=int(total_rows/chunk_size)+1, desc=f"Processing {os.path.basename(file_path)}"):
            # Extract first 12 columns for NMF
            emg_data = chunk.iloc[:, :12].values
            
            # Apply NMF transformation
            nmf_transformed = nmf_model.transform(emg_data)
            
            # Extract glove columns
            glove_data = []
            for col in glove_columns:
                if col in chunk.columns:
                    # Check for null values and replace with defaults
                    null_mask = chunk[col].isnull()
                    if null_mask.any():
                        has_nulls = True
                    glove_data.append(chunk[col].fillna(DEFAULT_GLOVE_VALUES[col]).values)
                else:
                    # Column missing, use default values
                    has_nulls = True
                    glove_data.append(np.full(len(chunk), DEFAULT_GLOVE_VALUES[col]))
            
            # Combine glove data
            glove_matrix = np.column_stack(glove_data)
            
            # Store the processed chunk
            data_chunks.append((nmf_transformed, glove_matrix))
            rows_processed += len(chunk)
        
        return data_chunks, rows_processed, has_nulls
        
    except Exception as e:
        raise Exception(f"Error processing {file_path}: {str(e)}")

def main():
    print("Loading file list...")
    df_log = pd.read_csv(FILE_LIST_CSV)
    
    # Correct file paths
    df_log['corrected_filename'] = df_log['filename'].apply(correct_file_path)
    
    # Create output directory
    os.makedirs(TFRecord_OUTPUT_DIR, exist_ok=True)
    
    # Load NMF model (update path to your trained model)
    nmf_model_path = r"D:\DB\nmf_model.h5"  # Update this path
    print("Loading NMF model...")
    nmf_model = load_nmf_model("nmf_model.pkl")
    n_components = nmf_model.components_.shape[0]
    
    # Create error log
    error_log = open(ERROR_LOG, 'w')
    error_log.write("File processing errors:\n")
    
    # Process each file in order
    current_row = 0
    processing_log = []
    files_with_errors = []
    files_with_nulls = []
    
    # TFRecord file counter
    tfrecord_counter = 0
    examples_in_current_file = 0
    writer = None
    
    for idx, row in tqdm(df_log.iterrows(), total=len(df_log), desc="Processing Files"):
        original_path = row['filename']
        file_path = row['corrected_filename']
        expected_count = row['rows_extracted']

        print(f"Processing: {os.path.basename(file_path)}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            print(f"ERROR: {error_msg}")
            error_log.write(f"{error_msg}\n")
            files_with_errors.append(file_path)
            continue
        
        try:
            # Determine DB type and get appropriate glove columns
            db_type = get_db_type(file_path)
            glove_columns = get_glove_columns(db_type)
            
            # Process the file
            data_chunks, rows_processed, has_nulls = process_file_with_nmf(
                file_path, nmf_model, glove_columns
            )
            
            # Write to TFRecord files
            for nmf_chunk, glove_chunk in data_chunks:
                # Create a new TFRecord file if needed
                if writer is None or examples_in_current_file >= EXAMPLES_PER_TFRECORD:
                    if writer:
                        writer.close()
                    tfrecord_path = os.path.join(TFRecord_OUTPUT_DIR, f"data_{tfrecord_counter:04d}.tfrecord")
                    writer = tf.io.TFRecordWriter(tfrecord_path)
                    tfrecord_counter += 1
                    examples_in_current_file = 0
                
                # Write examples to TFRecord
                for i in range(len(nmf_chunk)):
                    # Create example
                    nmf_features = nmf_chunk[i:i+1]  # Keep as 2D array for consistency
                    glove_features = glove_chunk[i:i+1]  # Keep as 2D array for consistency
                    example = create_tf_example(nmf_features, glove_features)
                    
                    # Write to TFRecord
                    writer.write(example.SerializeToString())
                    examples_in_current_file += 1
            
            # Update logs
            if has_nulls:
                files_with_nulls.append(file_path)
            
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': rows_processed,
                'status': 'SUCCESS',
                'has_nulls': has_nulls
            })
            
            print(f"✓ Processed {rows_processed} rows from {os.path.basename(file_path)}")
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            print(f"ERROR: {error_msg}")
            error_log.write(f"{error_msg}\n")
            files_with_errors.append(file_path)
            
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'status': f'ERROR: {str(e)}',
                'has_nulls': False
            })
    
    # Close the last TFRecord writer
    if writer:
        writer.close()
    
    # Close error log
    error_log.close()
    
    # Save processing log
    log_df = pd.DataFrame(processing_log)
    log_df.to_csv(LOG_OUTPUT, index=False)
    print(f"Processing log saved to: {LOG_OUTPUT}")
    
    # Final report
    total_processed = sum(log['rows_processed'] for log in processing_log if log['status'] == 'SUCCESS')
    print(f"Total rows processed: {total_processed}")
    print(f"TFRecord files created: {tfrecord_counter}")
    
    if files_with_errors:
        print(f"⚠️  {len(files_with_errors)} files had errors during processing:")
        for file in files_with_errors:
            print(f"  - {file}")
        print(f"See {ERROR_LOG} for details.")
    else:
        print("✅ All files processed successfully.")
    
    if files_with_nulls:
        print(f"ℹ️  {len(files_with_nulls)} files contained null values:")
        for file in files_with_nulls:
            print(f"  - {file}")
    
    # Create metadata file
    metadata = {
        'total_rows': total_processed,
        'nmf_components': n_components,
        'glove_columns': str(GLOVE_COLUMNS_DB7),
        'tfrecord_files': tfrecord_counter,
        'examples_per_tfrecord': EXAMPLES_PER_TFRECORD
    }
    
    with open(os.path.join(TFRecord_OUTPUT_DIR, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"TFRecord files saved to: {TFRecord_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

---

## File: DB\creation\creation_final.py
**Path:** `C:\stage\stage\DB\creation\creation_final.py`

```python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Configuration
FILE_LIST_CSV = r"train_map.csv"
TFRecord_OUTPUT_DIR = r"F:\DB\tfrecords"
LOG_OUTPUT = r"F:\DB\processing_log.csv"
ERROR_LOG = r"F:\DB\processing_errors.log"
CHUNK_SIZE = 10000
EXAMPLES_PER_TFRECORD = 50000

# Define the correct base path for your files
CORRECT_BASE_PATH = r"E:\DB"

# Define glove column mappings for different databases
GLOVE_COLUMNS_DB7 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_7", "glove_8", "glove_9", "glove_10",
    "glove_11", "glove_12", "glove_13", "glove_14", "glove_15",
    "glove_16", "glove_17", "glove_18"
]

GLOVE_COLUMNS_DB23 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_8", "glove_9", "glove_11", "glove_12",
    "glove_13", "glove_15", "glove_16", "glove_17", "glove_19",
    "glove_20", "glove_21", "glove_22"
]

def correct_file_path(original_path):
    if "DB2" in original_path or "DB3" in original_path or "DB7" in original_path:
        parts = original_path.split("DB\\", 1)
        if len(parts) > 1:
            relative_path = parts[1]
            return os.path.join(CORRECT_BASE_PATH, relative_path)
    return original_path

def get_db_type(file_path):
    if "DB7" in file_path:
        return "DB7"
    elif "DB2" in file_path or "DB3" in file_path:
        return "DB23"
    else:
        raise ValueError(f"Unknown DB type in path: {file_path}")

def get_glove_columns(db_type):
    if db_type == "DB7":
        return GLOVE_COLUMNS_DB7
    elif db_type == "DB23":
        return GLOVE_COLUMNS_DB23
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

def create_tf_example(glove_row):
    feature_dict = {
        f'glove_{i}': tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(x)]))
        for i, x in enumerate(glove_row)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def process_file_glove_only(file_path, glove_columns, chunk_size=CHUNK_SIZE):
    try:
        total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1
        data_chunks = []
        rows_processed = 0

        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size),
                         total=int(total_rows/chunk_size)+1,
                         desc=f"Processing {os.path.basename(file_path)}"):
            missing_cols = [col for col in glove_columns if col not in chunk.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
            glove_data = chunk[glove_columns].values.astype(np.float32)
            data_chunks.append(glove_data)
            rows_processed += len(chunk)
        return data_chunks, rows_processed
    except Exception as e:
        raise Exception(f"Error processing {file_path}: {str(e)}")

def main():
    print("Loading file list...")
    df_log = pd.read_csv(FILE_LIST_CSV)
    df_log['corrected_filename'] = df_log['filename'].apply(correct_file_path)
    os.makedirs(TFRecord_OUTPUT_DIR, exist_ok=True)

    processing_log = []
    files_with_errors = []
    tfrecord_mapping_log = []  # NEW: logs which source file rows went into which TFRecord

    tfrecord_counter = 0
    examples_in_current_file = 0
    writer = None
    current_tfrecord_path = ""

    global_row_offset = 0  # Tracks absolute row index across all files

    for idx, row in tqdm(df_log.iterrows(), total=len(df_log), desc="Processing Files"):
        original_path = row['filename']
        file_path = row['corrected_filename']
        expected_count = row.get('rows_extracted', -1)

        print(f"Processing: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            with open(ERROR_LOG, 'a') as elog:
                elog.write(f"{error_msg}\n")
            files_with_errors.append(file_path)
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'status': 'FILE_NOT_FOUND',
                'has_nulls': False
            })
            continue

        try:
            db_type = get_db_type(file_path)
            glove_columns = get_glove_columns(db_type)
            print(f"  → DB Type: {db_type}, Using {len(glove_columns)} glove columns")

            data_chunks, rows_processed = process_file_glove_only(file_path, glove_columns)

            file_start_global = global_row_offset
            file_end_global = global_row_offset + rows_processed - 1

            local_row_in_file = 0

            for chunk in data_chunks:
                for glove_row in chunk:
                    if writer is None or examples_in_current_file >= EXAMPLES_PER_TFRECORD:
                        if writer:
                            writer.close()
                        current_tfrecord_path = os.path.join(TFRecord_OUTPUT_DIR, f"data_{tfrecord_counter:04d}.tfrecord")
                        writer = tf.io.TFRecordWriter(current_tfrecord_path)
                        tfrecord_counter += 1
                        examples_in_current_file = 0

                    example = create_tf_example(glove_row)
                    writer.write(example.SerializeToString())
                    examples_in_current_file += 1

                    # Log mapping: this source file's row → TFRecord file + global index
                    tfrecord_mapping_log.append({
                        'source_file': file_path,
                        'source_row_local': local_row_in_file,
                        'source_row_global': global_row_offset,
                        'tfrecord_file': os.path.basename(current_tfrecord_path),
                        'tfrecord_example_index': examples_in_current_file - 1,
                        'tfrecord_file_index': tfrecord_counter - 1
                    })

                    local_row_in_file += 1
                    global_row_offset += 1

            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': rows_processed,
                'status': 'SUCCESS',
                'has_nulls': False,
                'global_row_start': file_start_global,
                'global_row_end': file_end_global
            })
            print(f"✓ Processed {rows_processed} rows from {os.path.basename(file_path)}")

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            with open(ERROR_LOG, 'a') as elog:
                elog.write(f"{error_msg}\n")
            files_with_errors.append(file_path)
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'status': f'ERROR: {str(e)}',
                'has_nulls': False
            })

    if writer:
        writer.close()

    # Save all logs
    log_df = pd.DataFrame(processing_log)
    log_df.to_csv(LOG_OUTPUT, index=False)

    mapping_df = pd.DataFrame(tfrecord_mapping_log)
    mapping_log_path = os.path.join(TFRecord_OUTPUT_DIR, "tfrecord_mapping.csv")
    mapping_df.to_csv(mapping_log_path, index=False)
    print(f"TFRecord ↔ Source file mapping saved to: {mapping_log_path}")

    total_processed = sum(log['rows_processed'] for log in processing_log if log['status'] == 'SUCCESS')
    print(f"Total rows processed: {total_processed}")
    print(f"TFRecord files created: {tfrecord_counter}")

    if files_with_errors:
        print(f"⚠️  {len(files_with_errors)} files had errors. See {ERROR_LOG}")
    else:
        print("✅ All files processed successfully.")

    metadata = {
        'total_rows': total_processed,
        'glove_columns_DB7': GLOVE_COLUMNS_DB7,
        'glove_columns_DB23': GLOVE_COLUMNS_DB23,
        'tfrecord_files': tfrecord_counter,
        'examples_per_tfrecord': EXAMPLES_PER_TFRECORD,
        'output_feature_names': [f'glove_{i}' for i in range(len(GLOVE_COLUMNS_DB7))]
    }

    with open(os.path.join(TFRecord_OUTPUT_DIR, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2)

    print(f"TFRecord files saved to: {TFRecord_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

---

## File: DB\creation\creation_final1.py
**Path:** `C:\stage\stage\DB\creation\creation_final1.py`

```python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import gc
import logging
import json
from datetime import datetime

# Configuration
FILE_LIST_CSV = r"train_map.csv"
TFRecord_OUTPUT_DIR = r"F:\DB\tfrecords"
LOG_OUTPUT = r"F:\DB\processing_log.csv"
ERROR_LOG = r"F:\DB\processing_errors.log"
CHECKPOINT_FILE = r"F:\DB\processing_checkpoint.json"
CHUNK_SIZE = 5000
EXAMPLES_PER_TFRECORD = 50000

# Define the correct base path for your files
CORRECT_BASE_PATH = r"E:\DB"

# Define glove column mappings for different databases
GLOVE_COLUMNS_DB7 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_7", "glove_8", "glove_9", "glove_10",
    "glove_11", "glove_12", "glove_13", "glove_14", "glove_15",
    "glove_16", "glove_17", "glove_18"
]

GLOVE_COLUMNS_DB23 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_8", "glove_9", "glove_11", "glove_12",
    "glove_13", "glove_15", "glove_16", "glove_17", "glove_19",
    "glove_20", "glove_21", "glove_22"
]

# Set up logging
logging.basicConfig(
    filename=ERROR_LOG, 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def correct_file_path(original_path):
    if "DB2" in original_path or "DB3" in original_path or "DB7" in original_path:
        parts = original_path.split("DB\\", 1)
        if len(parts) > 1:
            relative_path = parts[1]
            return os.path.join(CORRECT_BASE_PATH, relative_path)
    return original_path

def get_db_type(file_path):
    if "DB7" in file_path:
        return "DB7"
    elif "DB2" in file_path or "DB3" in file_path:
        return "DB23"
    else:
        raise ValueError(f"Unknown DB type in path: {file_path}")

def get_glove_columns(db_type):
    if db_type == "DB7":
        return GLOVE_COLUMNS_DB7
    elif db_type == "DB23":
        return GLOVE_COLUMNS_DB23
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

def clean_dataframe(df, glove_columns):
    """Clean DataFrame by replacing infs and NaNs with zeros"""
    df_clean = df[glove_columns].copy()
    
    # Replace infinite values with zero
    df_clean = df_clean.replace([np.inf, -np.inf], 0)
    
    # Fill NaN values with zero
    df_clean = df_clean.fillna(0)
    
    return df_clean.values.astype(np.float32)

def create_tf_example(glove_row):
    feature_dict = {
        f'glove_{i}': tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(x)]))
        for i, x in enumerate(glove_row)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def process_file_glove_only(file_path, glove_columns, chunk_size=CHUNK_SIZE):
    try:
        # Get total rows more efficiently
        with open(file_path, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1
        
        rows_processed = 0
        rows_cleaned = 0
        
        # Process file in chunks and yield cleaned data incrementally
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False),
                         total=int(total_rows/chunk_size)+1,
                         desc=f"Processing {os.path.basename(file_path)}"):
            
            # Check for required columns
            missing_cols = [col for col in glove_columns if col not in chunk.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
            
            # Clean the data (handle infs and NaNs)
            glove_data = clean_dataframe(chunk, glove_columns)
            
            # Count how many rows had issues
            original_count = len(chunk)
            cleaned_count = len(glove_data)
            if cleaned_count < original_count:
                rows_cleaned += (original_count - cleaned_count)
            
            rows_processed += cleaned_count
            
            # Yield the cleaned data instead of accumulating it
            yield glove_data
            
        # Return total counts after processing all chunks
        return rows_processed, rows_cleaned
        
    except Exception as e:
        raise Exception(f"Error processing {file_path}: {str(e)}")

def load_checkpoint():
    """Load processing checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        "processed_files": [],
        "current_tfrecord_counter": 0,
        "global_row_offset": 0,
        "start_time": datetime.now().isoformat()
    }

def save_checkpoint(checkpoint_data):
    """Save processing checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def main():
    print("Loading file list...")
    df_log = pd.read_csv(FILE_LIST_CSV)
    df_log['corrected_filename'] = df_log['filename'].apply(correct_file_path)
    os.makedirs(TFRecord_OUTPUT_DIR, exist_ok=True)

    # Load checkpoint and existing logs
    checkpoint = load_checkpoint()
    processed_files = set(checkpoint.get("processed_files", []))
    
    # Load existing log if it exists
    processing_log = []
    if os.path.exists(LOG_OUTPUT):
        try:
            existing_log = pd.read_csv(LOG_OUTPUT)
            processing_log = existing_log.to_dict('records')
            # Add files from log to processed_files set
            for log_entry in processing_log:
                if log_entry['status'] == 'SUCCESS':
                    processed_files.add(log_entry['filename'])
        except:
            print("Warning: Could not read existing log file. Starting fresh.")
    
    files_with_errors = []
    tfrecord_mapping_log = []

    tfrecord_counter = checkpoint.get("current_tfrecord_counter", 0)
    global_row_offset = checkpoint.get("global_row_offset", 0)
    examples_in_current_file = 0
    writer = None
    current_tfrecord_path = ""

    # Get the next available TFRecord file number if not starting from checkpoint
    if tfrecord_counter == 0 and os.path.exists(TFRecord_OUTPUT_DIR):
        existing_files = [f for f in os.listdir(TFRecord_OUTPUT_DIR) if f.startswith('data_') and f.endswith('.tfrecord')]
        if existing_files:
            tfrecord_counter = max([int(f.split('_')[1].split('.')[0]) for f in existing_files]) + 1

    # Create a progress bar for files
    progress_bar = tqdm(df_log.iterrows(), total=len(df_log), desc="Processing Files")
    
    for idx, row in progress_bar:
        original_path = row['filename']
        file_path = row['corrected_filename']
        expected_count = row.get('rows_extracted', -1)

        # Skip already processed files
        if file_path in processed_files:
            progress_bar.set_description(f"Skipping: {os.path.basename(file_path)}")
            continue

        progress_bar.set_description(f"Processing: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logging.error(error_msg)
            files_with_errors.append(file_path)
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'rows_cleaned': 0,
                'status': 'FILE_NOT_FOUND',
                'has_nulls': False,
                'timestamp': datetime.now().isoformat()
            })
            continue

        try:
            db_type = get_db_type(file_path)
            glove_columns = get_glove_columns(db_type)
            progress_bar.set_description(f"Processing: {os.path.basename(file_path)} ({db_type})")

            file_start_global = global_row_offset
            local_row_in_file = 0
            rows_processed_in_file = 0
            rows_cleaned_in_file = 0

            # Process file and get generator
            chunk_generator = process_file_glove_only(file_path, glove_columns)
            
            # Process each chunk as it's yielded
            for glove_data in chunk_generator:
                # Process each row in the chunk
                for glove_row in glove_data:
                    if writer is None or examples_in_current_file >= EXAMPLES_PER_TFRECORD:
                        if writer:
                            writer.close()
                            writer = None
                        current_tfrecord_path = os.path.join(TFRecord_OUTPUT_DIR, f"data_{tfrecord_counter:04d}.tfrecord")
                        writer = tf.io.TFRecordWriter(current_tfrecord_path)
                        tfrecord_counter += 1
                        examples_in_current_file = 0

                    example = create_tf_example(glove_row)
                    writer.write(example.SerializeToString())
                    examples_in_current_file += 1

                    # Log mapping
                    tfrecord_mapping_log.append({
                        'source_file': file_path,
                        'source_row_local': local_row_in_file,
                        'source_row_global': global_row_offset,
                        'tfrecord_file': os.path.basename(current_tfrecord_path),
                        'tfrecord_example_index': examples_in_current_file - 1,
                        'tfrecord_file_index': tfrecord_counter - 1
                    })

                    local_row_in_file += 1
                    global_row_offset += 1
                    rows_processed_in_file += 1
                    
                # Explicitly free memory after processing each chunk
                del glove_data
                gc.collect()

            # After processing all chunks, update the processing log
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': rows_processed_in_file,
                'rows_cleaned': rows_cleaned_in_file,
                'status': 'SUCCESS',
                'has_nulls': rows_cleaned_in_file > 0,
                'global_row_start': file_start_global,
                'global_row_end': global_row_offset - 1,
                'timestamp': datetime.now().isoformat()
            })
            
            # Mark file as processed
            processed_files.add(file_path)
            
            # Update checkpoint
            checkpoint.update({
                "processed_files": list(processed_files),
                "current_tfrecord_counter": tfrecord_counter,
                "global_row_offset": global_row_offset,
                "last_processed_file": file_path,
                "last_processed_time": datetime.now().isoformat()
            })
            save_checkpoint(checkpoint)
            
            progress_bar.set_description(f"✓ Processed: {os.path.basename(file_path)}")

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logging.error(error_msg)
            files_with_errors.append(file_path)
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'rows_cleaned': 0,
                'status': f'ERROR: {str(e)}',
                'has_nulls': False,
                'timestamp': datetime.now().isoformat()
            })
            
            # Ensure writer is closed if an error occurs
            if writer:
                writer.close()
                writer = None

        # Force garbage collection after processing each file
        gc.collect()
        
        # Save logs periodically
        if idx % 5 == 0:  # Save every 5 files
            log_df = pd.DataFrame(processing_log)
            log_df.to_csv(LOG_OUTPUT, index=False)

    # Close the final writer if it exists
    if writer:
        writer.close()

    # Save all logs
    log_df = pd.DataFrame(processing_log)
    log_df.to_csv(LOG_OUTPUT, index=False)

    mapping_df = pd.DataFrame(tfrecord_mapping_log)
    mapping_log_path = os.path.join(TFRecord_OUTPUT_DIR, "tfrecord_mapping.csv")
    mapping_df.to_csv(mapping_log_path, index=False)
    print(f"TFRecord ↔ Source file mapping saved to: {mapping_log_path}")

    # Remove checkpoint file after successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    total_processed = sum(log['rows_processed'] for log in processing_log if log['status'] == 'SUCCESS')
    total_cleaned = sum(log['rows_cleaned'] for log in processing_log if log['status'] == 'SUCCESS')
    print(f"Total rows processed: {total_processed}")
    print(f"Total rows cleaned: {total_cleaned}")
    print(f"TFRecord files created: {tfrecord_counter}")

    if files_with_errors:
        print(f"⚠️  {len(files_with_errors)} files had errors. See {ERROR_LOG}")
        print("Problem files:")
        for file in files_with_errors:
            print(f"  - {file}")
    else:
        print("✅ All files processed successfully.")

    metadata = {
        'total_rows': total_processed,
        'rows_cleaned': total_cleaned,
        'glove_columns_DB7': GLOVE_COLUMNS_DB7,
        'glove_columns_DB23': GLOVE_COLUMNS_DB23,
        'tfrecord_files': tfrecord_counter,
        'examples_per_tfrecord': EXAMPLES_PER_TFRECORD,
        'output_feature_names': [f'glove_{i}' for i in range(len(GLOVE_COLUMNS_DB7))],
        'processing_date': datetime.now().isoformat()
    }

    with open(os.path.join(TFRecord_OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"TFRecord files saved to: {TFRecord_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

---

## File: DB\creation\creation_final2.py
**Path:** `C:\stage\stage\DB\creation\creation_final2.py`

```python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import gc
import logging
import json
from datetime import datetime
import csv

# Configuration
FILE_LIST_CSV = r"train_map.csv"
TFRecord_OUTPUT_DIR = r"F:\DB\tfrecords"
LOG_OUTPUT = r"F:\DB\processing_log.csv"
ERROR_LOG = r"F:\DB\processing_errors.log"
CHECKPOINT_FILE = r"F:\DB\processing_checkpoint.json"
MAPPING_LOG = r"F:\DB\tfrecord_mapping.csv"  # Separate mapping log file
CHUNK_SIZE = 5000
EXAMPLES_PER_TFRECORD = 50000

# Define the correct base path for your files
CORRECT_BASE_PATH = r"E:\DB"

# Define glove column mappings for different databases
GLOVE_COLUMNS_DB7 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_7", "glove_8", "glove_9", "glove_10",
    "glove_11", "glove_12", "glove_13", "glove_14", "glove_15",
    "glove_16", "glove_17", "glove_18"
]

GLOVE_COLUMNS_DB23 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_8", "glove_9", "glove_11", "glove_12",
    "glove_13", "glove_15", "glove_16", "glove_17", "glove_19",
    "glove_20", "glove_21", "glove_22"
]

# Set up logging
logging.basicConfig(
    filename=ERROR_LOG, 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def correct_file_path(original_path):
    if "DB2" in original_path or "DB3" in original_path or "DB7" in original_path:
        parts = original_path.split("DB\\", 1)
        if len(parts) > 1:
            relative_path = parts[1]
            return os.path.join(CORRECT_BASE_PATH, relative_path)
    return original_path

def get_db_type(file_path):
    if "DB7" in file_path:
        return "DB7"
    elif "DB2" in file_path or "DB3" in file_path:
        return "DB23"
    else:
        raise ValueError(f"Unknown DB type in path: {file_path}")

def get_glove_columns(db_type):
    if db_type == "DB7":
        return GLOVE_COLUMNS_DB7
    elif db_type == "DB23":
        return GLOVE_COLUMNS_DB23
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

def clean_dataframe(df, glove_columns):
    """Clean DataFrame by replacing infs and NaNs with zeros"""
    df_clean = df[glove_columns].copy()
    
    # Replace infinite values with zero
    df_clean = df_clean.replace([np.inf, -np.inf], 0)
    
    # Fill NaN values with zero
    df_clean = df_clean.fillna(0)
    
    return df_clean.values.astype(np.float32)

def create_tf_example(glove_row):
    feature_dict = {
        f'glove_{i}': tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(x)]))
        for i, x in enumerate(glove_row)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def process_file_glove_only(file_path, glove_columns, chunk_size=CHUNK_SIZE):
    try:
        # Get total rows more efficiently
        with open(file_path, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1
        
        rows_processed = 0
        rows_cleaned = 0
        
        # Process file in chunks and yield cleaned data incrementally
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False),
                         total=int(total_rows/chunk_size)+1,
                         desc=f"Processing {os.path.basename(file_path)}"):
            
            # Check for required columns
            missing_cols = [col for col in glove_columns if col not in chunk.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
            
            # Clean the data (handle infs and NaNs)
            glove_data = clean_dataframe(chunk, glove_columns)
            
            # Count how many rows had issues
            original_count = len(chunk)
            cleaned_count = len(glove_data)
            if cleaned_count < original_count:
                rows_cleaned += (original_count - cleaned_count)
            
            rows_processed += cleaned_count
            
            # Yield the cleaned data instead of accumulating it
            yield glove_data
            
        # Return total counts after processing all chunks
        return rows_processed, rows_cleaned
        
    except Exception as e:
        raise Exception(f"Error processing {file_path}: {str(e)}")

def load_checkpoint():
    """Load processing checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        "processed_files": [],
        "current_tfrecord_counter": 0,
        "global_row_offset": 0,
        "start_time": datetime.now().isoformat()
    }

def save_checkpoint(checkpoint_data):
    """Save processing checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def initialize_mapping_log():
    """Initialize the mapping log file with headers if it doesn't exist"""
    if not os.path.exists(MAPPING_LOG):
        with open(MAPPING_LOG, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'source_file', 'source_row_local', 'source_row_global',
                'tfrecord_file', 'tfrecord_example_index', 'tfrecord_file_index'
            ])

def append_to_mapping_log(mapping_entry):
    """Append a single mapping entry to the CSV file"""
    with open(MAPPING_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            mapping_entry['source_file'],
            mapping_entry['source_row_local'],
            mapping_entry['source_row_global'],
            mapping_entry['tfrecord_file'],
            mapping_entry['tfrecord_example_index'],
            mapping_entry['tfrecord_file_index']
        ])

def main():
    print("Loading file list...")
    df_log = pd.read_csv(FILE_LIST_CSV)
    df_log['corrected_filename'] = df_log['filename'].apply(correct_file_path)
    os.makedirs(TFRecord_OUTPUT_DIR, exist_ok=True)

    # Initialize mapping log
    initialize_mapping_log()

    # Load checkpoint and existing logs
    checkpoint = load_checkpoint()
    processed_files = set(checkpoint.get("processed_files", []))
    
    # Load existing log if it exists
    processing_log = []
    if os.path.exists(LOG_OUTPUT):
        try:
            existing_log = pd.read_csv(LOG_OUTPUT)
            processing_log = existing_log.to_dict('records')
            # Add files from log to processed_files set
            for log_entry in processing_log:
                if log_entry['status'] == 'SUCCESS':
                    processed_files.add(log_entry['filename'])
        except:
            print("Warning: Could not read existing log file. Starting fresh.")
    
    files_with_errors = []

    tfrecord_counter = checkpoint.get("current_tfrecord_counter", 0)
    global_row_offset = checkpoint.get("global_row_offset", 0)
    examples_in_current_file = 0
    writer = None
    current_tfrecord_path = ""

    # Get the next available TFRecord file number if not starting from checkpoint
    if tfrecord_counter == 0 and os.path.exists(TFRecord_OUTPUT_DIR):
        existing_files = [f for f in os.listdir(TFRecord_OUTPUT_DIR) if f.startswith('data_') and f.endswith('.tfrecord')]
        if existing_files:
            tfrecord_counter = max([int(f.split('_')[1].split('.')[0]) for f in existing_files]) + 1

    # Create a progress bar for files
    progress_bar = tqdm(df_log.iterrows(), total=len(df_log), desc="Processing Files")
    
    for idx, row in progress_bar:
        original_path = row['filename']
        file_path = row['corrected_filename']
        expected_count = row.get('rows_extracted', -1)

        # Skip already processed files
        if file_path in processed_files:
            progress_bar.set_description(f"Skipping: {os.path.basename(file_path)}")
            continue

        progress_bar.set_description(f"Processing: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logging.error(error_msg)
            files_with_errors.append(file_path)
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'rows_cleaned': 0,
                'status': 'FILE_NOT_FOUND',
                'has_nulls': False,
                'timestamp': datetime.now().isoformat()
            })
            continue

        try:
            db_type = get_db_type(file_path)
            glove_columns = get_glove_columns(db_type)
            progress_bar.set_description(f"Processing: {os.path.basename(file_path)} ({db_type})")

            file_start_global = global_row_offset
            local_row_in_file = 0
            rows_processed_in_file = 0
            rows_cleaned_in_file = 0

            # Process file and get generator
            chunk_generator = process_file_glove_only(file_path, glove_columns)
            
            # Process each chunk as it's yielded
            for glove_data in chunk_generator:
                # Process each row in the chunk
                for glove_row in glove_data:
                    if writer is None or examples_in_current_file >= EXAMPLES_PER_TFRECORD:
                        if writer:
                            writer.close()
                            writer = None
                        current_tfrecord_path = os.path.join(TFRecord_OUTPUT_DIR, f"data_{tfrecord_counter:04d}.tfrecord")
                        writer = tf.io.TFRecordWriter(current_tfrecord_path)
                        tfrecord_counter += 1
                        examples_in_current_file = 0

                    example = create_tf_example(glove_row)
                    writer.write(example.SerializeToString())
                    examples_in_current_file += 1

                    # Write mapping entry directly to file instead of storing in memory
                    mapping_entry = {
                        'source_file': file_path,
                        'source_row_local': local_row_in_file,
                        'source_row_global': global_row_offset,
                        'tfrecord_file': os.path.basename(current_tfrecord_path),
                        'tfrecord_example_index': examples_in_current_file - 1,
                        'tfrecord_file_index': tfrecord_counter - 1
                    }
                    append_to_mapping_log(mapping_entry)

                    local_row_in_file += 1
                    global_row_offset += 1
                    rows_processed_in_file += 1
                    
                # Explicitly free memory after processing each chunk
                del glove_data
                gc.collect()

            # After processing all chunks, update the processing log
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': rows_processed_in_file,
                'rows_cleaned': rows_cleaned_in_file,
                'status': 'SUCCESS',
                'has_nulls': rows_cleaned_in_file > 0,
                'global_row_start': file_start_global,
                'global_row_end': global_row_offset - 1,
                'timestamp': datetime.now().isoformat()
            })
            
            # Mark file as processed
            processed_files.add(file_path)
            
            # Update checkpoint
            checkpoint.update({
                "processed_files": list(processed_files),
                "current_tfrecord_counter": tfrecord_counter,
                "global_row_offset": global_row_offset,
                "last_processed_file": file_path,
                "last_processed_time": datetime.now().isoformat()
            })
            save_checkpoint(checkpoint)
            
            progress_bar.set_description(f"✓ Processed: {os.path.basename(file_path)}")

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logging.error(error_msg)
            files_with_errors.append(file_path)
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'rows_cleaned': 0,
                'status': f'ERROR: {str(e)}',
                'has_nulls': False,
                'timestamp': datetime.now().isoformat()
            })
            
            # Ensure writer is closed if an error occurs
            if writer:
                writer.close()
                writer = None

        # Force garbage collection after processing each file
        gc.collect()
        
        # Save logs periodically
        if idx % 5 == 0:  # Save every 5 files
            log_df = pd.DataFrame(processing_log)
            log_df.to_csv(LOG_OUTPUT, index=False)

    # Close the final writer if it exists
    if writer:
        writer.close()

    # Save final processing log
    log_df = pd.DataFrame(processing_log)
    log_df.to_csv(LOG_OUTPUT, index=False)

    # Remove checkpoint file after successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    total_processed = sum(log['rows_processed'] for log in processing_log if log['status'] == 'SUCCESS')
    total_cleaned = sum(log['rows_cleaned'] for log in processing_log if log['status'] == 'SUCCESS')
    print(f"Total rows processed: {total_processed}")
    print(f"Total rows cleaned: {total_cleaned}")
    print(f"TFRecord files created: {tfrecord_counter}")

    if files_with_errors:
        print(f"⚠️  {len(files_with_errors)} files had errors. See {ERROR_LOG}")
        print("Problem files:")
        for file in files_with_errors:
            print(f"  - {file}")
    else:
        print("✅ All files processed successfully.")

    metadata = {
        'total_rows': total_processed,
        'rows_cleaned': total_cleaned,
        'glove_columns_DB7': GLOVE_COLUMNS_DB7,
        'glove_columns_DB23': GLOVE_COLUMNS_DB23,
        'tfrecord_files': tfrecord_counter,
        'examples_per_tfrecord': EXAMPLES_PER_TFRECORD,
        'output_feature_names': [f'glove_{i}' for i in range(len(GLOVE_COLUMNS_DB7))],
        'processing_date': datetime.now().isoformat()
    }

    with open(os.path.join(TFRecord_OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"TFRecord files saved to: {TFRecord_OUTPUT_DIR}")
    print(f"Mapping log saved to: {MAPPING_LOG}")

if __name__ == "__main__":
    main()
```

---

## File: DB\creation\db.py
**Path:** `C:\stage\stage\DB\creation\db.py`

```python
import os
import pandas as pd
import h5py
import numpy as np
from collections import defaultdict
import logging
import sys
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("csv_to_hdf5.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CSVtoHDF5")

def validate_columns(header, expected_columns):
    """
    Validates if all expected columns exist in the header (case-insensitive, space-stripped)
    Returns:
        - valid: bool (True if all columns exist)
        - missing: list of missing columns (original expected format)
        - col_mapping: dict mapping expected column names to actual header names
    """
    header_clean = [h.strip().lower() for h in header]
    expected_clean = [e.strip().lower() for e in expected_columns]
    
    # Create mapping from clean name to original header
    clean_to_orig = {}
    for orig, clean in zip(header, header_clean):
        if clean not in clean_to_orig:
            clean_to_orig[clean] = orig
    
    # Check missing columns
    missing = []
    col_mapping = {}
    for exp, exp_clean in zip(expected_columns, expected_clean):
        if exp_clean in clean_to_orig:
            col_mapping[exp] = clean_to_orig[exp_clean]
        else:
            missing.append(exp)
    
    return not missing, missing, col_mapping

def process_csv_files(root_folders, hdf5_path, log_path, expected_columns, chunk_size=10000):
    """
    Processes CSV files from multiple root folders into a single HDF5 file
    
    Args:
        root_folders: List of root folders to search for CSV files
        hdf5_path: Output HDF5 file path
        log_path: Output log file path
        expected_columns: List of columns to extract (in desired order)
        chunk_size: CSV reading chunk size
    """
    # Collect all CSV files
    csv_files = []
    for root in root_folders:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith('.csv'):
                    csv_files.append(os.path.join(dirpath, f))
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    if not csv_files:
        logger.error("No CSV files found in specified folders")
        return

    # Initialize HDF5 file
    with h5py.File(hdf5_path, 'w') as h5file:
        # Create dataset with unlimited first dimension
        dtype = h5py.string_dtype()  # Using string dtype for mixed data types
        max_shape = (None, len(expected_columns))
        dataset = h5file.create_dataset(
            'data', 
            shape=(0, len(expected_columns)), 
            maxshape=max_shape,
            dtype=dtype,
            chunks=(chunk_size, len(expected_columns))
        )
        
        # Create attributes for column names
        h5file.attrs['columns'] = expected_columns

        # Process files and log extraction details
        extraction_log = []
        global_row_counter = 0
        
        for file_path in tqdm(csv_files, desc="Processing CSV files"):
            try:
                # Read header only
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    header_line = f.readline().strip()
                    if not header_line:
                        logger.warning(f"Empty file or missing header: {file_path}")
                        continue
                    header = [col.strip() for col in header_line.split(',')]
                
                # Validate columns
                valid, missing, col_mapping = validate_columns(header, expected_columns)
                if not valid:
                    logger.warning(f"Skipping {file_path} - Missing columns: {missing}")
                    continue
                
                # Prepare column indices for efficient reading
                col_indices = [header.index(col_mapping[col]) for col in expected_columns]
                
                # Count total rows in file (excluding header)
                total_rows = 0
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    total_rows = sum(1 for _ in f) - 1  # Subtract header
                
                if total_rows <= 0:
                    logger.info(f"Skipping {file_path} - No data rows")
                    continue
                
                # Process in chunks
                start_row = global_row_counter + 1
                rows_processed = 0
                
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    next(f)  # Skip header
                    while True:
                        chunk_lines = []
                        for _ in range(chunk_size):
                            line = f.readline()
                            if not line:
                                break
                            chunk_lines.append(line.strip())
                        
                        if not chunk_lines:
                            break
                        
                        # Parse chunk
                        chunk_data = []
                        for i, line in enumerate(chunk_lines):
                            try:
                                # Handle CSV formatting (commas in quotes)
                                parts = []
                                in_quote = False
                                current = []
                                for char in line:
                                    if char == '"':
                                        in_quote = not in_quote
                                    elif char == ',' and not in_quote:
                                        parts.append(''.join(current))
                                        current = []
                                    else:
                                        current.append(char)
                                parts.append(''.join(current))
                                
                                # Extract required columns
                                row = [parts[idx].strip('"') for idx in col_indices]
                                chunk_data.append(row)
                            except Exception as e:
                                logger.warning(f"Error parsing line {start_row + rows_processed + i} in {file_path}: {str(e)}")
                        
                        # Write to HDF5
                        chunk_array = np.array(chunk_data, dtype=dtype)
                        current_size = dataset.shape[0]
                        new_size = current_size + len(chunk_data)
                        dataset.resize(new_size, axis=0)
                        dataset[current_size:new_size] = chunk_array
                        
                        rows_processed += len(chunk_data)
                
                # Update global counter and log
                end_row = start_row + rows_processed - 1
                extraction_log.append({
                    'filename': file_path,
                    'start_line': start_row,
                    'end_line': end_row,
                    'rows_extracted': rows_processed
                })
                global_row_counter = end_row
                
                logger.debug(f"Processed {file_path}: {rows_processed} rows ({start_row}-{end_row})")
            
            except Exception as e:
                logger.error(f"Critical error processing {file_path}: {str(e)}", exc_info=True)
        
        # Write extraction log
        with open(log_path, 'w') as log_file:
            log_file.write("filename,start_line,end_line,rows_extracted\n")
            for entry in extraction_log:
                log_file.write(f"{entry['filename']},{entry['start_line']},{entry['end_line']},{entry['rows_extracted']}\n")
        
        logger.info(f"HDF5 file created: {hdf5_path}")
        logger.info(f"Extraction log created: {log_path}")
        logger.info(f"Total rows processed: {global_row_counter}")

if __name__ == "__main__":
    # Configuration - MODIFY THESE VALUES
    ROOT_FOLDERS = [
        "E:\DB\DB2\E2",
        "E:\DB\DB2\E1",
        "E:\DB\DB3\E1",
        "E:\DB\DB3\E2",
        "E:\DB\DB7\E1",
        "E:\DB\DB7\E2"

    ]
    HDF5_OUTPUT = "F:\DB\combined_data.h5"
    LOG_OUTPUT = "F:\DB\extraction_log.csv"
    EXPECTED_COLUMNS = [
        "emg_1","emg_2","emg_3","emg_4",
        "emg_5","emg_6","emg_7","emg_8",
        "emg_9","emg_10","emg_11","emg_12"
    ]
    CHUNK_SIZE = 50000  # Adjust based on memory constraints

    # Validate configuration
    if not all(os.path.isdir(folder) for folder in ROOT_FOLDERS):
        logger.error("One or more root folders are invalid")
        sys.exit(1)
    
    if not EXPECTED_COLUMNS:
        logger.error("Expected columns list cannot be empty")
        sys.exit(1)

    # Execute conversion
    logger.info("Starting CSV to HDF5 conversion")
    logger.info(f"Processing folders: {ROOT_FOLDERS}")
    logger.info(f"Expected columns: {EXPECTED_COLUMNS}")
    
    process_csv_files(
        root_folders=ROOT_FOLDERS,
        hdf5_path=HDF5_OUTPUT,
        log_path=LOG_OUTPUT,
        expected_columns=EXPECTED_COLUMNS,
        chunk_size=CHUNK_SIZE
    )
    
    logger.info("Conversion completed successfully")
```

---

## File: DB\creation\labels.py
**Path:** `C:\stage\stage\DB\creation\labels.py`

```python
import os
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

# Configuration
FILE_LIST_CSV = r"D:\DB\extraction_log.csv"     # Path to your extraction_log.csv
HDF5_OUTPUT = r"F:\DB\labels.h5"                # Output HDF5 file
LOG_OUTPUT = r"D:\DB\label_extraction_log.csv"  # Output log (recreated)
CHUNK_SIZE = 10000                              # For streaming chunks to HDF5
DATA_DTYPE = 'float32'


def get_db_type(file_path):
    """Determine DB type from path."""
    if "DB7" in file_path:
        return "DB7"
    elif "DB2" in file_path or "DB3" in file_path:
        return "DB23"
    else:
        raise ValueError(f"Unknown DB type in path: {file_path}")


def get_column_mapping(db_type):
    """Return glove column names needed, in correct physical sensor order."""
    if db_type == "DB7":
        return [
            "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
            "glove_6", "glove_7", "glove_8", "glove_9", "glove_10",
            "glove_11", "glove_12", "glove_13", "glove_14", "glove_15",
            "glove_16", "glove_17", "glove_18"
        ]
    elif db_type == "DB23":
        return [
            "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
            "glove_6", "glove_8", "glove_9", "glove_11", "glove_12",
            "glove_13", "glove_15", "glove_16", "glove_17", "glove_19",
            "glove_20", "glove_21", "glove_22"
        ]
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")


def read_and_map_columns(file_path, target_cols, nrows, chunk_size=CHUNK_SIZE):
    """
    Generator that yields chunks of data mapped to target columns.
    Fills missing values with NaN.
    """
    df_iter = pd.read_csv(file_path, chunksize=chunk_size, dtype=np.float32, low_memory=False)

    cols_seen = None
    for df_chunk in df_iter:
        if cols_seen is None:
            cols_seen = set(df_chunk.columns.str.strip())
            print(f"Columns in {os.path.basename(file_path)}: {sorted(cols_seen)}")

        # Select only the required columns, fill missing with NaN
        data_chunk = []
        for col in target_cols:
            if col in df_chunk.columns:
                data_chunk.append(df_chunk[col].values)
            else:
                data_chunk.append(np.full(len(df_chunk), np.nan))

        # Stack into shape (n_rows, n_cols)
        yield np.column_stack(data_chunk)


def main():
    print("Loading extraction order...")
    df_log = pd.read_csv(FILE_LIST_CSV)
    expected_columns = [f"label_{i+1}" for i in range(18)]  # Final output labels

    total_expected_rows = df_log['rows_extracted'].sum()
    print(f"Total rows expected: {total_expected_rows}")

    # Initialize HDF5
    print("Initializing HDF5 dataset...")
    with h5py.File(HDF5_OUTPUT, 'w') as h5f:
        maxshape = (None, 18)
        dset = h5f.create_dataset(
            'labels',
            shape=(0, 18),
            maxshape=maxshape,
            dtype=DATA_DTYPE,
            chunks=(CHUNK_SIZE, 18),
            compression='gzip',
            compression_opts=4,
            fillvalue=np.nan
        )
        dset.attrs['columns'] = expected_columns
        dset.attrs['source_files'] = len(df_log)

    # Process each file in order
    current_row = 0
    extraction_log = []

    for idx, row in tqdm(df_log.iterrows(), total=len(df_log), desc="Processing Files"):
        file_path = row['filename']
        expected_count = row['rows_extracted']

        print(f"Processing: {os.path.basename(file_path)} ({expected_count} rows)")

        # Determine which glove columns to extract
        db_type = get_db_type(file_path)
        required_glove_cols = get_column_mapping(db_type)

        # Stream and write chunks
        file_rows_written = 0
        with h5py.File(HDF5_OUTPUT, 'a') as h5f:
            dset = h5f['labels']

            for data_chunk in read_and_map_columns(file_path, required_glove_cols, expected_count, CHUNK_SIZE):
                chunk_len = len(data_chunk)
                if file_rows_written + chunk_len > expected_count:
                    # Truncate last chunk if more than expected
                    data_chunk = data_chunk[:expected_count - file_rows_written]
                    chunk_len = len(data_chunk)

                # Resize and write
                current_size = dset.shape[0]
                dset.resize(current_size + chunk_len, axis=0)
                dset[current_size:current_size + chunk_len] = data_chunk.astype(DATA_DTYPE)

                file_rows_written += chunk_len

        # Log this file
        start_line = current_row + 1
        end_line = current_row + file_rows_written
        extraction_log.append({
            'filename': file_path,
            'start_line': start_line,
            'end_line': end_line,
            'rows_extracted': file_rows_written
        })
        current_row = end_line

        print(f"✓ Extracted {file_rows_written}/{expected_count} rows")

    # Final validation
    if current_row != total_expected_rows:
        print(f"⚠️  Row count mismatch: expected {total_expected_rows}, got {current_row}")
    else:
        print(f"✅ Successfully extracted all {current_row} rows.")

    # Save new log
    log_df = pd.DataFrame(extraction_log)
    log_df.to_csv(LOG_OUTPUT, index=False)
    print(f"Extraction log saved to: {LOG_OUTPUT}")

    # Update HDF5 final metadata
    with h5py.File(HDF5_OUTPUT, 'a') as h5f:
        h5f.attrs['total_rows'] = current_row
        h5f.attrs['status'] = 'COMPLETE'

    print(f"HDF5 saved to: {HDF5_OUTPUT}")


if __name__ == "__main__":
    main()
```

---

## File: DB\creation\labels1.py
**Path:** `C:\stage\stage\DB\creation\labels1.py`

```python
import os
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

# Configuration - UPDATE THESE PATHS TO MATCH YOUR CURRENT SETUP
FILE_LIST_CSV = r"D:\DB\extraction_log.csv"     # Path to your extraction_log.csv
HDF5_OUTPUT = r"D:\data\y.h5"                  # Output HDF5 file
LOG_OUTPUT = r"D:\DB\labels logs\label_extraction_log_updated.csv"  # Updated output log
ERROR_LOG = r"D:\DB\labels logs\extraction_errors.log"      # Log for errors
NULL_LOG = r"D:\DB\labels logs\null_values.log"             # Log for files with null values
CORRUPT_LOG = r"D:\DB\labels logs\corrupt_files.log"        # Log for corrupt files
CHUNK_SIZE = 10000                              # For streaming chunks to HDF5
DATA_DTYPE = 'float32'

# Define the correct base path for your files - UPDATE THIS TO MATCH YOUR CURRENT SETUP
CORRECT_BASE_PATH = r"E:\DB"

# Define default values for each glove sensor based on typical rest positions
DEFAULT_GLOVE_VALUES = {
    "glove_1": 0.0, "glove_2": 0.0, "glove_3": 0.0, "glove_4": 0.0, "glove_5": 0.0,
    "glove_6": 0.0, "glove_7": 0.0, "glove_8": 0.0, "glove_9": 0.0, "glove_10": 0.0,
    "glove_11": 0.0, "glove_12": 0.0, "glove_13": 0.0, "glove_14": 0.0, "glove_15": 0.0,
    "glove_16": 0.0, "glove_17": 0.0, "glove_18": 0.0, "glove_19": 0.0, "glove_20": 0.0,
    "glove_21": 0.0, "glove_22": 0.0
}

def correct_file_path(original_path):
    """Correct the file path by updating the base path"""
    # Extract the relative path from the original path
    if "DB2" in original_path or "DB3" in original_path or "DB7" in original_path:
        # Get the part after "DB\"
        parts = original_path.split("DB\\", 1)
        if len(parts) > 1:
            relative_path = parts[1]
            return os.path.join(CORRECT_BASE_PATH, relative_path)
    
    # If we can't parse the path, return the original
    return original_path

def get_db_type(file_path):
    """Determine DB type from path."""
    if "DB7" in file_path:
        return "DB7"
    elif "DB2" in file_path or "DB3" in file_path:
        return "DB23"
    else:
        raise ValueError(f"Unknown DB type in path: {file_path}")

def get_column_mapping(db_type):
    """Return glove column names needed, in correct physical sensor order."""
    if db_type == "DB7":
        return [
            "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
            "glove_6", "glove_7", "glove_8", "glove_9", "glove_10",
            "glove_11", "glove_12", "glove_13", "glove_14", "glove_15",
            "glove_16", "glove_17", "glove_18"
        ]
    elif db_type == "DB23":
        return [
            "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
            "glove_6", "glove_8", "glove_9", "glove_11", "glove_12",
            "glove_13", "glove_15", "glove_16", "glove_17", "glove_19",
            "glove_20", "glove_21", "glove_22"
        ]
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

def read_and_map_columns(file_path, target_cols, nrows, chunk_size=CHUNK_SIZE):
    """
    Generator that yields chunks of data mapped to target columns.
    Uses default values instead of NaN for missing sensors.
    Returns the data chunk and a tuple indicating if nulls were found and if the chunk is entirely null.
    """
    try:
        df_iter = pd.read_csv(file_path, chunksize=chunk_size, dtype=np.float32, low_memory=False)
    except Exception as e:
        raise Exception(f"Error reading {file_path}: {str(e)}")

    cols_seen = None
    for df_chunk in df_iter:
        if cols_seen is None:
            cols_seen = set(df_chunk.columns.str.strip())
            print(f"Columns in {os.path.basename(file_path)}: {sorted(cols_seen)}")

        # Select only the required columns, use default values for missing sensors
        data_chunk = []
        has_nulls = False
        is_entirely_null = True
        
        for col in target_cols:
            if col in df_chunk.columns:
                # Check for null values
                null_count = df_chunk[col].isnull().sum()
                if null_count > 0:
                    has_nulls = True
                
                # Check if all values in this column are null
                if null_count < len(df_chunk):
                    is_entirely_null = False
                
                # Replace any NaN values in the column with the default value
                col_data = df_chunk[col].fillna(DEFAULT_GLOVE_VALUES[col]).values
                data_chunk.append(col_data)
            else:
                # Column is missing entirely, which means all values are effectively null
                has_nulls = True
                data_chunk.append(np.full(len(df_chunk), DEFAULT_GLOVE_VALUES[col]))

        # Stack into shape (n_rows, n_cols)
        yield np.column_stack(data_chunk), has_nulls, is_entirely_null

def main():
    print("Loading extraction order...")
    df_log = pd.read_csv(FILE_LIST_CSV)
    
    # Correct file paths
    df_log['corrected_filename'] = df_log['filename'].apply(correct_file_path)
    
    # Create error logs
    error_log = open(ERROR_LOG, 'w')
    error_log.write("File extraction errors:\n")
    
    null_log = open(NULL_LOG, 'w')
    null_log.write("Files with null values:\n")
    
    corrupt_log = open(CORRUPT_LOG, 'w')
    corrupt_log.write("Corrupt files (entirely null in required columns):\n")
    
    expected_columns = [f"label_{i+1}" for i in range(18)]  # Final output labels
    total_expected_rows = df_log['rows_extracted'].sum()
    print(f"Total rows expected: {total_expected_rows}")

    # Initialize HDF5
    print("Initializing HDF5 dataset...")
    with h5py.File(HDF5_OUTPUT, 'w') as h5f:
        maxshape = (None, 18)
        dset = h5f.create_dataset(
            'labels',
            shape=(0, 18),
            maxshape=maxshape,
            dtype=DATA_DTYPE,
            chunks=(CHUNK_SIZE, 18),
            compression='gzip',
            compression_opts=4
        )
        dset.attrs['columns'] = expected_columns
        dset.attrs['source_files'] = len(df_log)

    # Process each file in order
    current_row = 0
    extraction_log = []
    files_with_errors = []
    files_with_nulls = []
    corrupt_files = []

    for idx, row in tqdm(df_log.iterrows(), total=len(df_log), desc="Processing Files"):
        original_path = row['filename']
        file_path = row['corrected_filename']
        expected_count = row['rows_extracted']

        print(f"Processing: {os.path.basename(file_path)} ({expected_count} rows)")

        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path} (original: {original_path})"
            print(f"ERROR: {error_msg}")
            error_log.write(f"{error_msg}\n")
            files_with_errors.append(file_path)
            continue

        try:
            # Determine which glove columns to extract
            db_type = get_db_type(file_path)
            required_glove_cols = get_column_mapping(db_type)

            # Track nulls and corruption for this file
            file_has_nulls = False
            file_is_corrupt = True  # Assume corrupt until proven otherwise
            
            # Stream and write chunks
            file_rows_written = 0
            with h5py.File(HDF5_OUTPUT, 'a') as h5f:
                dset = h5f['labels']

                for data_chunk, has_nulls, is_entirely_null in read_and_map_columns(file_path, required_glove_cols, expected_count, CHUNK_SIZE):
                    # Update file-level null and corruption status
                    if has_nulls:
                        file_has_nulls = True
                    
                    if not is_entirely_null:
                        file_is_corrupt = False
                    
                    chunk_len = len(data_chunk)
                    if file_rows_written + chunk_len > expected_count:
                        # Truncate last chunk if more than expected
                        data_chunk = data_chunk[:expected_count - file_rows_written]
                        chunk_len = len(data_chunk)

                    # Resize and write
                    current_size = dset.shape[0]
                    dset.resize(current_size + chunk_len, axis=0)
                    dset[current_size:current_size + chunk_len] = data_chunk.astype(DATA_DTYPE)

                    file_rows_written += chunk_len

            # Log nulls if found
            if file_has_nulls:
                null_msg = f"File contains null values: {file_path}"
                print(f"NOTE: {null_msg}")
                null_log.write(f"{file_path}\n")
                files_with_nulls.append(file_path)
            
            # Log corruption if entire file is null in required columns
            if file_is_corrupt:
                corrupt_msg = f"File is entirely null in required columns (corrupt): {file_path}"
                print(f"ERROR: {corrupt_msg}")
                corrupt_log.write(f"{file_path}\n")
                corrupt_files.append(file_path)
                # Mark as error
                error_log.write(f"{corrupt_msg}\n")
                files_with_errors.append(file_path)

            # Log this file
            start_line = current_row + 1
            end_line = current_row + file_rows_written
            extraction_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'start_line': start_line,
                'end_line': end_line,
                'rows_extracted': file_rows_written,
                'status': 'SUCCESS' if not file_is_corrupt else 'CORRUPT'
            })
            current_row = end_line

            print(f"✓ Extracted {file_rows_written}/{expected_count} rows")

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            print(f"ERROR: {error_msg}")
            error_log.write(f"{error_msg}\n")
            files_with_errors.append(file_path)
            
            extraction_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'start_line': current_row + 1,
                'end_line': current_row,
                'rows_extracted': 0,
                'status': f'ERROR: {str(e)}'
            })

    # Close all logs
    error_log.close()
    null_log.close()
    corrupt_log.close()

    # Final validation
    if current_row != total_expected_rows:
        print(f"⚠️  Row count mismatch: expected {total_expected_rows}, got {current_row}")
    else:
        print(f"✅ Successfully extracted all {current_row} rows.")

    # Save new log
    log_df = pd.DataFrame(extraction_log)
    log_df.to_csv(LOG_OUTPUT, index=False)
    print(f"Extraction log saved to: {LOG_OUTPUT}")

    # Report on files with errors
    if files_with_errors:
        print(f"⚠️  {len(files_with_errors)} files had errors during processing:")
        for file in files_with_errors:
            print(f"  - {file}")
        print(f"See {ERROR_LOG} for details.")
    else:
        print("✅ All files processed successfully.")
    
    # Report on files with nulls
    if files_with_nulls:
        print(f"ℹ️  {len(files_with_nulls)} files contained null values:")
        for file in files_with_nulls:
            print(f"  - {file}")
        print(f"See {NULL_LOG} for details.")
    
    # Report on corrupt files
    if corrupt_files:
        print(f"🚫 {len(corrupt_files)} files were entirely null in required columns (corrupt):")
        for file in corrupt_files:
            print(f"  - {file}")
        print(f"See {CORRUPT_LOG} for details.")

    # Update HDF5 final metadata
    with h5py.File(HDF5_OUTPUT, 'a') as h5f:
        h5f.attrs['total_rows'] = current_row
        h5f.attrs['status'] = 'COMPLETE' if not files_with_errors else 'PARTIAL'
        h5f.attrs['files_with_nulls'] = len(files_with_nulls)
        h5f.attrs['corrupt_files'] = len(corrupt_files)

    print(f"HDF5 saved to: {HDF5_OUTPUT}")

if __name__ == "__main__":
    main()
```

---

## File: DB\creation\report.py
**Path:** `C:\stage\stage\DB\creation\report.py`

```python
import pandas as pd

# Load the mapping file
mapping_file = 'train_map - Copy.csv'
mapping_df = pd.read_csv(mapping_file)

# Create an empty list to store the reports for each file
all_reports = []

# Loop through each row in the mapping file
for index, row in mapping_df.iterrows():
    file_path = row['filename']
    
    try:
        # Load the actual CSV data file
        data_df = pd.read_csv(file_path)
        
        # Calculate the number of nulls in each column
        null_counts = data_df.isnull().sum()
        
        # Create a report DataFrame for this specific file
        report_df = pd.DataFrame({
            'filename': [file_path] * len(null_counts),
            'column_name': null_counts.index,
            'null_count': null_counts.values,
            'total_rows_in_file': [len(data_df)] * len(null_counts)
        })
        
        # Append the report to the list
        all_reports.append(report_df)
        
        print(f"Successfully processed: {file_path}")
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Combine all individual reports into one master report
if all_reports:
    master_report = pd.concat(all_reports, ignore_index=True)
    
    # Optionally, save the master report to a CSV file
    master_report.to_csv('column_null_report.csv', index=False)
    print("\nMaster report saved to 'column_null_report.csv'")
    
    # Display the master report
    print("\n--- MASTER REPORT: NULL COUNTS BY FILE AND COLUMN ---")
    print(master_report.to_string(index=False))
else:
    print("No files were successfully processed.")
```

---

## File: DB\creation\resti.py
**Path:** `C:\stage\stage\DB\creation\resti.py`

```python
import h5py
import pandas as pd
import numpy as np
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('restimulus_creation.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

# Read the extraction log
try:
    log = pd.read_csv('extraction_log.csv')
    logging.info("Extraction log loaded successfully.")
except FileNotFoundError:
    logging.error("extraction_log.csv not found!")
    raise
except Exception as e:
    logging.error(f"Error reading extraction_log.csv: {e}")
    raise

# Validate required columns
required_columns = {'filename', 'start_line', 'end_line'}
if not required_columns.issubset(log.columns):
    missing = required_columns - set(log.columns)
    logging.error(f"Missing columns in extraction_log.csv: {missing}")
    raise ValueError(f"Missing columns: {missing}")

# Create or open HDF5 file
h5_filename = 'restimulus.h5'
total_rows = log['end_line'].iloc[-1]

with h5py.File(h5_filename, 'w') as h5f:
    # Create dataset
    restimulus_dset = h5f.create_dataset('restimulus', (total_rows,), dtype=np.int8)
    logging.info(f"Created HDF5 dataset 'restimulus' with shape ({total_rows},) and dtype {np.int8}.")

    # Process each row in the log
    for idx, row in log.iterrows():
        filename = row['filename']
        start_idx = int(row['start_line']) - 1  # Convert to 0-based index
        end_idx = int(row['end_line']) - 1
        num_lines = end_idx - start_idx + 1

        try:
            # Check if file exists
            if not os.path.exists(filename):
                logging.warning(f"File not found: {filename}. Skipping entry {idx}.")
                continue

            # Read only 'restimulus' column
            df = pd.read_csv(filename, usecols=['restimulus'], nrows=num_lines)
            restimulus_data = df['restimulus'].fillna(0).astype(np.int8).values  # Handle NaNs

            # Check length
            if len(restimulus_data) != num_lines:
                logging.warning(
                    f"Length mismatch in {filename}: expected {num_lines}, got {len(restimulus_data)}. "
                    f"Using first {min(len(restimulus_data), num_lines)} values."
                )
                restimulus_data = restimulus_data[:num_lines]

            # Write to HDF5
            restimulus_dset[start_idx:start_idx + len(restimulus_data)] = restimulus_data
            logging.info(f"Successfully wrote {len(restimulus_data)} entries from {filename} to range [{start_idx}:{start_idx + len(restimulus_data)}]")

        except ValueError as ve:
            # Raised by pandas if 'restimulus' column is missing
            if "did not find column" in str(ve) or "Unknown column" in str(ve):
                logging.warning(f"Column 'restimulus' not found in {filename}. Filling with zeros.")
                zeros = np.zeros(num_lines, dtype=np.int8)
                restimulus_dset[start_idx:end_idx + 1] = zeros
            else:
                logging.error(f"ValueError reading {filename}: {ve}")
                # Optionally fill with zeros or skip
                zeros = np.zeros(num_lines, dtype=np.int8)
                restimulus_dset[start_idx:end_idx + 1] = zeros

        except Exception as e:
            logging.error(f"Unexpected error processing {filename}: {e}")
            # Fill with zeros to preserve alignment
            zeros = np.zeros(num_lines, dtype=np.int8)
            restimulus_dset[start_idx:end_idx + 1] = zeros

logging.info("Restimulus HDF5 file created successfully!")
```

---

## File: DB\manupilate\check.py
**Path:** `C:\stage\stage\DB\manupilate\check.py`

```python
import h5py
import numpy as np

def check_hdf5_structure(file_path):
    """Check the structure of an HDF5 file"""
    print(f"Checking structure of {file_path}:")
    with h5py.File(file_path, 'r') as f:
        print("Keys in file:", list(f.keys()))
        for key in f.keys():
            print(f"Dataset '{key}' shape: {f[key].shape}")

def compute_emg_stats(x_file, restimulus_file):
    """Compute EMG mean and std using only rest periods"""
    # First check the structure of the files
    check_hdf5_structure(x_file)
    check_hdf5_structure(restimulus_file)
    
    with h5py.File(x_file, 'r') as f_x, h5py.File(restimulus_file, 'r') as f_r:
        # Find the correct dataset name for EMG data
        emg_key = None
        for key in f_x.keys():
            if 'emg' in key.lower():
                emg_key = key
                break
        
        if emg_key is None:
            raise ValueError(f"No EMG dataset found in {x_file}. Available datasets: {list(f_x.keys())}")
        
        emg_data = f_x[emg_key]
        restimulus = f_r['restimulus'][:]
        
        # Find rest periods (where restimulus != 0)
        rest_indices = np.where(restimulus != 0)[0]
        
        n_channels = emg_data.shape[1]
        rest_mean = np.zeros(n_channels)
        rest_std = np.zeros(n_channels)
        
        # Read rest data in chunks
        chunk_size = 10000
        for i in range(0, len(rest_indices), chunk_size):
            indices_chunk = rest_indices[i:i+chunk_size]
            emg_chunk = emg_data[indices_chunk, :]
            rest_mean += np.sum(emg_chunk, axis=0)
            rest_std += np.sum(emg_chunk**2, axis=0)
        
        rest_mean /= len(rest_indices)
        rest_std = np.sqrt(rest_std / len(rest_indices) - rest_mean**2)
        
        return rest_mean, rest_std

def compute_glove_stats(y_file):
    """Compute min and range for glove data normalization"""
    # First check the structure of the file
    check_hdf5_structure(y_file)
    
    with h5py.File(y_file, 'r') as f:
        # Find the correct dataset name for glove data
        glove_key = None
        for key in f.keys():
            if 'glove' in key.lower():
                glove_key = key
                break
        
        if glove_key is None:
            raise ValueError(f"No glove dataset found in {y_file}. Available datasets: {list(f.keys())}")
        
        glove_data = f[glove_key]
        n_samples = glove_data.shape[0]
        n_sensors = glove_data.shape[1]
        
        min_vals = np.inf * np.ones(n_sensors)
        max_vals = -np.inf * np.ones(n_sensors)
        
        chunk_size = 10000
        for i in range(0, n_samples, chunk_size):
            chunk = glove_data[i:i+chunk_size, :]
            min_vals = np.minimum(min_vals, np.min(chunk, axis=0))
            max_vals = np.maximum(max_vals, np.max(chunk, axis=0))
            
        return min_vals, max_vals

# Use raw strings for Windows paths
x_file_path = r'F:\DB\x.h5'
restimulus_file_path = r'F:\DB\restimulus.h5'
y_file_path = r'F:\DB\y.h5'

try:
    # Compute and save EMG normalization parameters
    print("Computing EMG normalization parameters...")
    emg_mean, emg_std = compute_emg_stats(x_file_path, restimulus_file_path)
    np.save('emg_rest_mean.npy', emg_mean)
    np.save('emg_rest_std.npy', emg_std)
    print("EMG normalization parameters saved")

    # Compute and save glove normalization parameters
    print("Computing glove normalization parameters...")
    glove_min, glove_max = compute_glove_stats(y_file_path)
    glove_range = glove_max - glove_min
    glove_range[glove_range == 0] = 1  # Avoid division by zero
    np.save('glove_min.npy', glove_min)
    np.save('glove_range.npy', glove_range)
    print("Glove normalization parameters saved")

    print("All normalization parameters computed and saved successfully!")
except Exception as e:
    print(f"Error: {e}")
    print("Please check the structure of your HDF5 files and adjust the dataset names accordingly.")
```

---

## File: DB\manupilate\check1.py
**Path:** `C:\stage\stage\DB\manupilate\check1.py`

```python
import h5py
import numpy as np
import joblib

def check_data_issues():
    # Check normalization parameters
    emg_mean = np.load('emg_rest_mean.npy')
    emg_std = np.load('emg_rest_std.npy')
    glove_min = np.load('glove_min.npy')
    glove_range = np.load('glove_range.npy')
    
    print("EMG mean:", emg_mean)
    print("EMG std:", emg_std)
    print("Glove min:", glove_min)
    print("Glove range:", glove_range)
    
    # Check for NaN or inf values
    print("EMG mean has NaN:", np.any(np.isnan(emg_mean)))
    print("EMG std has NaN:", np.any(np.isnan(emg_std)))
    print("Glove min has NaN:", np.any(np.isnan(glove_min)))
    print("Glove range has NaN:", np.any(np.isnan(glove_range)))
    
    # Check NMF model
    nmf = joblib.load('nmf_model.pkl')
    print("NMF components shape:", nmf.components_.shape)
    
    # Check a small sample of data
    with h5py.File(r'D:\stage\x.h5', 'r') as f:
        emg_sample = f['data'][:1000, :]
        print("EMG sample shape:", emg_sample.shape)
        print("EMG sample has NaN:", np.any(np.isnan(emg_sample)))
        
    with h5py.File(r'D:\stage\y.h5', 'r') as f:
        glove_sample = f['labels'][:1000, :]
        print("Glove sample shape:", glove_sample.shape)
        print("Glove sample has NaN:", np.any(np.isnan(glove_sample)))

if __name__ == "__main__":
    check_data_issues()
    
```

---

## File: DB\manupilate\check_labels.py
**Path:** `C:\stage\stage\DB\manupilate\check_labels.py`

```python
import h5py

def check_hdf5_structure(file_path):
    print(f"Checking structure of {file_path}:")
    with h5py.File(file_path, 'r') as f:
        print("Keys in file:", list(f.keys()))
        for key in f.keys():
            print(f"Dataset '{key}' shape: {f[key].shape}")

# Check all HDF5 files
check_hdf5_structure(r'F:\DB\x.h5')
check_hdf5_structure(r'F:\DB\y.h5')
check_hdf5_structure(r'F:\DB\restimulus.h5')
```

---

## File: DB\manupilate\clean.py
**Path:** `C:\stage\stage\DB\manupilate\clean.py`

```python
import h5py
import numpy as np
import os

def delete_hdf5_rows(input_file, output_file, dataset_name, deletion_ranges):
    """
    Delete multiple non-consecutive row ranges from an HDF5 dataset.
    
    Args:
        input_file: Original HDF5 file path
        output_file: New HDF5 file path (will be created)
        dataset_name: Name of dataset to modify (e.g., 'data', 'restimulus', 'labels')
        deletion_ranges: List of (start_line, end_line) tuples (1-based global row numbers)
            Example: [(100, 200), (500, 600), (1000, 1100)]
    """
    # Convert deletion ranges to 0-based indices
    ranges_0based = [(start-1, end-1) for start, end in deletion_ranges]
    
    with h5py.File(input_file, 'r') as f_in:
        # Verify dataset exists
        if dataset_name not in f_in:
            raise KeyError(f"Dataset '{dataset_name}' not found in {input_file}")
        
        dset = f_in[dataset_name]
        total_rows = dset.shape[0]
        print(f"Original dataset has {total_rows} rows")
        
        # Calculate rows to keep
        rows_to_delete = 0
        for s, e in ranges_0based:
            if s < 0 or e >= total_rows:
                raise IndexError(f"Range {s+1}-{e+1} out of bounds (dataset has {total_rows} rows)")
            rows_to_delete += (e - s + 1)
        new_shape = (total_rows - rows_to_delete,) + dset.shape[1:]
        print(f"New dataset will have {new_shape[0]} rows ({rows_to_delete} rows deleted)")
        
        # Create new HDF5 file
        with h5py.File(output_file, 'w') as f_out:
            # Copy all other datasets and file attributes
            for name in f_in:
                if name == dataset_name:
                    # Create new dataset for filtered data
                    dset_out = f_out.create_dataset(
                        name,
                        shape=new_shape,
                        dtype=dset.dtype,
                        chunks=dset.chunks,
                        compression=dset.compression,
                        compression_opts=dset.compression_opts
                    )
                    
                    # Copy rows that are NOT in deletion ranges
                    current_idx = 0
                    for i in range(total_rows):
                        # Check if this row is in any deletion range
                        in_range = any(start <= i <= end for start, end in ranges_0based)
                        if not in_range:
                            dset_out[current_idx] = dset[i]
                            current_idx += 1
                    
                    # Copy dataset attributes
                    for attr_name, attr_value in dset.attrs.items():
                        dset_out.attrs[attr_name] = attr_value
                else:
                    # Copy other datasets unchanged
                    f_in.copy(name, f_out)
            
            # Copy file-level attributes
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value
    
    print(f"✅ Successfully created cleaned file: {output_file}")
    print(f"Original rows: {total_rows} → New rows: {new_shape[0]}")

# Example usage (CUSTOMIZE THESE VALUES)
if __name__ == "__main__":
    # 1. Verify your dataset name (run this first):
    # with h5py.File("your_file.h5", 'r') as f: print(f.keys())
    
    # 2. Configure deletion ranges (1-based global row numbers from your logs)
    deletion_ranges = [
        (240313855, 244527699),   # Delete rows 100-200
        (289278610, 294846247)  # Delete rows 1000-1100
    ]
    
    # 3. Set paths (EXAMPLE VALUES - REPLACE WITH YOURS)
    INPUT_FILE = "D:/stage/data/x.h5"  # Your original HDF5 file
    OUTPUT_FILE = "D:/stage/data/cleaned_x.h5"  # New file to create
    DATASET_NAME = "data"            # Check with: h5py.File(INPUT_FILE).keys()
    
    # 4. Run deletion
    delete_hdf5_rows(INPUT_FILE, OUTPUT_FILE, DATASET_NAME, deletion_ranges)
```

---

## File: DB\manupilate\clean1.py
**Path:** `C:\stage\stage\DB\manupilate\clean1.py`

```python
import h5py
import numpy as np
import os

def delete_hdf5_rows(input_file, output_file, dataset_name, deletion_ranges):
    """
    Delete multiple non-consecutive row ranges from an HDF5 dataset.
    
    Args:
        input_file: Original HDF5 file path
        output_file: New HDF5 file path (will be created)
        dataset_name: Name of dataset to modify (e.g., 'data', 'restimulus', 'labels')
        deletion_ranges: List of (start_line, end_line) tuples (1-based global row numbers)
            Example: [(100, 200), (500, 600), (1000, 1100)]
    """
    # Convert deletion ranges to 0-based indices
    ranges_0based = [(start-1, end-1) for start, end in deletion_ranges]
    
    with h5py.File(input_file, 'r') as f_in:
        # Verify dataset exists
        if dataset_name not in f_in:
            raise KeyError(f"Dataset '{dataset_name}' not found in {input_file}")
        
        dset = f_in[dataset_name]
        total_rows = dset.shape[0]
        print(f"Original dataset has {total_rows} rows")
        
        # Calculate rows to keep
        rows_to_delete = 0
        for s, e in ranges_0based:
            if s < 0 or e >= total_rows:
                raise IndexError(f"Range {s+1}-{e+1} out of bounds (dataset has {total_rows} rows)")
            rows_to_delete += (e - s + 1)
        new_shape = (total_rows - rows_to_delete,) + dset.shape[1:]
        print(f"New dataset will have {new_shape[0]} rows ({rows_to_delete} rows deleted)")
        
        # Create new HDF5 file
        with h5py.File(output_file, 'w') as f_out:
            # Copy all other datasets and file attributes
            for name in f_in:
                if name == dataset_name:
                    # Create new dataset for filtered data
                    dset_out = f_out.create_dataset(
                        name,
                        shape=new_shape,
                        dtype=dset.dtype,
                        chunks=dset.chunks,
                        compression=dset.compression,
                        compression_opts=dset.compression_opts
                    )
                    
                    # Copy rows that are NOT in deletion ranges
                    current_idx = 0
                    for i in range(total_rows):
                        # Check if this row is in any deletion range
                        in_range = any(start <= i <= end for start, end in ranges_0based)
                        if not in_range:
                            dset_out[current_idx] = dset[i]
                            current_idx += 1
                    
                    # Copy dataset attributes
                    for attr_name, attr_value in dset.attrs.items():
                        dset_out.attrs[attr_name] = attr_value
                else:
                    # Copy other datasets unchanged
                    f_in.copy(name, f_out)
            
            # Copy file-level attributes
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value
    
    print(f"✅ Successfully created cleaned file: {output_file}")
    print(f"Original rows: {total_rows} → New rows: {new_shape[0]}")

# Example usage (CUSTOMIZE THESE VALUES)
if __name__ == "__main__":
    # 1. Verify your dataset name (run this first):
    # with h5py.File("your_file.h5", 'r') as f: print(f.keys())
    
    # 2. Configure deletion ranges (1-based global row numbers from your logs)
    deletion_ranges = [
        (240313855, 244527699),   # Delete rows 100-200
        (289278610, 294846247)  # Delete rows 1000-1100
    ]
    
    # 3. Set paths (EXAMPLE VALUES - REPLACE WITH YOURS)
    INPUT_FILE = "D:/stage/data/y.h5"  # Your original HDF5 file
    OUTPUT_FILE = "D:/stage/data/cleaned_y.h5"  # New file to create
    DATASET_NAME = "labels"            # Check with: h5py.File(INPUT_FILE).keys()
    
    # 4. Run deletion
    delete_hdf5_rows(INPUT_FILE, OUTPUT_FILE, DATASET_NAME, deletion_ranges)
```

---

## File: DB\manupilate\clean2.py
**Path:** `C:\stage\stage\DB\manupilate\clean2.py`

```python
import h5py
import numpy as np
import os

def delete_hdf5_rows(input_file, output_file, dataset_name, deletion_ranges):
    """
    Delete multiple non-consecutive row ranges from an HDF5 dataset.
    
    Args:
        input_file: Original HDF5 file path
        output_file: New HDF5 file path (will be created)
        dataset_name: Name of dataset to modify (e.g., 'data', 'restimulus', 'labels')
        deletion_ranges: List of (start_line, end_line) tuples (1-based global row numbers)
            Example: [(100, 200), (500, 600), (1000, 1100)]
    """
    # Convert deletion ranges to 0-based indices
    ranges_0based = [(start-1, end-1) for start, end in deletion_ranges]
    
    with h5py.File(input_file, 'r') as f_in:
        # Verify dataset exists
        if dataset_name not in f_in:
            raise KeyError(f"Dataset '{dataset_name}' not found in {input_file}")
        
        dset = f_in[dataset_name]
        total_rows = dset.shape[0]
        print(f"Original dataset has {total_rows} rows")
        
        # Calculate rows to keep
        rows_to_delete = 0
        for s, e in ranges_0based:
            if s < 0 or e >= total_rows:
                raise IndexError(f"Range {s+1}-{e+1} out of bounds (dataset has {total_rows} rows)")
            rows_to_delete += (e - s + 1)
        new_shape = (total_rows - rows_to_delete,) + dset.shape[1:]
        print(f"New dataset will have {new_shape[0]} rows ({rows_to_delete} rows deleted)")
        
        # Create new HDF5 file
        with h5py.File(output_file, 'w') as f_out:
            # Copy all other datasets and file attributes
            for name in f_in:
                if name == dataset_name:
                    # Create new dataset for filtered data
                    dset_out = f_out.create_dataset(
                        name,
                        shape=new_shape,
                        dtype=dset.dtype,
                        chunks=dset.chunks,
                        compression=dset.compression,
                        compression_opts=dset.compression_opts
                    )
                    
                    # Copy rows that are NOT in deletion ranges
                    current_idx = 0
                    for i in range(total_rows):
                        # Check if this row is in any deletion range
                        in_range = any(start <= i <= end for start, end in ranges_0based)
                        if not in_range:
                            dset_out[current_idx] = dset[i]
                            current_idx += 1
                    
                    # Copy dataset attributes
                    for attr_name, attr_value in dset.attrs.items():
                        dset_out.attrs[attr_name] = attr_value
                else:
                    # Copy other datasets unchanged
                    f_in.copy(name, f_out)
            
            # Copy file-level attributes
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value
    
    print(f"✅ Successfully created cleaned file: {output_file}")
    print(f"Original rows: {total_rows} → New rows: {new_shape[0]}")

# Example usage (CUSTOMIZE THESE VALUES)
if __name__ == "__main__":
    # 1. Verify your dataset name (run this first):
    # with h5py.File("your_file.h5", 'r') as f: print(f.keys())
    
    # 2. Configure deletion ranges (1-based global row numbers from your logs)
    deletion_ranges = [
        (240313855, 244527699),   # Delete rows 100-200
        (289278610, 294846247)  # Delete rows 1000-1100
    ]
    
    # 3. Set paths (EXAMPLE VALUES - REPLACE WITH YOURS)
    INPUT_FILE = "D:/stage/data/restimulus.h5"  # Your original HDF5 file
    OUTPUT_FILE = "D:/stage/data/cleaned_restimulus.h5"  # New file to create
    DATASET_NAME = "restimulus"            # Check with: h5py.File(INPUT_FILE).keys()
    
    # 4. Run deletion
    delete_hdf5_rows(INPUT_FILE, OUTPUT_FILE, DATASET_NAME, deletion_ranges)
```

---

## File: DB\manupilate\clean_logs.py
**Path:** `C:\stage\stage\DB\manupilate\clean_logs.py`

```python
import csv

def repack_log_after_manual_cleanup(input_csv: str, output_csv: str):
    """
    After you manually delete corrupt file entries from the log,
    this repacks the remaining entries continuously:
        start_line = previous_end_line + 1
        end_line = start_line + rows_extracted - 1
    """
    new_entries = []
    next_start = 1  # Start from global row 1

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            filename = row[0]
            # We ignore original start/end — recalculate based on continuity
            rows_extracted = int(row[3])

            start_line = next_start
            end_line = start_line + rows_extracted - 1

            new_entries.append([filename, start_line, end_line, rows_extracted])

            # Next file starts right after this one ends
            next_start = end_line + 1

    # Write to new CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'start_line', 'end_line', 'rows_extracted'])  # Header
        writer.writerows(new_entries)

    print(f"✅ Repacked log saved to: {output_csv}")
    print(f"Total rows in new log: {next_start - 1}")


# ====== CONFIGURE PATHS ======
INPUT_LOG = "D:/stage/v2.0/map.csv"    # 👈 Your manually cleaned CSV (with bad lines removed)
OUTPUT_LOG = "D:/stage/v2.0/cleaned_map.csv"

repack_log_after_manual_cleanup(INPUT_LOG, OUTPUT_LOG)
```

---

## File: DB\manupilate\names.py
**Path:** `C:\stage\stage\DB\manupilate\names.py`

```python
import h5py
with h5py.File("D:\\stage\\data\\x.h5", 'r') as f:
    print(f.keys())
with h5py.File("D:\\stage\\data\\y.h5", 'r') as f:
    print(f.keys())
with h5py.File("D:\\stage\\data\\restimulus.h5", 'r') as f:
    print(f.keys())

```

---

## File: prosthetic-hand-visualizer\blender_export.py
**Path:** `C:\stage\stage\prosthetic-hand-visualizer\blender_export.py`

```python
# blender_export.py
def export_hand_model():
    """Export the rigged hand model to GLTF format"""
    
    bpy.ops.export_scene.gltf(
        filepath="C:/path/to/your/hand_model.gltf",  # Update this path
        export_format='GLTF_EMBEDDED',
        export_yup=True,
        export_apply=True,
        export_animations=False,
        export_skins=True,
        export_morph=False
    )

export_hand_model()
```

---

## File: prosthetic-hand-visualizer\blender_hand_mesh.py
**Path:** `C:\stage\stage\prosthetic-hand-visualizer\blender_hand_mesh.py`

```python
# blender_hand_mesh.py
def create_hand_mesh():
    """Create the hand geometry and skin it to the armature"""
    
    # Create basic hand shape using metaballs or sculpting
    bpy.ops.object.metaball_add(type='BALL', location=(0, 0, 0))
    hand_mesh = bpy.context.object
    hand_mesh.name = "Hand_Mesh"
    
    # Add subsurface modifier for smoothness
    hand_mesh.modifiers.new("Subdivision", type='SUBSURF')
    hand_mesh.modifiers["Subdivision"].levels = 2
    
    return hand_mesh

def rig_hand_to_armature(hand_mesh, armature):
    """Skin the hand mesh to the armature"""
    
    # Add armature modifier
    armature_modifier = hand_mesh.modifiers.new("Armature", type='ARMATURE')
    armature_modifier.object = armature
    
    # Enter weight painting mode (you'd need to paint weights properly)
    bpy.context.view_layer.objects.active = hand_mesh
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    
    return hand_mesh

# Complete the setup
hand_mesh = create_hand_mesh()
rigged_hand = rig_hand_to_armature(hand_mesh, armature)
```

---

## File: prosthetic-hand-visualizer\blender_hand_setup.py
**Path:** `C:\stage\stage\prosthetic-hand-visualizer\blender_hand_setup.py`

```python
# blender_hand_setup.py
import bpy
import bmesh
import mathutils

def create_hand_skeleton():
    # Clear existing mesh
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Create armature (skeleton)
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.object
    armature.name = "Hand_Armature"
    
    return armature

def create_bone_hierarchy():
    """Create the complete 22-bone hierarchy for the hand"""
    bones = {}
    
    # Wrist bone (root)
    bpy.ops.armature.bone_primitive_add()
    wrist = bpy.context.active_bone
    wrist.name = "wrist"
    bones['wrist'] = wrist
    
    # Finger bones creation function
    def create_finger(finger_name, parent_bone, positions):
        finger_bones = []
        current_parent = parent_bone
        
        for i, pos in enumerate(positions):
            bpy.ops.armature.bone_primitive_add()
            bone = bpy.context.active_bone
            bone.name = f"{finger_name}_{i}"
            bone.parent = current_parent
            bone.head = current_parent.tail
            bone.tail = pos
            finger_bones.append(bone)
            current_parent = bone
            
        return finger_bones
    
    # Define finger positions (simplified - you'd adjust these)
    finger_positions = {
        'thumb': [
            mathutils.Vector((0.1, 0, 0.1)),    # CMC
            mathutils.Vector((0.2, 0, 0.2)),    # MCP  
            mathutils.Vector((0.3, 0, 0.25)),   # IP
            mathutils.Vector((0.35, 0, 0.3))    # Tip
        ],
        'index': [
            mathutils.Vector((0.1, 0.05, 0)),   # MCP
            mathutils.Vector((0.2, 0.05, 0)),   # PIP
            mathutils.Vector((0.3, 0.05, 0)),   # DIP
            mathutils.Vector((0.35, 0.05, 0))   # Tip
        ]
        # Add middle, ring, pinky similarly...
    }
    
    # Create all fingers
    bones['thumb'] = create_finger('thumb', bones['wrist'], finger_positions['thumb'])
    bones['index'] = create_finger('index', bones['wrist'], finger_positions['index'])
    # Add other fingers...
    
    return bones

# Run the setup
armature = create_hand_skeleton()
bones = create_bone_hierarchy()
```

---

## File: prosthetic-hand-visualizer\enhanced_server.py
**Path:** `C:\stage\stage\prosthetic-hand-visualizer\enhanced_server.py`

```python
# enhanced_server.py
import numpy as np

class BiomechanicalConstraints:
    """Apply biomechanical constraints to joint angles"""
    
    @staticmethod
    def apply_finger_constraints(glove_values):
        """Ensure finger movements are biomechanically realistic"""
        constrained_values = glove_values.copy()
        
        # Thumb constraints
        constrained_values[0] = np.clip(glove_values[0], 0, 0.8)  # CMC flexion
        constrained_values[1] = np.clip(glove_values[1], 0, 0.6)  # CMC abduction
        
        # Finger coupling constraints (PIP/DIP relationship)
        for finger_start in [4, 8, 12, 16]:  # MCP indices for each finger
            pip_index = finger_start + 2
            dip_index = finger_start + 3
            
            # Ensure DIP flexion doesn't exceed PIP flexion
            if constrained_values[dip_index] > constrained_values[pip_index]:
                constrained_values[dip_index] = constrained_values[pip_index] * 0.8
        
        return constrained_values

# Enhanced WebSocket endpoint with constraints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    biomechanical = BiomechanicalConstraints()
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "next_frame":
                # ... existing data loading code ...
                
                # Apply biomechanical constraints
                constrained_truth = biomechanical.apply_finger_constraints(ground_truth_glove_values)
                constrained_pred = biomechanical.apply_finger_constraints(predicted_glove_values)
                
                payload = {
                    "ground_truth": constrained_truth.tolist(),
                    "prediction": constrained_pred.tolist(),
                    "metrics": {
                        "mse": mse,
                        "constrained_mse": calculate_mse(constrained_truth, constrained_pred)
                    }
                }
                
                await websocket.send_text(json.dumps(payload))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## File: prosthetic-hand-visualizer\backend\server.py
**Path:** `C:\stage\stage\prosthetic-hand-visualizer\backend\server.py`

```python
# backend/server.py
import asyncio
import websockets
import json
import random  # We'll use fake data first

async def send_hand_data(websocket):
    print("Client connected")
    try:
        while True:
            # Generate fake hand data for testing
            ground_truth = [random.random() for _ in range(22)]  # 22 random values 0-1
            prediction = [random.random() for _ in range(22)]    # 22 random values 0-1
            
            data = {
                "ground_truth": ground_truth,
                "prediction": prediction,
                "metrics": {"mse": random.random() * 0.1}
            }
            
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.1)  # Send 10 times per second
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(send_hand_data, "localhost", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
```

---

## File: PROSTHETIC_AI_CORE\scrab.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\scrab.py`

```python
import os

# Define the root directory and the output file
root_dir = r"C:\stage\stage"
output_file = "CODEBASE_SUMMARY.md"

def collect_python_files(target_dir, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Project Source Code Summary\n\n")
        
        for subdir, dirs, files in os.walk(target_dir):
            # Skip hidden directories and pycache
            if '__pycache__' in subdir or '.git' in subdir:
                continue
                
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(subdir, file)
                    relative_path = os.path.relpath(file_path, target_dir)
                    
                    f.write(f"## File: {relative_path}\n")
                    f.write(f"**Path:** `{file_path}`\n\n")
                    f.write("```python\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as code_file:
                            f.write(code_file.read())
                    except Exception as e:
                        f.write(f"# Error reading file: {e}")
                    f.write("\n```\n\n---\n\n")

    print(f"Summary generated: {output_path}")

if __name__ == "__main__":
    collect_python_files(root_dir, output_file)
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\ofgjeofg.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\ofgjeofg.py`

```python

```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\phase2_plot.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\phase2_plot.py`

```python
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
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_filtered_signal.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_filtered_signal.py`

```python
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
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_filtered_signal1.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_filtered_signal1.py`

```python
# Save this as plot_notch_filter_effect.py
# This script specifically isolates and visualizes the effect of *only*
# the 50Hz notch filter on a small slice of the raw signal.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv" # Use the same file path as before

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

# --- ⚙️ TIME SLICE TO PLOT (ZOOMED IN) ---
# We'll find a "noisy" section, often during rest
START_SAMPLE = 30000 
# We'll plot a tiny 0.2-second slice (400 samples)
DURATION_SAMPLES = 400 

# --- ⚙️ FILTER PARAMETERS ---
FS = 2000.0 # 2000 Hz
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
# We are *NOT* using the bandpass filter in this script

# --- 🛠️ HELPER FUNCTION: Filter Design ---
def design_notch_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Creates *only* the 50Hz notch filter."""
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs)
    return b_notch, a_notch

# --- Main Plotting Function ---
def plot_filter_comparison(file_path):
    print(f"Loading raw data for {CHANNEL_TO_PLOT} from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the single channel we want to inspect
        df = pd.read_csv(file_path, usecols=[CHANNEL_TO_PLOT])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal
    signal_raw = df[CHANNEL_TO_PLOT].values
    
    # 3. Create the filtered signal (NOTCH ONLY)
    print(f"   ... Applying ONLY the 50Hz notch filter...")
    b_notch, a_notch = design_notch_filter(FS)
    signal_notched = signal.filtfilt(b_notch, a_notch, signal_raw)
    print("   ... Filtering complete.")

    # 4. Get the *slice* of both signals
    stop_sample = START_SAMPLE + DURATION_SAMPLES
    raw_slice = signal_raw[START_SAMPLE:stop_sample]
    notched_slice = signal_notched[START_SAMPLE:stop_sample]
    
    # Create a time axis in milliseconds
    time_axis_ms = (np.arange(len(raw_slice)) / FS) * 1000.0 

    # 5. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # --- Plot 1: Raw Signal (should show 50Hz "fuzz") ---
    ax1.plot(time_axis_ms, raw_slice, label="Raw Signal", color='r', alpha=0.9)
    ax1.set_title(f"BEFORE: Raw {CHANNEL_TO_PLOT} (Note the 50Hz 'fuzz')", fontsize=14)
    ax1.set_ylabel("Raw EMG Value")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Notched Signal (should be smooth) ---
    ax2.plot(time_axis_ms, notched_slice, label="Notch Filtered Signal", color='b')
    ax2.set_title(f"AFTER: {CHANNEL_TO_PLOT} with 50Hz Hum Removed", fontsize=14)
    ax2.set_ylabel("Filtered EMG Value")
    ax2.set_xlabel(f"Time (milliseconds) - A {DURATION_SAMPLES/FS:.2f} second slice", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = "notch_filter_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... This plot *only* shows the removal of the 50Hz hum.")

if __name__ == "__main__":
    plot_filter_comparison(RAW_FILE_TO_INSPECT)
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_psd_effect.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_psd_effect.py`

```python
# Save this as plot_psd_effect.py
# This script visualizes the *frequency domain* (PSD)
# to show the effect of the filters.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

# --- ⚙️ FILTER PARAMETERS ---
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
def plot_psd_comparison(file_path):
    print(f"Loading raw data for {CHANNEL_TO_PLOT} from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the single channel we want to inspect
        df = pd.read_csv(file_path, usecols=[CHANNEL_TO_PLOT, 'restimulus'])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal *during movement* (more signal to see)
    signal_raw = df[df['restimulus'] != 0][CHANNEL_TO_PLOT].values
    if len(signal_raw) == 0:
        print("   ⚠️ Warning: No movement data found, using 'rest' data.")
        signal_raw = df[CHANNEL_TO_PLOT].values

    # 3. Create the filtered signal
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    signal_filtered = signal.filtfilt(b_notch, a_notch, signal_raw)
    signal_filtered = signal.filtfilt(b_band, a_band, signal_filtered)
    print("   ... Filtering complete.")

    # 4. Calculate the Power Spectral Density (PSD) for both signals
    # This is the "frequency graph" you were asking for.
    print("   ... Calculating Power Spectral Density (PSD)...")
    # Using signal.welch() is a standard method to get a clean PSD
    f_raw, Pxx_raw = signal.welch(signal_raw, FS, nperseg=2048)
    f_filtered, Pxx_filtered = signal.welch(signal_filtered, FS, nperseg=2048)

    # 5. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.suptitle("Filter Effect in the Frequency Domain (Power Spectral Density)", fontsize=16)
    
    # --- Plot 1: Raw Signal PSD ---
    ax1.semilogy(f_raw, Pxx_raw, label="Raw Signal", color='r', alpha=0.8)
    ax1.set_title(f"BEFORE: Raw {CHANNEL_TO_PLOT} (Note the 50Hz spike)")
    ax1.set_ylabel("Power / Frequency (dB/Hz) - Log Scale")
    ax1.axvline(NOTCH_FREQ, color='k', linestyle='--', label="50Hz Hum")
    ax1.axvspan(0, BAND_LOW, color='gray', alpha=0.3, label="Low-freq Noise")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Filtered Signal PSD ---
    ax2.semilogy(f_filtered, Pxx_filtered, label="Filtered Signal", color='b')
    ax2.set_title(f"AFTER: {CHANNEL_TO_PLOT} (Noise successfully removed)")
    ax2.set_ylabel("Power / Frequency (dB/Hz) - Log Scale")
    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.axvline(NOTCH_FREQ, color='k', linestyle='--', label="50Hz Hum (Removed)")
    ax2.axvspan(0, BAND_LOW, color='gray', alpha=0.3, label="Low-freq Noise (Removed)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Limit x-axis to 0-500Hz to see the details
    ax2.set_xlim(0, 500)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "psd_filter_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... This plot *proves* the filters are working. Look for the 50Hz spike in the top plot and see it's gone in the bottom plot.")

if __name__ == "__main__":
    plot_psd_comparison(RAW_FILE_TO_INSPECT)
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_raw_glove_targets.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_raw_glove_targets.py`

```python
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
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_raw_signal.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_raw_signal.py`

```python
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
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_rectification_effect.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_rectification_effect.py`

```python
# Save this as plot_rectification_effect.py
# This script visualizes the effect of Rectification (absolute value)
# on a filtered and Z-scored signal.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

# --- ⚙️ TIME SLICE TO PLOT (ZOOMED IN) ---
# We'll find a "noisy" section, often during rest
START_SAMPLE = 30000 
# We'll plot a 0.5-second slice (1000 samples) to see the wave clearly
DURATION_SAMPLES = 1000 

# --- ⚙️ FILTER PARAMETERS ---
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
def plot_rectification_comparison(file_path):
    print(f"Loading raw data for {CHANNEL_TO_PLOT} from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the single channel we want to inspect
        df = pd.read_csv(file_path, usecols=[CHANNEL_TO_PLOT])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal
    signal_raw = df[CHANNEL_TO_PLOT].values.reshape(-1, 1) # Reshape for scaler
    
    # 3. Create the filtered signal
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    
    signal_filtered = signal.filtfilt(b_notch, a_notch, signal_raw, axis=0)
    signal_filtered = signal.filtfilt(b_band, a_band, signal_filtered, axis=0)

    # 4. Create the Z-scored signal ("BEFORE" data)
    print("   ... Applying Z-Score (StandardScaler)...")
    scaler = StandardScaler()
    signal_zscored = scaler.fit_transform(signal_filtered)
    print("   ... Z-Score complete.")
    
    # 5. Create the Rectified signal ("AFTER" data)
    print("   ... Applying Rectification (Absolute Value)...")
    signal_rectified = np.abs(signal_zscored)

    # 6. Get the *slice* of both signals for plotting
    stop_sample = START_SAMPLE + DURATION_SAMPLES
    zscored_slice = signal_zscored[START_SAMPLE:stop_sample]
    rectified_slice = signal_rectified[START_SAMPLE:stop_sample]
    
    # Create a time axis in milliseconds
    time_axis_ms = (np.arange(len(zscored_slice)) / FS) * 1000.0 

    # 7. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.suptitle("Effect of Rectification (Absolute Value)", fontsize=16)
    
    # --- Plot 1: Z-Scored Signal (Before) ---
    ax1.plot(time_axis_ms, zscored_slice, label="Z-Scored Signal", color='b')
    ax1.axhline(0, color='k', linestyle='--', label="Zero line")
    ax1.set_title(f"BEFORE: Z-Scored {CHANNEL_TO_PLOT} (Positive and Negative Values)", fontsize=14)
    ax1.set_ylabel("Normalized Value (Std. Dev)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Rectified Signal (After) ---
    ax2.plot(time_axis_ms, rectified_slice, label="Rectified Signal", color='g')
    ax2.axhline(0, color='k', linestyle='--', label="Zero line")
    ax2.set_title(f"AFTER: Rectified {CHANNEL_TO_PLOT} (All Non-Negative)", fontsize=14)
    ax2.set_ylabel("Normalized Value (Absolute)")
    ax2.set_xlabel(f"Time (milliseconds) - A {DURATION_SAMPLES/FS:.2f} second slice", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "rectification_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... The top plot shows the wave centered at 0. The bottom plot shows all negative parts 'flipped' to positive.")

if __name__ == "__main__":
    plot_rectification_comparison(RAW_FILE_TO_INSPECT)
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_zscore_effect.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\plot_zscore_effect.py`

```python
# Save this as plot_zscore_effect.py
# This script visualizes the statistical distribution of the 12 EMG channels
# *before* and *after* Z-Score normalization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The 12 channels we will analyze
EMG_COLS = [f"emg_{i}" for i in range(1, 13)]

# We'll use a large chunk of data for a good statistical sample
# Let's take 50,000 samples (25 seconds) from a movement section
START_SAMPLE = 30000 
SAMPLE_COUNT = 50000

# --- ⚙️ FILTER PARAMETERS ---
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
def plot_zscore_comparison(file_path):
    print(f"Loading raw data from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load *only* the EMG data
        df = pd.read_csv(file_path, usecols=EMG_COLS)
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal
    signal_raw = df.values
    
    # 3. Create the filtered signal ("BEFORE" data)
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    
    signal_filtered = np.zeros_like(signal_raw)
    for i in range(signal_raw.shape[1]): # Filter each channel
        signal_filtered[:, i] = signal.filtfilt(b_notch, a_notch, signal_raw[:, i])
        signal_filtered[:, i] = signal.filtfilt(b_band, a_band, signal_filtered[:, i])
    print("   ... Filtering complete.")

    # 4. Create the Z-scored signal ("AFTER" data)
    print("   ... Applying Z-Score (StandardScaler)...")
    scaler = StandardScaler()
    signal_zscored = scaler.fit_transform(signal_filtered)
    print("   ... Z-Score complete.")

    # 5. Get the *slice* of both signals for plotting
    stop_sample = START_SAMPLE + SAMPLE_COUNT
    filtered_slice = signal_filtered[START_SAMPLE:stop_sample]
    zscored_slice = signal_zscored[START_SAMPLE:stop_sample]
    
    # 6. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.suptitle("Effect of Z-Score Normalization on EMG Channel Distributions", fontsize=16)
    
    # --- Plot 1: BEFORE Z-Score ---
    ax1.set_title("BEFORE: Filtered Data (12 Channels)")
    ax1.set_xlabel("Signal Value (Amplitude)")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)
    # This will plot 12 *different* density curves
    for i in range(filtered_slice.shape[1]):
        sns.kdeplot(filtered_slice[:, i], ax=ax1, label=f'emg_{i+1}', warn_singular=False)
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    # This plot should look "messy" with 12 different peaks and widths

    # --- Plot 2: AFTER Z-Score ---
    ax2.set_title("AFTER: Z-Scored Data (12 Channels)")
    ax2.set_xlabel("Normalized Value (Standard Deviations)")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)
    # This will plot 12 *identical* density curves
    for i in range(zscored_slice.shape[1]):
        sns.kdeplot(zscored_slice[:, i], ax=ax2, warn_singular=False)
    # This plot should show one single, clean bell curve
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "zscore_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... The 'BEFORE' plot shows 12 different distributions.")
    print("   ... The 'AFTER' plot should show one single, centered distribution (a perfect bell curve).")

if __name__ == "__main__":
    sns.set_theme() # Use Seaborn's nice default styling
    plot_zscore_comparison(RAW_FILE_TO_INSPECT)
```

---

## File: PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\test.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\docs\graphs\rapport graphs\test.py`

```python

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v10.0\analyze_10hz_data.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v10.0\analyze_10hz_data.py`

```python
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
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v10.0\convert_to_10hz_csv.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v10.0\convert_to_10hz_csv.py`

```python
# Save this as convert_to_10hz_csv.py
# This script converts raw 2000Hz CSVs into a single 10Hz feature-engineered CSV
# for Exploratory Data Analysis (EDA), with verbose logging.

import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # We will define the file list in the main() function.
    OUTPUT_CSV_PATH = "10hz_feature_data.csv" # The SINGLE output CSV
    
    # Signal Processing
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # Feature Engineering (Downsampling)
    WINDOW_MS = 100 # 100ms window
    WINDOW_SAMPLES = int(FS * (WINDOW_MS / 1000.0)) # = 200 samples

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def calculate_hudgins_features(window: np.ndarray) -> np.ndarray:
    """
    Calculates the 4 Hudgins' features (RMS, WL, ZC, SSC) for a 12-channel window.
    Input window shape: (200, 12)
    Output feature vector shape: (48,)
    """
    n_samples, n_channels = window.shape
    features = []
    
    # 1. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2, axis=0))
    features.append(rms)
    
    # 2. Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    features.append(wl)
    
    # 3. Zero Crossings (ZC) (with a small threshold for noise)
    zc = np.sum((window[:-1] * window[1:] < 0) & (np.abs(window[:-1] - window[1:]) > 1e-6), axis=0)
    features.append(zc)
    
    # 4. Slope Sign Changes (SSC) (with a small threshold for noise)
    ssc = np.sum((window[1:-1] - window[:-2]) * (window[1:-1] - window[2:]) > 1e-6, axis=0)
    features.append(ssc)
    
    return np.concatenate(features, axis=0).astype(np.float32)

# --- 1. DATA GATHERING (2000Hz -> 10Hz) ---
def build_10hz_list_of_rows(file_list: List[str]) -> List[Dict]:
    """
    Processes all raw CSVs, extracts features, and returns a list of rows
    (as dictionaries) for the 10Hz data.
    """
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_rows = [] # List to hold all our new 10Hz rows (as dicts)
    group_id_counter = 0 # This is the critical ID for preventing sequence breaks
    
    print("Processing Raw Files (2000Hz -> 10Hz)...")
    
    for file_index, fp in enumerate(tqdm(file_list, desc="Overall Progress", unit="file", total=len(file_list))):
        print(f"\nProcessing File {file_index+1}/{len(file_list)}: {os.path.basename(fp)}")
        try:
            # Load all necessary columns
            use_cols = [f"emg_{i}" for i in range(1, 13)] + [f"glove_{i}" for i in range(1, 23)] + ["restimulus"]
            df = pd.read_csv(fp, usecols=lambda c: c in use_cols)
            
            emg_cols = [f"emg_{i}" for i in range(1, 13)]
            glove_cols = [f"glove_{i}" for i in range(1, 23)]
            
            emg_data = df[emg_cols].values
            glove_data = df[glove_cols].values
            labels = df['restimulus'].values.astype(int)
            
            # Filter the raw EMG data
            for i in range(emg_data.shape[1]):
                emg_data[:, i] = signal.filtfilt(b_notch, a_notch, emg_data[:, i])
                emg_data[:, i] = signal.filtfilt(b_band, a_band, emg_data[:, i])
            
            # Find boundaries of "pure" chunks
            boundaries = np.where(np.diff(labels) != 0)[0] + 1
            chunk_starts = np.concatenate(([0], boundaries))
            chunk_stops = np.concatenate((boundaries, [len(labels)]))
            
            print(f"   ... Found {len(chunk_starts)} 'pure' movement/rest chunks.")
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                if start >= stop: continue
                
                chunk_emg = emg_data[start:stop]
                chunk_glove = glove_data[start:stop]
                chunk_label = labels[start] # All labels in this chunk are the same
                
                n_samples = len(chunk_emg)
                # Use non-overlapping windows, discard partial remainder
                n_windows = n_samples // Config.WINDOW_SAMPLES 
                remainder = n_samples % Config.WINDOW_SAMPLES
                
                if n_windows == 0:
                    print(f"   LOG: Discarding chunk (rows {start}-{stop}, label {chunk_label}) - too short ({n_samples} samples) for one window.")
                    continue
                
                # *** YOUR REQUESTED LOGGING ***
                if remainder > 0:
                    print(f"   LOG: Discarding last {remainder} samples from chunk (rows {start}-{stop}, label {chunk_label}) to ensure pure windows.")
                
                # Process all *full* non-overlapping windows in this chunk
                for i in range(n_windows):
                    win_start = start + (i * Config.WINDOW_SAMPLES)
                    win_stop = win_start + Config.WINDOW_SAMPLES
                    
                    # We reference the *original* dataframe slices
                    emg_window = emg_data[win_start:win_stop]
                    glove_window = glove_data[win_start:win_stop]
                    
                    features_48 = calculate_hudgins_features(emg_window)
                    target_22 = glove_window[-1, :].astype(np.float32)
                    
                    # Create a dictionary for this 10Hz row
                    row_data = {
                        'group_id': group_id_counter, # This ID marks the 'pure' chunk
                        'restimulus': chunk_label
                    }
                    # Add features
                    for f_idx in range(48):
                        row_data[f'feat_{f_idx+1}'] = features_48[f_idx] # Use 1-indexing
                    # Add targets
                    for t_idx in range(22):
                        row_data[f'glove_{t_idx+1}'] = target_22[t_idx] # Use 1-indexing
                        
                    all_rows.append(row_data)
                
                group_id_counter += 1 # Increment for each new, pure chunk
                
        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process file {fp}. Error: {e}")
            continue
            
    print(f"\n... 10Hz data extraction complete. Found {len(all_rows)} total 10Hz samples.")
    return all_rows

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING CONVERSION TO 10Hz CSV (PHASE 2a)")
    print("=============================================================")
    
    # ==================== EDIT THIS LIST ====================
    # Paste your file paths here for the test run.
    # Use 'r' before the string to handle Windows backslashes '\'.
    FILES_TO_PROCESS = [
        r"D:\stage\v5.0\data\DB2\s1\E1_A1.csv",
        r"DS:\stage\v5.0\data\DB2\s1\E2_A1.csv",
        r"D:\stage\v5.0\data\DB2\s2\E1_A1.csv"
        # Add more files here when you are ready for the full run
    ]
    # ========================================================
    
    print(f"Starting processing for {len(FILES_TO_PROCESS)} manually specified files...")

    # --- STEP 1: Process 2000Hz -> 10Hz ---
    list_of_rows = build_10hz_list_of_rows(FILES_TO_PROCESS)
    
    if not list_of_rows:
        print("❌ FATAL: No 10Hz data could be extracted. Check file paths and data.")
        return

    # --- STEP 2: Convert list of dicts to DataFrame and save ---
    print(f"\n... Converting to DataFrame and saving to {Config.OUTPUT_CSV_PATH}...")
    try:
        final_df = pd.DataFrame(list_of_rows)
        # We will use CSV as you requested, not Parquet.
        final_df.to_csv(Config.OUTPUT_CSV_PATH, index=False)
        print(f"   ... Saved as {Config.OUTPUT_CSV_PATH}.")

    except Exception as e:
        print(f"❌ FATAL: Failed to save DataFrame to CSV. Error: {e}")
        return

    print("\n" + "=" * 60)
    print(f"🎉 CONVERSION TO 10Hz CSV COMPLETE!")
    print(f"   Total 10Hz samples created: {len(list_of_rows):,}")
    print(f"   Output file: {Config.OUTPUT_CSV_PATH}")
    print("\n   ✅ YOU CAN NOW OPEN THIS CSV FILE TO SEE THE DATA.")
    print("\nNext step: Run the 'Phase 2b' analysis script on this new file.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v10.0\convert_to_tfrecord.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v10.0\convert_to_tfrecord.py`

```python
# Save this as convert_to_tfrecord.py
# This is the FINAL data conversion script (Phase 2c).
# It converts the 10Hz CSV into a balanced, normalized, sequential TFRecord dataset.

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Input Files ---
    DATA_CSV_FILE = "10hz_feature_data.csv"
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    # --- Output ---
    OUTPUT_DIR = "tfrecord_dataset_v2"
    FEATURE_SCALER_FILE = "feature_scaler.pkl"
    TARGET_SCALER_FILE = "target_scaler.pkl" # We save the Min/Max as a scaler
    
    # --- Clustering & Balancing ---
    NUM_MOVEMENT_CLUSTERS = 6
    
    # --- Sequencing ---
    SEQUENCE_LENGTH = 50 # 50 samples * 100ms/sample = 5 seconds
    
    # --- TFRecord ---
    SEQUENCES_PER_SHARD = 500
    VALIDATION_SPLIT = 0.15 # 15% of our balanced sequences will be for validation

# --- 🛠️ HELPER FUNCTIONS ---

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the 10Hz CSV data."""
    print(f"Loading 10Hz data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        raise FileNotFoundError
    
    df = pd.read_csv(file_path)
    print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
    return df

def create_virtual_motors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Creates the 7 Virtual Motor columns based on the sdata201453.pdf paper."""
    print("--- 1. Creating 7 Virtual Motors ---")
    
    # Sensor map based on sdata201453.pdf
    VIRTUAL_MOTORS: Dict[str, List[str]] = {
        "motor_thumb_flex": [f"glove_{i}" for i in [3, 4]],
        "motor_index_flex": [f"glove_{i}" for i in [6, 7, 8]],
        "motor_middle_flex": [f"glove_{i}" for i in [10, 11, 12]],
        "motor_ring_flex": [f"glove_{i}" for i in [14, 15, 16]],
        "motor_pinky_flex": [f"glove_{i}" for i in [18, 19, 20]],
        "motor_thumb_abduct": ["glove_2"],
        "motor_wrist_flex": ["glove_21"]
    }
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
    
    # Get the input feature column names (feat_1 to feat_48)
    feature_cols = [f'feat_{i+1}' for i in range(48)]
    
    print(f"   ... Created {len(motor_cols)} motor columns.")
    return df, feature_cols, motor_cols

def normalize_data(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], norm_params_file: str, out_dir: str) -> pd.DataFrame:
    """
    Applies Min-Max normalization to motors (targets) and Z-Score to features (inputs).
    """
    print("--- 2. Normalizing Data ---")
    
    # --- A. Normalize Motors (Targets) ---
    print(f"   ... Applying Min-Max (0-1) normalization to {len(m_cols)} motors...")
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    for motor in m_cols:
        p = params[motor]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        
        # Ensure range is not zero to avoid division by zero
        if motor_range == 0: motor_range = 1.0 
        
        # Apply normalization and clip between 0 and 1
        df[motor] = (df[motor] - motor_min) / motor_range
        df[motor] = np.clip(df[motor], 0.0, 1.0)
    
    # Save the Min/Max params as our "target scaler" for the model
    # We will need to "un-normalize" the model's output later
    joblib.dump(params, os.path.join(out_dir, Config.TARGET_SCALER_FILE))

    # --- B. Normalize Features (Inputs) ---
    print(f"   ... Applying Z-Score normalization to {len(f_cols)} features...")
    feature_data = df[f_cols].values
    feature_scaler = StandardScaler()
    
    # Fit and transform the features
    features_scaled = feature_scaler.fit_transform(feature_data)
    
    # Save the scaler for the model to use on new data
    joblib.dump(feature_scaler, os.path.join(out_dir, Config.FEATURE_SCALER_FILE))
    
    # Put the normalized features back into the DataFrame
    df[f_cols] = features_scaled
    print("   ... Normalization complete. Scalers saved.")
    return df

def balance_data(df: pd.DataFrame, m_cols: List[str]) -> pd.DataFrame:
    """
    Performs stratified sampling using 'rest' + K-Means movement clusters.
    """
    print("--- 3. Balancing Data (Stratified Sampling) ---")
    
    # Separate Rest and Movement data
    rest_df = df[df['restimulus'] == 0].copy()
    move_df = df[df['restimulus'] != 0].copy()
    
    if rest_df.empty or move_df.empty:
        print("   ⚠️ Warning: Missing rest or movement data. Balancing may be skewed.")
        return df # Return original df if we can't balance

    print(f"   ... Found {len(rest_df)} 'rest' samples and {len(move_df)} 'movement' samples.")
    
    # --- A. Cluster Movements ---
    print(f"   ... Finding {Config.NUM_MOVEMENT_CLUSTERS} movement archetypes (K-Means)...")
    # We cluster on the *normalized* motor data
    kmeans = KMeans(n_clusters=Config.NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_df['cluster'] = kmeans.fit_predict(move_df[m_cols])
    
    # --- B. Stratified Sampling ---
    all_balanced_dfs = []
    
    # Find the count of each cluster
    cluster_counts = move_df['cluster'].value_counts()
    
    # Find the "minority class" (smallest movement cluster)
    min_move_count = cluster_counts.min()
    print(f"   ... Minority movement cluster has {min_move_count} samples.")
    
    # Sample from movement clusters
    for i in range(Config.NUM_MOVEMENT_CLUSTERS):
        cluster_df = move_df[move_df['cluster'] == i]
        all_balanced_dfs.append(cluster_df.sample(n=min_move_count, random_state=42, replace=False))
        
    # Sample from "Rest"
    # We will take an equal amount of "rest" samples
    rest_sample_size = min(len(rest_df), min_move_count)
    print(f"   ... Sampling {rest_sample_size} 'rest' samples.")
    all_balanced_dfs.append(rest_df.sample(n=rest_sample_size, random_state=42, replace=False))
    
    # Combine all balanced data
    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} samples.")
    return balanced_df

def write_sequences_to_tfrecord(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], output_dir: str):
    """
    Builds sequences from the balanced/normalized 10Hz data and saves to TFRecord.
    """
    print("--- 4. Creating Sequences & Saving to TFRecord ---")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Group by the 'group_id' to prevent sequences from crossing boundaries
    grouped = df.groupby('group_id')
    
    train_writer = None
    val_writer = None
    shard_counts = {'train': 0, 'val': 0}
    seq_counts = {'train': 0, 'val': 0}

    # Iterate over each "pure" chunk
    for group_id, group_df in tqdm(grouped, desc="   ... Processing groups into sequences", unit="group"):
        if len(group_df) < Config.SEQUENCE_LENGTH:
            continue # This chunk is too short to make a sequence
            
        features = group_df[f_cols].values
        targets = group_df[m_cols].values.astype(np.float32) # Ensure targets are float32
        
        # Slide our 50-sample window over this pure chunk
        for i in range(len(group_df) - Config.SEQUENCE_LENGTH + 1):
            
            # --- Assign to Train or Validation ---
            # We use the group_id to make the split. This ensures
            # ALL sequences from one chunk go to *either* train *or* val.
            # This is a robust way to prevent data leakage.
            if group_id % 100 < (Config.VALIDATION_SPLIT * 100):
                split = 'val'
            else:
                split = 'train'

            # --- Handle Shard Rollover ---
            if seq_counts[split] % Config.SEQUENCES_PER_SHARD == 0:
                if split == 'train' and train_writer: train_writer.close()
                if split == 'val' and val_writer: val_writer.close()
                
                path = os.path.join(output_dir, split, f'data_{shard_counts[split]:04d}.tfrecord')
                if split == 'train':
                    train_writer = tf.io.TFRecordWriter(path)
                else:
                    val_writer = tf.io.TFRecordWriter(path)
                shard_counts[split] += 1
            
            # Extract the 50-sample sequences
            seq_features = features[i : i + Config.SEQUENCE_LENGTH]
            seq_targets = targets[i : i + Config.SEQUENCE_LENGTH]
            
            # Serialize and create the TFExample
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(seq_features)),
                'targets': _bytes_feature(tf.io.serialize_tensor(seq_targets))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Write to the correct file
            if split == 'train':
                train_writer.write(example.SerializeToString())
            else:
                val_writer.write(example.SerializeToString())
            
            seq_counts[split] += 1

    if train_writer: train_writer.close()
    if val_writer: val_writer.close()
    
    print(f"\n   ... Sequencing complete.")
    print(f"   ... Wrote {seq_counts['train']:,} training sequences to {shard_counts['train']} shards.")
    print(f"   ... Wrote {seq_counts['val']:,} validation sequences to {shard_counts['val']} shards.")
    return seq_counts['train'] + seq_counts['val']

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING FINAL CONVERSION (PHASE 2c)")
    print("=============================================================")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load 10Hz CSV
    df = load_data(Config.DATA_CSV_FILE)
    if df is None: return

    # 2. Create 7 Virtual Motors
    df, feature_cols, motor_cols = create_virtual_motors(df)
    
    # 3. Normalize Inputs (Z-Score) and Outputs (Min-Max)
    df = normalize_data(df, feature_cols, motor_cols, Config.NORMALIZATION_PARAMS_FILE, Config.OUTPUT_DIR)

    # 4. Balance the data using Stratified Sampling
    balanced_df = balance_data(df, motor_cols)
    
    # 5. Create Sequences and Save to TFRecord
    total_seqs = write_sequences_to_tfrecord(balanced_df, feature_cols, motor_cols, Config.OUTPUT_DIR)

    # --- FINAL METADATA ---
    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_balanced_10hz_samples": len(balanced_df),
        "total_sequences_written": total_seqs,
        "feature_scaler": Config.FEATURE_SCALER_FILE,
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 FINAL CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {total_seqs:,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("   Scalers saved: feature_scaler.pkl, target_scaler.pkl")
    print("\n   ✅ THE DATASET IS NOW 100% READY FOR TRAINING.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v10.0\convert_to_tfrecord_v2.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v10.0\convert_to_tfrecord_v2.py`

```python
# Save this as convert_to_tfrecord_v2.py
# This is the FINAL data conversion script (Phase 2c).
# It converts the 10Hz CSV into a balanced, normalized, sequential TFRecord dataset.
# *** VERSION 2.1: Includes the float32 cast fix ***

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    DATA_CSV_FILE = "10hz_feature_data.csv"
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    OUTPUT_DIR = "tfrecord_dataset_v2"
    FEATURE_SCALER_FILE = "feature_scaler.pkl"
    TARGET_SCALER_FILE = "target_scaler.pkl" 
    
    NUM_MOVEMENT_CLUSTERS = 6
    SEQUENCE_LENGTH = 50 # 5 seconds
    
    SEQUENCES_PER_SHARD = 500
    VALIDATION_SPLIT = 0.15 

# --- 🛠️ HELPER FUNCTIONS ---

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the 10Hz CSV data."""
    print(f"Loading 10Hz data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        raise FileNotFoundError
    
    df = pd.read_csv(file_path)
    print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
    return df

def create_virtual_motors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Creates the 7 Virtual Motor columns."""
    print("--- 1. Creating 7 Virtual Motors ---")
    
    VIRTUAL_MOTORS: Dict[str, List[str]] = {
        "motor_thumb_flex": [f"glove_{i}" for i in [3, 4]],
        "motor_index_flex": [f"glove_{i}" for i in [6, 7, 8]],
        "motor_middle_flex": [f"glove_{i}" for i in [10, 11, 12]],
        "motor_ring_flex": [f"glove_{i}" for i in [14, 15, 16]],
        "motor_pinky_flex": [f"glove_{i}" for i in [18, 19, 20]],
        "motor_thumb_abduct": ["glove_2"],
        "motor_wrist_flex": ["glove_21"]
    }
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
    
    feature_cols = [f'feat_{i+1}' for i in range(48)]
    return df, feature_cols, motor_cols

def normalize_data(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], norm_params_file: str, out_dir: str) -> pd.DataFrame:
    """
    Applies Min-Max to motors (targets) and Z-Score to features (inputs).
    This is applied to the *entire* 10Hz dataframe.
    """
    print("--- 2. Normalizing All 10Hz Data ---")
    
    # --- A. Normalize Motors (Targets) ---
    print(f"   ... Applying Min-Max (0-1) to {len(m_cols)} motors...")
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    for motor in m_cols:
        p = params[motor]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        if motor_range == 0: motor_range = 1.0 
        
        df[motor] = (df[motor] - motor_min) / motor_range
        df[motor] = np.clip(df[motor], 0.0, 1.0)
    
    joblib.dump(params, os.path.join(out_dir, Config.TARGET_SCALER_FILE))

    # --- B. Normalize Features (Inputs) ---
    print(f"   ... Applying Z-Score to {len(f_cols)} features...")
    feature_data = df[f_cols].values
    feature_scaler = StandardScaler()
    
    # Fit and transform the features
    # Save as float32 to prevent type mismatch error
    features_scaled = feature_scaler.fit_transform(feature_data).astype(np.float32)
    
    joblib.dump(feature_scaler, os.path.join(out_dir, Config.FEATURE_SCALER_FILE))
    
    # Put the normalized features back into the DataFrame
    df[f_cols] = features_scaled
    print("   ... Normalization complete. Scalers saved.")
    return df

def create_all_sequences(df: pd.DataFrame, f_cols: List[str], m_cols: List[str]) -> pd.DataFrame:
    """
    Builds all possible sequences from the full, normalized 10Hz dataframe.
    """
    print("--- 3. Creating All Possible Sequences ---")
    
    all_sequences = [] # This will be a list of dicts
    
    # Group by the 'group_id' to prevent sequences from crossing boundaries
    grouped = df.groupby('group_id')
    
    for group_id, group_df in tqdm(grouped, desc="   ... Processing groups", unit="group"):
        if len(group_df) < Config.SEQUENCE_LENGTH:
            continue # This chunk is too short
            
        # ==================== THE FIX (Alternative Location) ====================
        # We ensure features are float32 before sequencing.
        # This is the one-line fix.
        features = group_df[f_cols].values.astype(np.float32) 
        # ======================================================================
        
        targets = group_df[m_cols].values.astype(np.float32)
        labels = group_df['restimulus'].values
        
        # Slide our 50-sample window over this pure chunk
        for i in range(len(group_df) - Config.SEQUENCE_LENGTH + 1):
            
            seq_label = labels[i + Config.SEQUENCE_LENGTH - 1]
            seq_features = features[i : i + Config.SEQUENCE_LENGTH]
            seq_targets = targets[i : i + Config.SEQUENCE_LENGTH]
            
            all_sequences.append({
                "features": seq_features,
                "targets": seq_targets,
                "label": seq_label
            })

    print(f"   ... Found {len(all_sequences):,} total valid sequences.")
    return pd.DataFrame(all_sequences)

def balance_sequences(seq_df: pd.DataFrame, m_cols: List[str]) -> pd.DataFrame:
    """
    Performs stratified sampling on the SEQUENCES, not the 10Hz data.
    """
    print("--- 4. Balancing Sequences (Stratified Sampling) ---")
    
    rest_seq_df = seq_df[seq_df['label'] == 0].copy()
    move_seq_df = seq_df[seq_df['label'] != 0].copy()
    
    if rest_seq_df.empty or move_seq_df.empty:
        print("   ⚠️ Warning: Missing rest or movement sequences. Balancing may be skewed.")
        return seq_df

    print(f"   ... Found {len(rest_seq_df)} 'rest' sequences and {len(move_seq_df)} 'movement' sequences.")
    
    print(f"   ... Finding {Config.NUM_MOVEMENT_CLUSTERS} movement archetypes (K-Means)...")
    
    avg_poses = np.array([seq.mean(axis=0) for seq in move_seq_df['targets']])
    
    kmeans = KMeans(n_clusters=Config.NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_seq_df['cluster'] = kmeans.fit_predict(avg_poses)
    
    all_balanced_dfs = []
    
    cluster_counts = move_seq_df['cluster'].value_counts()
    
    min_move_count = cluster_counts.min()
    print(f"   ... Minority movement cluster has {min_move_count} sequences.")
    
    for i in range(Config.NUM_MOVEMENT_CLUSTERS):
        cluster_df = move_seq_df[move_seq_df['cluster'] == i]
        all_balanced_dfs.append(cluster_df.sample(n=min_move_count, random_state=42, replace=False))
        
    rest_sample_size = min(len(rest_seq_df), min_move_count)
    print(f"   ... Sampling {rest_sample_size} 'rest' sequences.")
    all_balanced_dfs.append(rest_seq_df.sample(n=rest_sample_size, random_state=42, replace=False))
    
    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} sequences.")
    return balanced_df

def write_sequences_to_tfrecord(seq_df: pd.DataFrame, output_dir: str):
    """
    Saves the final, balanced DataFrame of sequences to TFRecord files.
    """
    print("--- 5. Saving Sequences to TFRecord ---")
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    val_size = int(len(seq_df) * Config.VALIDATION_SPLIT)
    val_df = seq_df.iloc[:val_size]
    train_df = seq_df.iloc[val_size:]
    
    print(f"   ... Splitting into {len(train_df)} train and {len(val_df)} val sequences.")
    
    shard_counts = {'train': 0, 'val': 0}

    # --- Write Training Data ---
    with tqdm(total=len(train_df), desc="   ... Writing train shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(train_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(train_dir, f'data_{shard_counts["train"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['train'] += 1
            
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    # --- Write Validation Data ---
    with tqdm(total=len(val_df), desc="   ... Writing val shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(val_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(val_dir, f'data_{shard_counts["val"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['val'] += 1
                
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    print(f"\n   ... Wrote {len(train_df):,} training sequences to {shard_counts['train']} shards.")
    print(f"   ... Wrote {len(val_df):,} validation sequences to {shard_counts['val']} shards.")
    return len(seq_df)

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING FINAL CONVERSION (PHASE 2c) - V2 (Corrected Order)")
    print("=============================================================")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = load_data(Config.DATA_CSV_FILE)
    if df is None: return

    df, feature_cols, motor_cols = create_virtual_motors(df)
    
    df = normalize_data(df, feature_cols, motor_cols, Config.NORMALIZATION_PARAMS_FILE, Config.OUTPUT_DIR)

    all_sequences_df = create_all_sequences(df, feature_cols, motor_cols)
    
    balanced_seq_df = balance_sequences(all_sequences_df, motor_cols)
    
    total_seqs = write_sequences_to_tfrecord(balanced_seq_df, Config.OUTPUT_DIR)

    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_unbalanced_sequences": len(all_sequences_df),
        "total_balanced_sequences": total_seqs,
        "feature_scaler": Config.FEATURE_SCALER_FILE,
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 FINAL CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {total_seqs:,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("   Scalers saved: feature_scaler.pkl, target_scaler.pkl")
    print("\n   ✅ THE DATASET IS NOW 100% READY FOR TRAINING.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v10.0\train_sequential_v2.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v10.0\train_sequential_v2.py`

```python
# Save this as train_sequential_v2.py
# This script trains our new sequential models (v4, v5)
# on the final TFRecord dataset.

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import glob
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Data ---
    TRAIN_TFRECORD_PATH = "tfrecord_dataset_v2/train/*.tfrecord"
    VAL_TFRECORD_PATH = "tfrecord_dataset_v2/val/*.tfrecord"
    
    # --- Model Hyperparameters (from our data conversion) ---
    SEQUENCE_LENGTH = 50 # 50 steps (5 seconds)
    NUM_FEATURES = 48    # 48 Hudgins' features (12 channels * 4)
    NUM_TARGETS = 7      # 7 Virtual Motors
    
    # --- Training ---
    BATCH_SIZE = 16 # Small dataset, so small batch size
    EPOCHS = 150    # Let's train for a while; EarlyStopping will find the best
    
# --- 1. DATA LOADER ---

def parse_sequence_tfrecord(example_proto):
    """
    Parses the TFRecord examples created by convert_to_tfrecord_v2.py.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'targets': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    features = tf.io.parse_tensor(parsed['features'], out_type=tf.float32)
    targets = tf.io.parse_tensor(parsed['targets'], out_type=tf.float32)
    
    # Set the shapes explicitly
    features = tf.reshape(features, [Config.SEQUENCE_LENGTH, Config.NUM_FEATURES])
    targets = tf.reshape(targets, [Config.SEQUENCE_LENGTH, Config.NUM_TARGETS])
    
    return features, targets

def create_dataset(tfrecord_pattern: str, batch_size: int, shuffle=True):
    """Creates a high-performance TensorFlow dataset from TFRecord files."""
    
    file_list = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    dataset = file_list.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        # Shuffle *before* mapping for better performance
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.map(parse_sequence_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    if shuffle:
        # Repeat indefinitely for training
        dataset = dataset.repeat()
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. MODEL ARCHITECTURES ---

def create_v4_lstm_baseline(model_name="v4_lstm_baseline"):
    """
    Model v4: A simple LSTM-only model. This is our sequential baseline.
    This is a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # LSTM layer that processes the sequence.
    # return_sequences=True is ESSENTIAL for Seq2Seq. It outputs
    # a prediction for *every* of the 50 time steps.
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    
    # A final Dense layer, wrapped in TimeDistributed.
    # This applies the *same* dense layer to *each* of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear') # 'linear' for regression
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse', # Mean Squared Error is standard for regression
        metrics=['mae'] # Mean Absolute Error is easier to interpret
    )
    return model

def create_v5_cnn_lstm(model_name="v5_cnn_lstm"):
    """
    Model v5: The SOTA-informed RCNN (CNN-LSTM) Hybrid.
    This is also a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # 1. CNN Feature Extraction Block
    # 1D Conv layers to find local patterns (motifs) in the 50-step sequence.
    # We use 'causal' padding to prevent the model from "cheating" by
    # looking at future time steps to predict the present.
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(x)
    
    # 2. LSTM Temporal Processing Block
    # The LSTM layer learns the long-term relationships between the
    # features extracted by the CNN.
    # We must use return_sequences=True.
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.4)(x)
    
    # 3. TimeDistributed Prediction Head
    # Applies a Dense layer to each of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear')
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

# --- 3. TRAINING & EVALUATION ---

def setup_experiment_directory(experiment_name):
    """Creates a unique directory for storing all artifacts of a training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def train_model(model_creator, model_name, train_dataset, val_dataset, train_steps, val_steps):
    """The main function to train a single model and save all results."""
    print(f"\n{'='*60}\n🚀 STARTING TRAINING FOR {model_name}\n{'='*60}")
    
    exp_dir = setup_experiment_directory(f"series_{model_name}")
    model = model_creator()
    model.summary()
    
    # Callbacks
    best_model_path = os.path.join(exp_dir, "models", f"best_{model_name}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1), # 25 patience
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
    ]

    print(f"\nTraining {model_name}...")
    start_time = datetime.datetime.now()

    history = model.fit(
        train_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    
    # Load best model and get its validation MAE
    best_val_mae = min(history.history['val_mae'])
    
    # Save results
    results = {
        'model_version': model_name,
        'best_validation_mae': float(best_val_mae),
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_mae']) + 1),
        'training_time_seconds': training_time.total_seconds(),
        'model_architecture': json.loads(model.to_json())
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create and save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE'); plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f'{model_name}_training_history.png'))
    plt.close()

    print(f"\n🎯 {model_name} - FINAL RESULTS:")
    print(f"   Best Validation MAE: {results['best_validation_mae']:.4f}")
    print(f"   Training Time: {str(training_time).split('.')[0]}")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"💾 Results and artifacts saved to: {exp_dir}")
    return results

# --- 4. MAIN EXECUTION ---
def main():
    print("🚀 STARTING SEQUENTIAL MODEL TRAINING (PHASE 2d)")
    print("=============================================================")
    
    # Find the data we just created
    train_files = glob.glob(Config.TRAIN_TFRECORD_PATH)
    val_files = glob.glob(Config.VAL_TFRECORD_PATH)
    
    if not train_files or not val_files:
        print(f"❌ FATAL: Could not find TFRecord files.")
        print(f"   Checked for train: {Config.TRAIN_TFRECORD_PATH}")
        print(f"   Checked for val: {Config.VAL_TFRECORD_PATH}")
        return

    # Get total number of samples from our log file
    # This is a bit of a hack, but it's the easiest way
    # In a real-world scenario, we'd read this from metadata.json
    total_train_seqs = 405
    total_val_seqs = 71
    
    # Calculate steps per epoch
    train_steps = total_train_seqs // Config.BATCH_SIZE
    val_steps = total_val_seqs // Config.BATCH_SIZE
    
    # Ensure steps are at least 1
    if train_steps == 0: train_steps = 1
    if val_steps == 0: val_steps = 1

    print(f"   Found {total_train_seqs} train sequences and {total_val_seqs} val sequences.")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Steps/Epoch: {train_steps}")
    print(f"   Validation Steps/Epoch: {val_steps}")

    # Create the datasets
    train_dataset = create_dataset(Config.TRAIN_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(Config.VAL_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=False)

    # Define models to train
    models_to_train = {
        'v4_lstm_baseline': create_v4_lstm_baseline,
        'v5_cnn_lstm': create_v5_cnn_lstm,
    }
    
    all_results = []
    
    for model_name, model_creator in models_to_train.items():
        results = train_model(
            model_creator, 
            model_name, 
            train_dataset, 
            val_dataset, 
            train_steps, 
            val_steps
        )
        all_results.append(results)

    print(f"\n{'='*70}\n📊 SEQUENTIAL EXPERIMENTS COMPLETE 📊\n{'='*70}")
    
    # Print summary table
    print(f"{'Model':<20} {'Best Val MAE':<15} {'Best Epoch':<12}")
    print('-'*70)
    for res in sorted(all_results, key=lambda x: x['best_validation_mae']):
        print(f"{res['model_version']:<20} {res['best_validation_mae']:.4f}{'':<11} {res['best_epoch']:<12}")
    
    best_model_results = min(all_results, key=lambda x: x['best_validation_mae'])
    print(f"\n🏆 BEST SEQUENTIAL MODEL: {best_model_results['model_version']} (Val MAE: {best_model_results['best_validation_mae']:.4f})")
    print(f"\n   Recall: Our 'Point-in-Time' (v2) model had a Test MAE of 0.1610")
    print(f"   ... We will need to run a final test, but this validation MAE is our new benchmark.")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v11.0\analyze_10hz_data.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v11.0\analyze_10hz_data.py`

```python
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

# --- ⚙️ CONFIGURATION (Defined as Global Constants) ---
DATA_CSV_FILE = "10hz_feature_data.csv"  # CRITICAL: Must be the final CSV file
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
    glove_cols = [f'glove_{i}' for i in range(1, NUM_GLOVE_SENSORS + 1)]  # Use global constant
    
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
    print(f"\n--- 🔬 Generating Figure 3.13 (k={NUM_MOVEMENT_CLUSTERS} Clustering) ---")  # Use global constant
    
    # 1. Apply PCA to reduce from 22 dimensions to 2
    pca = PCA(n_components=2)
    glove_pca = pca.fit_transform(glove_data_scaled)
    
    print(f"   ... PCA Explained Variance (2 components): {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # 2. Apply K-Means clustering to find the archetypal movements
    print(f"   ... Applying K-Means to find {NUM_MOVEMENT_CLUSTERS} movement groups...")  # Use global constant
    kmeans = KMeans(n_clusters=NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_df['cluster'] = kmeans.fit_predict(glove_data_scaled)
    
    # 3. Plot the clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=glove_pca[:, 0], 
        y=glove_pca[:, 1], 
        hue=move_df['cluster'],
        palette=sns.color_palette("hsv", n_colors=NUM_MOVEMENT_CLUSTERS),  # Use global constant
        alpha=0.5, 
        s=10,
        legend='full'
    )
    plt.title(f"Movement Groups (PCA + K-Means Clustering, k={NUM_MOVEMENT_CLUSTERS})", fontsize=16)  # Use global constant
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
    
    # ✅ FIXED: Use global constant directly (NO 'Config' class)
    move_df, glove_data_scaled = load_and_prepare_data(DATA_CSV_FILE) 
    
    if move_df is not None:
        analyze_movement_clusters(move_df, glove_data_scaled)
        
    print("\n" + "=" * 60)
    print("🎉 PLOTTING COMPLETE!")
    print("   Ensure the new 'movement_clusters_pca.png' is used in Figure 3.13.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v11.0\analyze_10hz_data_v0.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v11.0\analyze_10hz_data_v0.py`

```python
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
    
    # A. Primary Flexion/Abduction (7 Motors)
    "motor_thumb_flex": ["glove_3", "glove_4"],
    "motor_index_flex": ["glove_5", "glove_6", "glove_7"],
    "motor_middle_flex": ["glove_8", "glove_9", "glove_10"],
    "motor_ring_flex": ["glove_12", "glove_13", "glove_14"],
    "motor_pinky_flex": ["glove_16", "glove_17", "glove_18"],
    "motor_thumb_abduct": ["glove_2"],
    "motor_wrist_flex_ext": ["glove_21"], # Corrected motor name
    
    # B. Secondary Adduction/Waving (5 Motors - Enhanced)
    "motor_index_adduct": ["glove_11"],
    "motor_middle_adduct": ["glove_15"],
    "motor_ring_adduct": ["glove_19"],
    "motor_pinky_adduct": ["glove_20"],
    "motor_hand_waving": ["glove_1", "glove_22"] # Glove 1 and 22 averaged
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
    print("\n--- 🔬 1. Creating 12 Virtual Motors ---")
    
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
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v11.0\analyze_movement_clusters.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v11.0\analyze_movement_clusters.py`

```python
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
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v11.0\convert_to_10hz_csv.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v11.0\convert_to_10hz_csv.py`

```python
# Save this as convert_to_10hz_csv.py
# This script converts raw 2000Hz CSVs into a single 10Hz feature-engineered CSV
# for Exploratory Data Analysis (EDA), with verbose logging.

import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # We will define the file list in the main() function.
    OUTPUT_CSV_PATH = "10hz_feature_data.csv" # The SINGLE output CSV
    
    # Signal Processing
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # Feature Engineering (Downsampling)
    WINDOW_MS = 100 # 100ms window
    WINDOW_SAMPLES = int(FS * (WINDOW_MS / 1000.0)) # = 200 samples

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def calculate_hudgins_features(window: np.ndarray) -> np.ndarray:
    """
    Calculates the 4 Hudgins' features (RMS, WL, ZC, SSC) for a 12-channel window.
    Input window shape: (200, 12)
    Output feature vector shape: (48,)
    """
    n_samples, n_channels = window.shape
    features = []
    
    # 1. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2, axis=0))
    features.append(rms)
    
    # 2. Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    features.append(wl)
    
    # 3. Zero Crossings (ZC) (with a small threshold for noise)
    zc = np.sum((window[:-1] * window[1:] < 0) & (np.abs(window[:-1] - window[1:]) > 1e-6), axis=0)
    features.append(zc)
    
    # 4. Slope Sign Changes (SSC) (with a small threshold for noise)
    ssc = np.sum((window[1:-1] - window[:-2]) * (window[1:-1] - window[2:]) > 1e-6, axis=0)
    features.append(ssc)
    
    return np.concatenate(features, axis=0).astype(np.float32)

# --- 1. DATA GATHERING (2000Hz -> 10Hz) ---
def build_10hz_list_of_rows(file_list: List[str]) -> List[Dict]:
    """
    Processes all raw CSVs, extracts features, and returns a list of rows
    (as dictionaries) for the 10Hz data.
    """
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_rows = [] # List to hold all our new 10Hz rows (as dicts)
    group_id_counter = 0 # This is the critical ID for preventing sequence breaks
    
    print("Processing Raw Files (2000Hz -> 10Hz)...")
    
    for file_index, fp in enumerate(tqdm(file_list, desc="Overall Progress", unit="file", total=len(file_list))):
        print(f"\nProcessing File {file_index+1}/{len(file_list)}: {os.path.basename(fp)}")
        try:
            # Load all necessary columns
            use_cols = [f"emg_{i}" for i in range(1, 13)] + [f"glove_{i}" for i in range(1, 23)] + ["restimulus"]
            df = pd.read_csv(fp, usecols=lambda c: c in use_cols)
            
            emg_cols = [f"emg_{i}" for i in range(1, 13)]
            glove_cols = [f"glove_{i}" for i in range(1, 23)]
            
            emg_data = df[emg_cols].values
            glove_data = df[glove_cols].values
            labels = df['restimulus'].values.astype(int)
            
            # Filter the raw EMG data
            for i in range(emg_data.shape[1]):
                emg_data[:, i] = signal.filtfilt(b_notch, a_notch, emg_data[:, i])
                emg_data[:, i] = signal.filtfilt(b_band, a_band, emg_data[:, i])
            
            # Find boundaries of "pure" chunks
            boundaries = np.where(np.diff(labels) != 0)[0] + 1
            chunk_starts = np.concatenate(([0], boundaries))
            chunk_stops = np.concatenate((boundaries, [len(labels)]))
            
            print(f"   ... Found {len(chunk_starts)} 'pure' movement/rest chunks.")
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                if start >= stop: continue
                
                chunk_emg = emg_data[start:stop]
                chunk_glove = glove_data[start:stop]
                chunk_label = labels[start] # All labels in this chunk are the same
                
                n_samples = len(chunk_emg)
                # Use non-overlapping windows, discard partial remainder
                n_windows = n_samples // Config.WINDOW_SAMPLES 
                remainder = n_samples % Config.WINDOW_SAMPLES
                
                if n_windows == 0:
                    print(f"   LOG: Discarding chunk (rows {start}-{stop}, label {chunk_label}) - too short ({n_samples} samples) for one window.")
                    continue
                
                # *** YOUR REQUESTED LOGGING ***
                if remainder > 0:
                    print(f"   LOG: Discarding last {remainder} samples from chunk (rows {start}-{stop}, label {chunk_label}) to ensure pure windows.")
                
                # Process all *full* non-overlapping windows in this chunk
                for i in range(n_windows):
                    win_start = start + (i * Config.WINDOW_SAMPLES)
                    win_stop = win_start + Config.WINDOW_SAMPLES
                    
                    # We reference the *original* dataframe slices
                    emg_window = emg_data[win_start:win_stop]
                    glove_window = glove_data[win_start:win_stop]
                    
                    features_48 = calculate_hudgins_features(emg_window)
                    target_22 = glove_window[-1, :].astype(np.float32)
                    
                    # Create a dictionary for this 10Hz row
                    row_data = {
                        'group_id': group_id_counter, # This ID marks the 'pure' chunk
                        'restimulus': chunk_label
                    }
                    # Add features
                    for f_idx in range(48):
                        row_data[f'feat_{f_idx+1}'] = features_48[f_idx] # Use 1-indexing
                    # Add targets
                    for t_idx in range(22):
                        row_data[f'glove_{t_idx+1}'] = target_22[t_idx] # Use 1-indexing
                        
                    all_rows.append(row_data)
                
                group_id_counter += 1 # Increment for each new, pure chunk
                
        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process file {fp}. Error: {e}")
            continue
            
    print(f"\n... 10Hz data extraction complete. Found {len(all_rows)} total 10Hz samples.")
    return all_rows

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING CONVERSION TO 10Hz CSV (PHASE 2a)")
    print("=============================================================")
    
    # ==================== EDIT THIS LIST ====================
    # Paste your file paths here for the test run.
    # Use 'r' before the string to handle Windows backslashes '\'.
    FILES_TO_PROCESS = [
        r"D:/DB\\DB2\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S12_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S13_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S14_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S15_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S16_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S17_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S18_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S19_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S20_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S21_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S22_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S23_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S24_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S25_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S26_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S27_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S28_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S29_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S30_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S31_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S32_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S33_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S34_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S35_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S36_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S37_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S38_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S39_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S40_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB2\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S12_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S13_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S14_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S15_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S16_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S17_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S18_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S19_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S20_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S21_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S22_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S23_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S24_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S25_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S26_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S27_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S28_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S29_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S30_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S31_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S32_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S33_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S34_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S35_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S36_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S37_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S38_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S39_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S40_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S9_E2_A1.csv",
        r"D:/DB\\DB3\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB3\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S9_E2_A1.csv"
    ]
    # ========================================================
    
    print(f"Starting processing for {len(FILES_TO_PROCESS)} manually specified files...")

    # --- STEP 1: Process 2000Hz -> 10Hz ---
    list_of_rows = build_10hz_list_of_rows(FILES_TO_PROCESS)
    
    if not list_of_rows:
        print("❌ FATAL: No 10Hz data could be extracted. Check file paths and data.")
        return

    # --- STEP 2: Convert list of dicts to DataFrame and save ---
    print(f"\n... Converting to DataFrame and saving to {Config.OUTPUT_CSV_PATH}...")
    try:
        final_df = pd.DataFrame(list_of_rows)
        # We will use CSV as you requested, not Parquet.
        final_df.to_csv(Config.OUTPUT_CSV_PATH, index=False)
        print(f"   ... Saved as {Config.OUTPUT_CSV_PATH}.")

    except Exception as e:
        print(f"❌ FATAL: Failed to save DataFrame to CSV. Error: {e}")
        return

    print("\n" + "=" * 60)
    print(f"🎉 CONVERSION TO 10Hz CSV COMPLETE!")
    print(f"   Total 10Hz samples created: {len(list_of_rows):,}")
    print(f"   Output file: {Config.OUTPUT_CSV_PATH}")
    print("\n   ✅ YOU CAN NOW OPEN THIS CSV FILE TO SEE THE DATA.")
    print("\nNext step: Run the 'Phase 2b' analysis script on this new file.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v11.0\convert_to_tfrecord_v2.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v11.0\convert_to_tfrecord_v2.py`

```python
# Save this as convert_to_tfrecord_v2.py
# This is the FINAL data conversion script (Phase 2c).
# It converts the 10Hz CSV into a balanced, normalized, sequential TFRecord dataset.
# *** VERSION 2.1: Includes the float32 cast fix ***

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    DATA_CSV_FILE = "10hz_feature_data.csv"
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    OUTPUT_DIR = "tfrecord_dataset_v2"
    FEATURE_SCALER_FILE = "feature_scaler.pkl"
    TARGET_SCALER_FILE = "target_scaler.pkl" 
    
    # MANDATORY FIX: Explicitly define the correct dimensions
    NUM_TARGETS = 12 # 12 Virtual Motors
    
    NUM_MOVEMENT_CLUSTERS = 12 # 12 movement archetypes
    SEQUENCE_LENGTH = 50 # 5 seconds
    
    SEQUENCES_PER_SHARD = 500
    VALIDATION_SPLIT = 0.15 

# --- 🛠️ HELPER FUNCTIONS ---

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the 10Hz CSV data."""
    print(f"Loading 10Hz data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        raise FileNotFoundError
    
    df = pd.read_csv(file_path)
    print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
    return df

def create_virtual_motors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Creates the 12 Virtual Motor columns based on the report's final grouping."""
    print("--- 1. Creating 12 Virtual Motors (Targets) ---")
    
    # MANDATORY FIX: Updated to 12-motor grouping (must match analyze_10hz_data.py)
    VIRTUAL_MOTORS: Dict[str, List[str]] = {
        # A. Primary Flexion/Abduction (7 Motors)
        "motor_thumb_flex": ["glove_3", "glove_4"],
        "motor_index_flex": ["glove_5", "glove_6", "glove_7"],
        "motor_middle_flex": ["glove_8", "glove_9", "glove_10"],
        "motor_ring_flex": ["glove_12", "glove_13", "glove_14"],
        "motor_pinky_flex": ["glove_16", "glove_17", "glove_18"],
        "motor_thumb_abduct": ["glove_2"],
        "motor_wrist_flex_ext": ["glove_21"], 
        
        # B. Secondary Adduction/Waving (5 Motors - Enhanced)
        "motor_index_adduct": ["glove_11"],
        "motor_middle_adduct": ["glove_15"],
        "motor_ring_adduct": ["glove_19"],
        "motor_pinky_adduct": ["glove_20"],
        "motor_hand_waving": ["glove_1", "glove_22"]
    }
    
    # Check for dimensional mismatch
    if len(VIRTUAL_MOTORS) != Config.NUM_TARGETS:
         print(f"❌ ERROR: Motor count ({len(VIRTUAL_MOTORS)}) does not match Config.NUM_TARGETS ({Config.NUM_TARGETS}).")
         raise ValueError("Dimensional Mismatch in VIRTUAL_MOTORS setup.")
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
    
    feature_cols = [f'feat_{i+1}' for i in range(48)]
    return df, feature_cols, motor_cols

def normalize_data(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], norm_params_file: str, out_dir: str) -> pd.DataFrame:
    """
    Applies Min-Max to motors (targets) and Z-Score to features (inputs).
    This is applied to the *entire* 10Hz dataframe.
    """
    print("--- 2. Normalizing All 10Hz Data ---")
    
    # --- A. Normalize Motors (Targets) ---
    # The print statement is now correct: it prints the actual number of motors being processed
    print(f"   ... Applying Min-Max (0-1) to {len(m_cols)} motors...") 
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    for motor in m_cols:
        p = params[motor]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        if motor_range == 0: motor_range = 1.0 
        
        df[motor] = (df[motor] - motor_min) / motor_range
        df[motor] = np.clip(df[motor], 0.0, 1.0)
    
    joblib.dump(params, os.path.join(out_dir, Config.TARGET_SCALER_FILE))

    # --- B. Normalize Features (Inputs) ---
    print(f"   ... Applying Z-Score to {len(f_cols)} features...")
    feature_data = df[f_cols].values
    feature_scaler = StandardScaler()
    
    # Fit and transform the features
    # Save as float32 to prevent type mismatch error
    features_scaled = feature_scaler.fit_transform(feature_data).astype(np.float32)
    
    joblib.dump(feature_scaler, os.path.join(out_dir, Config.FEATURE_SCALER_FILE))
    
    # Put the normalized features back into the DataFrame
    df[f_cols] = features_scaled
    print("   ... Normalization complete. Scalers saved.")
    return df

def create_all_sequences(df: pd.DataFrame, f_cols: List[str], m_cols: List[str]) -> pd.DataFrame:
    """
    Builds all possible sequences from the full, normalized 10Hz dataframe.
    """
    print("--- 3. Creating All Possible Sequences ---")
    
    all_sequences = [] # This will be a list of dicts
    
    # Group by the 'group_id' to prevent sequences from crossing boundaries
    grouped = df.groupby('group_id')
    
    for group_id, group_df in tqdm(grouped, desc="   ... Processing groups", unit="group"):
        if len(group_df) < Config.SEQUENCE_LENGTH:
            continue # This chunk is too short
            
        # ==================== THE FIX (Alternative Location) ====================
        # We ensure features are float32 before sequencing.
        # This is the one-line fix.
        features = group_df[f_cols].values.astype(np.float32) 
        # ======================================================================
        
        targets = group_df[m_cols].values.astype(np.float32)
        labels = group_df['restimulus'].values
        
        # Slide our 50-sample window over this pure chunk
        for i in range(len(group_df) - Config.SEQUENCE_LENGTH + 1):
            
            seq_label = labels[i + Config.SEQUENCE_LENGTH - 1]
            seq_features = features[i : i + Config.SEQUENCE_LENGTH]
            seq_targets = targets[i : i + Config.SEQUENCE_LENGTH]
            
            all_sequences.append({
                "features": seq_features,
                "targets": seq_targets,
                "label": seq_label
            })

    print(f"   ... Found {len(all_sequences):,} total valid sequences.")
    return pd.DataFrame(all_sequences)

def balance_sequences(seq_df: pd.DataFrame, m_cols: List[str]) -> pd.DataFrame:
    """
    Performs stratified sampling on the SEQUENCES, not the 10Hz data.
    """
    print("--- 4. Balancing Sequences (Stratified Sampling) ---")
    
    rest_seq_df = seq_df[seq_df['label'] == 0].copy()
    move_seq_df = seq_df[seq_df['label'] != 0].copy()
    
    if rest_seq_df.empty or move_seq_df.empty:
        print("   ⚠️ Warning: Missing rest or movement sequences. Balancing may be skewed.")
        return seq_df

    print(f"   ... Found {len(rest_seq_df)} 'rest' sequences and {len(move_seq_df)} 'movement' sequences.")
    
    # The number of movement clusters must match the number of movement motors (12)
    # The stratified sampling will now reflect the 13 classes (12 movements + 1 rest)
    print(f"   ... Finding {Config.NUM_MOVEMENT_CLUSTERS} movement archetypes (K-Means)...") 
    
    avg_poses = np.array([seq.mean(axis=0) for seq in move_seq_df['targets']])
    
    kmeans = KMeans(n_clusters=Config.NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_seq_df['cluster'] = kmeans.fit_predict(avg_poses)
    
    all_balanced_dfs = []
    
    cluster_counts = move_seq_df['cluster'].value_counts()
    
    min_move_count = cluster_counts.min()
    print(f"   ... Minority movement cluster has {min_move_count} sequences.")
    
    for i in range(Config.NUM_MOVEMENT_CLUSTERS):
        cluster_df = move_seq_df[move_seq_df['cluster'] == i]
        all_balanced_dfs.append(cluster_df.sample(n=min_move_count, random_state=42, replace=False))
        
    # The rest class is balanced against the size of the smallest movement cluster
    rest_sample_size = min(len(rest_seq_df), min_move_count) 
    print(f"   ... Sampling {rest_sample_size} 'rest' sequences.")
    all_balanced_dfs.append(rest_seq_df.sample(n=rest_sample_size, random_state=42, replace=False))
    
    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} sequences.")
    return balanced_df

def write_sequences_to_tfrecord(seq_df: pd.DataFrame, output_dir: str):
    """
    Saves the final, balanced DataFrame of sequences to TFRecord files.
    """
    print("--- 5. Saving Sequences to TFRecord ---")
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    val_size = int(len(seq_df) * Config.VALIDATION_SPLIT)
    val_df = seq_df.iloc[:val_size]
    train_df = seq_df.iloc[val_size:]
    
    print(f"   ... Splitting into {len(train_df)} train and {len(val_df)} val sequences.")
    
    shard_counts = {'train': 0, 'val': 0}

    # --- Write Training Data ---
    with tqdm(total=len(train_df), desc="   ... Writing train shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(train_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(train_dir, f'data_{shard_counts["train"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['train'] += 1
            
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    # --- Write Validation Data ---
    with tqdm(total=len(val_df), desc="   ... Writing val shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(val_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(val_dir, f'data_{shard_counts["val"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['val'] += 1
                
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    print(f"\n   ... Wrote {len(train_df):,} training sequences to {shard_counts['train']} shards.")
    print(f"   ... Wrote {len(val_df):,} validation sequences to {shard_counts['val']} shards.")
    return len(seq_df)

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING FINAL CONVERSION (PHASE 2c) - V2 (Corrected Order)")
    print("=============================================================")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = load_data(Config.DATA_CSV_FILE)
    if df is None: return

    df, feature_cols, motor_cols = create_virtual_motors(df)
    
    df = normalize_data(df, feature_cols, motor_cols, Config.NORMALIZATION_PARAMS_FILE, Config.OUTPUT_DIR)

    all_sequences_df = create_all_sequences(df, feature_cols, motor_cols)
    
    balanced_seq_df = balance_sequences(all_sequences_df, motor_cols)
    
    total_seqs = write_sequences_to_tfrecord(balanced_seq_df, Config.OUTPUT_DIR)

    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_unbalanced_sequences": len(all_sequences_df),
        "total_balanced_sequences": total_seqs,
        "feature_scaler": Config.FEATURE_SCALER_FILE,
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 FINAL CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {total_seqs:,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("   Scalers saved: feature_scaler.pkl, target_scaler.pkl")
    print("\n   ✅ THE DATASET IS NOW 100% READY FOR TRAINING.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v11.0\train_sequential_v2.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v11.0\train_sequential_v2.py`

```python
# Save this as train_sequential_v2.py
# This script trains our new sequential models (v4, v5)
# on the final TFRecord dataset.

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import glob
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Data ---
    TRAIN_TFRECORD_PATH = "tfrecord_dataset_v2/train/*.tfrecord"
    VAL_TFRECORD_PATH = "tfrecord_dataset_v2/val/*.tfrecord"
    
    # --- Model Hyperparameters (from our data conversion) ---
    SEQUENCE_LENGTH = 50 # 50 steps (5 seconds)
    NUM_FEATURES = 48    # 48 Hudgins' features (12 channels * 4)
    NUM_TARGETS = 12      # 7 Virtual Motors
    
    # --- Training ---
    BATCH_SIZE = 16 # Small dataset, so small batch size
    EPOCHS = 150    # Let's train for a while; EarlyStopping will find the best
    
# --- 1. DATA LOADER ---

def parse_sequence_tfrecord(example_proto):
    """
    Parses the TFRecord examples created by convert_to_tfrecord_v2.py.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'targets': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    features = tf.io.parse_tensor(parsed['features'], out_type=tf.float32)
    targets = tf.io.parse_tensor(parsed['targets'], out_type=tf.float32)
    
    # Set the shapes explicitly
    features = tf.reshape(features, [Config.SEQUENCE_LENGTH, Config.NUM_FEATURES])
    targets = tf.reshape(targets, [Config.SEQUENCE_LENGTH, Config.NUM_TARGETS])
    
    return features, targets

def create_dataset(tfrecord_pattern: str, batch_size: int, shuffle=True):
    """Creates a high-performance TensorFlow dataset from TFRecord files."""
    
    file_list = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    dataset = file_list.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        # Shuffle *before* mapping for better performance
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.map(parse_sequence_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    if shuffle:
        # Repeat indefinitely for training
        dataset = dataset.repeat()
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. MODEL ARCHITECTURES ---

def create_v4_lstm_baseline(model_name="v4_lstm_baseline"):
    """
    Model v4: A simple LSTM-only model. This is our sequential baseline.
    This is a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # LSTM layer that processes the sequence.
    # return_sequences=True is ESSENTIAL for Seq2Seq. It outputs
    # a prediction for *every* of the 50 time steps.
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    
    # A final Dense layer, wrapped in TimeDistributed.
    # This applies the *same* dense layer to *each* of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear') # 'linear' for regression
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse', # Mean Squared Error is standard for regression
        metrics=['mae'] # Mean Absolute Error is easier to interpret
    )
    return model

def create_v5_cnn_lstm(model_name="v5_cnn_lstm"):
    """
    Model v5: The SOTA-informed RCNN (CNN-LSTM) Hybrid.
    This is also a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # 1. CNN Feature Extraction Block
    # 1D Conv layers to find local patterns (motifs) in the 50-step sequence.
    # We use 'causal' padding to prevent the model from "cheating" by
    # looking at future time steps to predict the present.
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(x)
    
    # 2. LSTM Temporal Processing Block
    # The LSTM layer learns the long-term relationships between the
    # features extracted by the CNN.
    # We must use return_sequences=True.
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.4)(x)
    
    # 3. TimeDistributed Prediction Head
    # Applies a Dense layer to each of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear')
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

# --- 3. TRAINING & EVALUATION ---

def setup_experiment_directory(experiment_name):
    """Creates a unique directory for storing all artifacts of a training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def train_model(model_creator, model_name, train_dataset, val_dataset, train_steps, val_steps):
    """The main function to train a single model and save all results."""
    print(f"\n{'='*60}\n🚀 STARTING TRAINING FOR {model_name}\n{'='*60}")
    
    exp_dir = setup_experiment_directory(f"series_{model_name}")
    model = model_creator()
    model.summary()
    
    # Callbacks
    best_model_path = os.path.join(exp_dir, "models", f"best_{model_name}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1), # 25 patience
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
    ]

    print(f"\nTraining {model_name}...")
    start_time = datetime.datetime.now()

    history = model.fit(
        train_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    
    # Load best model and get its validation MAE
    best_val_mae = min(history.history['val_mae'])
    
    # Save results
    results = {
        'model_version': model_name,
        'best_validation_mae': float(best_val_mae),
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_mae']) + 1),
        'training_time_seconds': training_time.total_seconds(),
        'model_architecture': json.loads(model.to_json())
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create and save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE'); plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f'{model_name}_training_history.png'))
    plt.close()

    print(f"\n🎯 {model_name} - FINAL RESULTS:")
    print(f"   Best Validation MAE: {results['best_validation_mae']:.4f}")
    print(f"   Training Time: {str(training_time).split('.')[0]}")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"💾 Results and artifacts saved to: {exp_dir}")
    return results

# --- 4. MAIN EXECUTION ---
def main():
    print("🚀 STARTING SEQUENTIAL MODEL TRAINING (PHASE 2d)")
    print("=============================================================")
    
    # Find the data we just created
    train_files = glob.glob(Config.TRAIN_TFRECORD_PATH)
    val_files = glob.glob(Config.VAL_TFRECORD_PATH)
    
    if not train_files or not val_files:
        print(f"❌ FATAL: Could not find TFRecord files.")
        print(f"   Checked for train: {Config.TRAIN_TFRECORD_PATH}")
        print(f"   Checked for val: {Config.VAL_TFRECORD_PATH}")
        return

    # Get total number of samples from our log file
    # This is a bit of a hack, but it's the easiest way
    # In a real-world scenario, we'd read this from metadata.json
    total_train_seqs = 405
    total_val_seqs = 71
    
    # Calculate steps per epoch
    train_steps = total_train_seqs // Config.BATCH_SIZE
    val_steps = total_val_seqs // Config.BATCH_SIZE
    
    # Ensure steps are at least 1
    if train_steps == 0: train_steps = 1
    if val_steps == 0: val_steps = 1

    print(f"   Found {total_train_seqs} train sequences and {total_val_seqs} val sequences.")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Steps/Epoch: {train_steps}")
    print(f"   Validation Steps/Epoch: {val_steps}")

    # Create the datasets
    train_dataset = create_dataset(Config.TRAIN_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(Config.VAL_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=False)

    # Define models to train
    models_to_train = {
        'v4_lstm_baseline': create_v4_lstm_baseline,
        'v5_cnn_lstm': create_v5_cnn_lstm,
    }
    
    all_results = []
    
    for model_name, model_creator in models_to_train.items():
        results = train_model(
            model_creator, 
            model_name, 
            train_dataset, 
            val_dataset, 
            train_steps, 
            val_steps
        )
        all_results.append(results)

    print(f"\n{'='*70}\n📊 SEQUENTIAL EXPERIMENTS COMPLETE 📊\n{'='*70}")
    
    # Print summary table
    print(f"{'Model':<20} {'Best Val MAE':<15} {'Best Epoch':<12}")
    print('-'*70)
    for res in sorted(all_results, key=lambda x: x['best_validation_mae']):
        print(f"{res['model_version']:<20} {res['best_validation_mae']:.4f}{'':<11} {res['best_epoch']:<12}")
    
    best_model_results = min(all_results, key=lambda x: x['best_validation_mae'])
    print(f"\n🏆 BEST SEQUENTIAL MODEL: {best_model_results['model_version']} (Val MAE: {best_model_results['best_validation_mae']:.4f})")
    print(f"\n   Recall: Our 'Point-in-Time' (v2) model had a Test MAE of 0.1610")
    print(f"   ... We will need to run a final test, but this validation MAE is our new benchmark.")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v12.0\convert_to_spectrogram.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v12.0\convert_to_spectrogram.py`

```python
# Save this as convert_to_spectrogram.py
# This script converts raw 2000Hz CSVs into a sequential
# dataset of 2D Spectrograms (images) and normalized 7-motor targets.

import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Input Files ---
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    # --- Output ---
    OUTPUT_DIR = "tfrecord_dataset_v3_spectrogram"
    TARGET_SCALER_FILE = "target_scaler.pkl" # We'll re-save our Min/Max scaler
    
    # --- Signal Processing ---
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # --- Sequencing ---
    # We want to match the 5-second movement, as per the dataset guide
    SEQUENCE_S = 5.0
    SEQUENCE_SAMPLES = int(FS * SEQUENCE_S) # 5.0s * 2000 Hz = 10,000 samples
    
    # --- Spectrogram (STFT) Parameters ---
    # We'll use the same 100ms (200-sample) window as our 10Hz features
    STFT_WINDOW_SAMPLES = 200
    # We'll overlap windows by 50% for a smoother image
    STFT_OVERLAP_SAMPLES = 100 
    
    # --- Balancing & TFRecord ---
    SEQUENCES_PER_SHARD = 200 # Spectrograms are large
    VALIDATION_SPLIT = 0.20 # 20% for validation

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def create_virtual_motors(glove_data: np.ndarray) -> np.ndarray:
    """
    Converts a (N, 22) glove array into a (N, 7) motor array.
    """
    # This is our known, fact-based mapping
    VIRTUAL_MOTORS: Dict[str, List[int]] = {
        "motor_thumb_flex": [2, 3], # glove_3, glove_4 (0-indexed)
        "motor_index_flex": [5, 6, 7],
        "motor_middle_flex": [9, 10, 11],
        "motor_ring_flex": [13, 14, 15],
        "motor_pinky_flex": [17, 18, 19],
        "motor_thumb_abduct": [1], # glove_2
        "motor_wrist_flex": [20]  # glove_21
    }
    
    # Create the new 7-motor array
    motor_data = np.zeros((glove_data.shape[0], 7), dtype=np.float32)
    for i, (motor_name, sensor_indices) in enumerate(VIRTUAL_MOTORS.items()):
        motor_data[:, i] = glove_data[:, sensor_indices].mean(axis=1)
        
    return motor_data

def normalize_motors(motor_data: np.ndarray, norm_params_file: str) -> np.ndarray:
    """
    Applies Min-Max (0-1) normalization to the 7 motor columns.
    """
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    normalized_data = np.zeros_like(motor_data)
    
    # This list *must* match the order in VIRTUAL_MOTORS
    motor_names = [
        "motor_thumb_flex", "motor_index_flex", "motor_middle_flex",
        "motor_ring_flex", "motor_pinky_flex", "motor_thumb_abduct", "motor_wrist_flex"
    ]
    
    for i, motor_name in enumerate(motor_names):
        p = params[motor_name]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        if motor_range == 0: motor_range = 1.0 
        
        normalized_data[:, i] = (motor_data[:, i] - motor_min) / motor_range
    
    # Clip to our target 0-1 range
    return np.clip(normalized_data, 0.0, 1.0).astype(np.float32)

def compute_spectrograms(emg_chunk: np.ndarray) -> np.ndarray:
    """
    Converts a (10000, 12) EMG chunk into a (12, 101, 99) spectrogram tensor.
    """
    n_channels = emg_chunk.shape[1]
    all_spectrograms = []
    
    for i in range(n_channels):
        f, t, Sxx = signal.stft(
            emg_chunk[:, i],
            fs=Config.FS,
            nperseg=Config.STFT_WINDOW_SAMPLES,
            noverlap=Config.STFT_OVERLAP_SAMPLES
        )
        # Sxx shape is (n_freqs, n_times), e.g., (101, 99)
        # We use np.abs() to get magnitude and add a small epsilon for log stability
        Sxx_mag = np.abs(Sxx)
        all_spectrograms.append(Sxx_mag)

    # Stack into a (12, 101, 99) tensor
    # We use log-magnitude, which is standard for audio/EMG "images"
    # This compresses the dynamic range
    spectrogram_stack = np.stack(all_spectrograms, axis=0)
    return np.log(spectrogram_stack + 1e-6).astype(np.float32)

def make_tfexample(spectrogram: np.ndarray, target_vector: np.ndarray, label: int) -> tf.train.Example:
    """Serializes a (12, 101, 99) spectrogram and (7,) target vector."""
    
    spectrogram_bytes = tf.io.serialize_tensor(spectrogram).numpy()
    target_bytes = tf.io.serialize_tensor(target_vector).numpy()
    
    feature_dict = {
        'spectrogram': tf.train.Feature(bytes_list=tf.train.BytesList(value=[spectrogram_bytes])),
        'target_vector': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING SPECTROGRAM CONVERSION (PHASE 3a)")
    print("=============================================================")
    
    # ==================== EDIT THIS LIST ====================
    # This should be the FULL list of your raw data files
    FILES_TO_PROCESS = [
        r"D:/DB\\DB2\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S12_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S13_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S14_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S15_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S16_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S17_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S18_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S19_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S20_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S21_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S22_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S23_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S24_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S25_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S26_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S27_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S28_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S29_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S30_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S31_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S32_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S33_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S34_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S35_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S36_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S37_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S38_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S39_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S40_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB2\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S12_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S13_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S14_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S15_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S16_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S17_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S18_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S19_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S20_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S21_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S22_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S23_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S24_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S25_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S26_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S27_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S28_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S29_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S30_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S31_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S32_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S33_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S34_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S35_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S36_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S37_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S38_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S39_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S40_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S9_E2_A1.csv",
        r"D:/DB\\DB3\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB3\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S9_E2_A1.csv"
    ]
    # ========================================================
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_sequences = [] # This will be a list of (spectrogram, target_vector, label)
    
    print(f"Starting processing for {len(FILES_TO_PROCESS)} files...")
    
    for fp in tqdm(FILES_TO_PROCESS, desc="Processing Files", unit="file"):
        try:
            use_cols = [f"emg_{i}" for i in range(1, 13)] + [f"glove_{i}" for i in range(1, 23)] + ["restimulus"]
            df = pd.read_csv(fp, usecols=lambda c: c in use_cols)
            
            emg_cols = [f"emg_{i}" for i in range(1, 13)]
            glove_cols = [f"glove_{i}" for i in range(1, 23)]
            
            emg_data = df[emg_cols].values
            glove_data = df[glove_cols].values
            labels = df['restimulus'].values.astype(int)
            
            # Filter the raw EMG data
            for i in range(emg_data.shape[1]):
                emg_data[:, i] = signal.filtfilt(b_notch, a_notch, emg_data[:, i])
                emg_data[:, i] = signal.filtfilt(b_band, a_band, emg_data[:, i])
            
            # Find boundaries of "pure" chunks
            boundaries = np.where(np.diff(labels) != 0)[0] + 1
            chunk_starts = np.concatenate(([0], boundaries))
            chunk_stops = np.concatenate((boundaries, [len(labels)]))
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                n_samples = stop - start
                chunk_label = labels[start]
                
                # How many 5-second (10,000-sample) sequences can we make from this chunk?
                n_sequences = n_samples // Config.SEQUENCE_SAMPLES
                
                if n_sequences == 0: continue
                    
                # Process all *full* non-overlapping 5-second sequences
                for i in range(n_sequences):
                    seq_start = start + (i * Config.SEQUENCE_SAMPLES)
                    seq_stop = seq_start + Config.SEQUENCE_SAMPLES
                    
                    emg_chunk = emg_data[seq_start:seq_stop]
                    glove_chunk = glove_data[seq_start:seq_stop]
                    
                    # 1. Create the Spectrogram "Image"
                    spectrogram = compute_spectrograms(emg_chunk)
                    
                    # 2. Create the Target Vector
                    # We only care about the hand pose at the *end* of the 5s movement
                    # We'll average the *last 100ms* of glove data for a stable target
                    target_glove_window = glove_chunk[-Config.STFT_WINDOW_SAMPLES:]
                    # Convert (200, 22) -> (22,) by averaging
                    target_glove_vector = target_glove_window.mean(axis=0)
                    # Convert (22,) -> (7,) by grouping
                    target_motor_vector = create_virtual_motors(target_glove_vector.reshape(1, -1))[0]
                    
                    # Add this (input, output, label) pair to our big list
                    all_sequences.append((spectrogram, target_motor_vector, chunk_label))

        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process file {fp}. Error: {e}")
            continue

    print(f"\n... Found {len(all_sequences)} total 5-second sequences.")
    
    # --- 4. Balance the Sequences ---
    print("--- 2. Balancing Sequences (Stratified Sampling) ---")
    
    # Put our sequences into a DataFrame to balance
    seq_df = pd.DataFrame(all_sequences, columns=['spectrogram', 'target_vector', 'label'])
    
    rest_df = seq_df[seq_df['label'] == 0]
    move_df = seq_df[seq_df['label'] != 0]
    
    if rest_df.empty or move_df.empty:
        print("❌ FATAL: Dataset is missing 'rest' or 'movement' sequences.")
        return

    print(f"   ... Found {len(rest_df)} 'rest' sequences and {len(move_df)} 'movement' sequences.")
    
    # Find minority class
    min_move_count = move_df.groupby('label').size().min()
    min_class_count = min(len(rest_df), min_move_count)
    
    print(f"   ... Minority class has {min_class_count} sequences. Balancing to this number.")
    
    all_balanced_dfs = []
    
    # Sample from "Rest"
    all_balanced_dfs.append(rest_df.sample(n=min_class_count, random_state=42))
    
    # Sample from each movement
    movement_labels = move_df['label'].unique()
    for label in movement_labels:
        move_label_df = move_df[move_df['label'] == label]
        all_balanced_dfs.append(move_label_df.sample(n=min_class_count, random_state=42, replace=True)) # Oversample if needed

    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} sequences.")

    # --- 5. Normalize Targets and Save TFRecords ---
    print("--- 3. Normalizing Targets and Saving to TFRecord ---")
    
    # Stack all target vectors for normalization
    target_vectors_stacked = np.stack(balanced_df['target_vector'].values)
    
    # Normalize the 7-motor targets using our JSON file
    normalized_targets = normalize_motors(target_vectors_stacked, Config.NORMALIZATION_PARAMS_FILE)
    
    # Put them back in the DataFrame
    balanced_df['target_vector'] = list(normalized_targets)
    
    # Save the normalization params file we used
    joblib.dump(json.load(open(Config.NORMALIZATION_PARAMS_FILE)), os.path.join(Config.OUTPUT_DIR, Config.TARGET_SCALER_FILE))

    # --- 6. Write TFRecords ---
    train_dir = os.path.join(Config.OUTPUT_DIR, "train")
    val_dir = os.path.join(Config.OUTPUT_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    val_size = int(len(balanced_df) * Config.VALIDATION_SPLIT)
    val_df = balanced_df.iloc[:val_size]
    train_df = balanced_df.iloc[val_size:]
    
    print(f"   ... Splitting into {len(train_df)} train and {len(val_df)} val sequences.")
    
    # --- Write Shards ---
    for split_name, df_split in [('train', train_df), ('val', val_df)]:
        shard_count = 0
        writer = None
        for i, (idx, row) in enumerate(tqdm(df_split.iterrows(), total=len(df_split), desc=f"   ... Writing {split_name} shards")):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(Config.OUTPUT_DIR, split_name, f'data_{shard_count:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_count += 1
            
            example = make_tfexample(row['spectrogram'], row['target_vector'], row['label'])
            writer.write(example.SerializeToString())
        if writer: writer.close()
        print(f"   ... Wrote {len(df_split)} {split_name} sequences to {shard_count} shards.")

    # --- FINAL METADATA ---
    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_balanced_sequences": len(balanced_df),
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 SPECTROGRAM CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {len(balanced_df):,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("\n   ✅ THE DATASET IS NOW READY FOR TRAINING (PHASE 3b).")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v12.0\plot_spectrogram_example.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v12.0\plot_spectrogram_example.py`

```python
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
```

---

## File: PROSTHETIC_AI_CORE\src\experiments\v12.0\train_spectrogram_v3.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\experiments\v12.0\train_spectrogram_v3.py`

```python
# Save this as train_spectrogram_v3.py
# This script trains our new 2D-CNN model (v6)
# on the final spectrogram dataset (tfrecord_dataset_v3_spectrogram).

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import glob
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Data ---
    TRAIN_TFRECORD_PATH = "tfrecord_dataset_v3_spectrogram/train/*.tfrecord"
    VAL_TFRECORD_PATH = "tfrecord_dataset_v3_spectrogram/val/*.tfrecord"
    
    # --- Model Hyperparameters (from our data conversion) ---
    # These shapes are based on the STFT parameters in the conversion script
    # Input shape: (n_freqs, n_times, n_channels)
    # (101, 99, 12)
    INPUT_SHAPE = (101, 101, 12)
    NUM_TARGETS = 7 # 7 Virtual Motors
    
    # --- Training ---
    BATCH_SIZE = 16 
    EPOCHS = 150    
    
    # --- From Log File ---
    TOTAL_TRAIN_SEQS = 1903
    TOTAL_VAL_SEQS = 475

# --- 1. DATA LOADER ---

def parse_spectrogram_tfrecord(example_proto):
    """
    Parses the TFRecord examples created by convert_to_spectrogram.py.
    """
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        'target_vector': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    # Spectrogram was saved as (12, 101, 99) -> (C, H, W)
    spectrogram = tf.io.parse_tensor(parsed['spectrogram'], out_type=tf.float32)
    
    # Target was saved as (7,)
    target = tf.io.parse_tensor(parsed['target_vector'], out_type=tf.float32)
    
    # *** CRITICAL: Transpose the spectrogram ***
    # Keras Conv2D expects (Height, Width, Channels)
    # We convert (C, H, W) -> (H, W, C)
    # (12, 101, 99) -> (101, 99, 12)
    spectrogram = tf.transpose(spectrogram, (1, 2, 0))
    
    # Set the shapes explicitly
    spectrogram = tf.reshape(spectrogram, Config.INPUT_SHAPE)
    target = tf.reshape(target, [Config.NUM_TARGETS])
    
    return spectrogram, target

def create_dataset(tfrecord_pattern: str, batch_size: int, shuffle=True):
    """Creates a high-performance TensorFlow dataset from TFRecord files."""
    
    file_list = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    dataset = file_list.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000) # Buffer size matches our dataset
        
    dataset = dataset.map(parse_spectrogram_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    if shuffle:
        dataset = dataset.repeat() # Repeat indefinitely for training
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. MODEL ARCHITECTURE (v6) ---

def create_v6_cnn_spectrogram(model_name="v6_cnn_spectrogram"):
    """
    Model v6: A 2D-CNN for End-to-End Spectrogram Regression.
    This is a Seq2Vec model: (5-sec image) -> (1 final prediction)
    """
    inputs = layers.Input(shape=Config.INPUT_SHAPE, name="emg_spectrogram_input")
    
    # 1. CNN Feature Extraction Block
    # We treat the (101, 99, 12) input as an "image"
    
    # Initial "stem"
    x = layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 2. Fully-Connected Prediction Head
    # Flatten the 2D features into a 1D vector
    x = layers.Flatten()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Final output layer
    outputs = layers.Dense(Config.NUM_TARGETS, activation='sigmoid', name="motor_output_vector")(x)
    # We use 'sigmoid' because our targets are all normalized to 0-1
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Start with a low LR
        loss='mse', # Mean Squared Error
        metrics=['mae'] # Mean Absolute Error
    )
    return model

# --- 3. TRAINING & EVALUATION ---

def setup_experiment_directory(experiment_name):
    """Creates a unique directory for storing all artifacts of a training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def train_model(model_creator, model_name, train_dataset, val_dataset, train_steps, val_steps):
    """The main function to train a single model and save all results."""
    print(f"\n{'='*60}\n🚀 STARTING TRAINING FOR {model_name}\n{'='*60}")
    
    exp_dir = setup_experiment_directory(f"series_{model_name}")
    model = model_creator()
    model.summary()
    
    # Callbacks
    best_model_path = os.path.join(exp_dir, "models", f"best_{model_name}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
    ]

    print(f"\nTraining {model_name}...")
    start_time = datetime.datetime.now()

    history = model.fit(
        train_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    
    # Load best model and get its validation MAE
    best_val_mae = min(history.history['val_mae'])
    
    # Save results
    results = {
        'model_version': model_name,
        'best_validation_mae': float(best_val_mae),
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_mae']) + 1),
        'training_time_seconds': training_time.total_seconds(),
        'model_architecture': json.loads(model.to_json())
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create and save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE'); plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f'{model_name}_training_history.png'))
    plt.close()

    print(f"\n🎯 {model_name} - FINAL RESULTS:")
    print(f"   Best Validation MAE: {results['best_validation_mae']:.4f}")
    print(f"   Training Time: {str(training_time).split('.')[0]}")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"💾 Results and artifacts saved to: {exp_dir}")
    return results

# --- 4. MAIN EXECUTION ---
def main():
    print("🚀 STARTING SPECTROGRAM MODEL TRAINING (PHASE 3b)")
    print("=============================================================")
    
    # Find the data we just created
    train_files = glob.glob(Config.TRAIN_TFRECORD_PATH)
    val_files = glob.glob(Config.VAL_TFRECORD_PATH)
    
    if not train_files or not val_files:
        print(f"❌ FATAL: Could not find TFRecord files.")
        print(f"   Checked for train: {Config.TRAIN_TFRECORD_PATH}")
        print(f"   Checked for val: {Config.VAL_TFRECORD_PATH}")
        return
    
    # Calculate steps per epoch
    train_steps = Config.TOTAL_TRAIN_SEQS // Config.BATCH_SIZE
    val_steps = Config.TOTAL_VAL_SEQS // Config.BATCH_SIZE
    
    if train_steps == 0: train_steps = 1
    if val_steps == 0: val_steps = 1

    print(f"   Found {Config.TOTAL_TRAIN_SEQS} train sequences and {Config.TOTAL_VAL_SEQS} val sequences.")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Steps/Epoch: {train_steps}")
    print(f"   Validation Steps/Epoch: {val_steps}")

    # Create the datasets
    train_dataset = create_dataset(Config.TRAIN_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(Config.VAL_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=False)

    # Train the single v6 model
    results = train_model(
        create_v6_cnn_spectrogram, 
        "v6_cnn_spectrogram", 
        train_dataset, 
        val_dataset, 
        train_steps, 
        val_steps
    )

    print(f"\n{'='*70}\n📊 END-TO-END EXPERIMENT COMPLETE 📊\n{'='*70}")
    
    print(f"{'Model':<25} {'Best Val MAE':<15}")
    print('-'*70)
    print(f"{results['model_version']:<25} {results['best_validation_mae']:.4f}")
    
    print(f"\n   Recall: Our 'Sequential' (v4) model had a Val MAE of 0.1149")
    
    if results['best_validation_mae'] < 0.1149:
        print(f"\n   🏆🏆🏆 NEW CHAMPION! The End-to-End model is the best one so far! 🏆🏆🏆")
    else:
        print(f"\n   ... The Sequential (v4) model remains the champion.")


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
```

---

## File: PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\blender_export.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\blender_export.py`

```python
# blender_export.py
def export_hand_model():
    """Export the rigged hand model to GLTF format"""
    
    bpy.ops.export_scene.gltf(
        filepath="C:/path/to/your/hand_model.gltf",  # Update this path
        export_format='GLTF_EMBEDDED',
        export_yup=True,
        export_apply=True,
        export_animations=False,
        export_skins=True,
        export_morph=False
    )

export_hand_model()
```

---

## File: PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\blender_hand_mesh.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\blender_hand_mesh.py`

```python
# blender_hand_mesh.py
def create_hand_mesh():
    """Create the hand geometry and skin it to the armature"""
    
    # Create basic hand shape using metaballs or sculpting
    bpy.ops.object.metaball_add(type='BALL', location=(0, 0, 0))
    hand_mesh = bpy.context.object
    hand_mesh.name = "Hand_Mesh"
    
    # Add subsurface modifier for smoothness
    hand_mesh.modifiers.new("Subdivision", type='SUBSURF')
    hand_mesh.modifiers["Subdivision"].levels = 2
    
    return hand_mesh

def rig_hand_to_armature(hand_mesh, armature):
    """Skin the hand mesh to the armature"""
    
    # Add armature modifier
    armature_modifier = hand_mesh.modifiers.new("Armature", type='ARMATURE')
    armature_modifier.object = armature
    
    # Enter weight painting mode (you'd need to paint weights properly)
    bpy.context.view_layer.objects.active = hand_mesh
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    
    return hand_mesh

# Complete the setup
hand_mesh = create_hand_mesh()
rigged_hand = rig_hand_to_armature(hand_mesh, armature)
```

---

## File: PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\blender_hand_setup.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\blender_hand_setup.py`

```python
# blender_hand_setup.py
import bpy
import bmesh
import mathutils

def create_hand_skeleton():
    # Clear existing mesh
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Create armature (skeleton)
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.object
    armature.name = "Hand_Armature"
    
    return armature

def create_bone_hierarchy():
    """Create the complete 22-bone hierarchy for the hand"""
    bones = {}
    
    # Wrist bone (root)
    bpy.ops.armature.bone_primitive_add()
    wrist = bpy.context.active_bone
    wrist.name = "wrist"
    bones['wrist'] = wrist
    
    # Finger bones creation function
    def create_finger(finger_name, parent_bone, positions):
        finger_bones = []
        current_parent = parent_bone
        
        for i, pos in enumerate(positions):
            bpy.ops.armature.bone_primitive_add()
            bone = bpy.context.active_bone
            bone.name = f"{finger_name}_{i}"
            bone.parent = current_parent
            bone.head = current_parent.tail
            bone.tail = pos
            finger_bones.append(bone)
            current_parent = bone
            
        return finger_bones
    
    # Define finger positions (simplified - you'd adjust these)
    finger_positions = {
        'thumb': [
            mathutils.Vector((0.1, 0, 0.1)),    # CMC
            mathutils.Vector((0.2, 0, 0.2)),    # MCP  
            mathutils.Vector((0.3, 0, 0.25)),   # IP
            mathutils.Vector((0.35, 0, 0.3))    # Tip
        ],
        'index': [
            mathutils.Vector((0.1, 0.05, 0)),   # MCP
            mathutils.Vector((0.2, 0.05, 0)),   # PIP
            mathutils.Vector((0.3, 0.05, 0)),   # DIP
            mathutils.Vector((0.35, 0.05, 0))   # Tip
        ]
        # Add middle, ring, pinky similarly...
    }
    
    # Create all fingers
    bones['thumb'] = create_finger('thumb', bones['wrist'], finger_positions['thumb'])
    bones['index'] = create_finger('index', bones['wrist'], finger_positions['index'])
    # Add other fingers...
    
    return bones

# Run the setup
armature = create_hand_skeleton()
bones = create_bone_hierarchy()
```

---

## File: PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\enhanced_server.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\enhanced_server.py`

```python
# enhanced_server.py
import numpy as np

class BiomechanicalConstraints:
    """Apply biomechanical constraints to joint angles"""
    
    @staticmethod
    def apply_finger_constraints(glove_values):
        """Ensure finger movements are biomechanically realistic"""
        constrained_values = glove_values.copy()
        
        # Thumb constraints
        constrained_values[0] = np.clip(glove_values[0], 0, 0.8)  # CMC flexion
        constrained_values[1] = np.clip(glove_values[1], 0, 0.6)  # CMC abduction
        
        # Finger coupling constraints (PIP/DIP relationship)
        for finger_start in [4, 8, 12, 16]:  # MCP indices for each finger
            pip_index = finger_start + 2
            dip_index = finger_start + 3
            
            # Ensure DIP flexion doesn't exceed PIP flexion
            if constrained_values[dip_index] > constrained_values[pip_index]:
                constrained_values[dip_index] = constrained_values[pip_index] * 0.8
        
        return constrained_values

# Enhanced WebSocket endpoint with constraints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    biomechanical = BiomechanicalConstraints()
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "next_frame":
                # ... existing data loading code ...
                
                # Apply biomechanical constraints
                constrained_truth = biomechanical.apply_finger_constraints(ground_truth_glove_values)
                constrained_pred = biomechanical.apply_finger_constraints(predicted_glove_values)
                
                payload = {
                    "ground_truth": constrained_truth.tolist(),
                    "prediction": constrained_pred.tolist(),
                    "metrics": {
                        "mse": mse,
                        "constrained_mse": calculate_mse(constrained_truth, constrained_pred)
                    }
                }
                
                await websocket.send_text(json.dumps(payload))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## File: PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\backend\server.py
**Path:** `C:\stage\stage\PROSTHETIC_AI_CORE\src\visualization\prosthetic-hand-visualizer\backend\server.py`

```python
# backend/server.py
import asyncio
import websockets
import json
import random  # We'll use fake data first

async def send_hand_data(websocket):
    print("Client connected")
    try:
        while True:
            # Generate fake hand data for testing
            ground_truth = [random.random() for _ in range(22)]  # 22 random values 0-1
            prediction = [random.random() for _ in range(22)]    # 22 random values 0-1
            
            data = {
                "ground_truth": ground_truth,
                "prediction": prediction,
                "metrics": {"mse": random.random() * 0.1}
            }
            
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.1)  # Send 10 times per second
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(send_hand_data, "localhost", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
```

---

## File: rapport graphs\ofgjeofg.py
**Path:** `C:\stage\stage\rapport graphs\ofgjeofg.py`

```python

```

---

## File: rapport graphs\phase2_plot.py
**Path:** `C:\stage\stage\rapport graphs\phase2_plot.py`

```python
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
```

---

## File: rapport graphs\plot_filtered_signal.py
**Path:** `C:\stage\stage\rapport graphs\plot_filtered_signal.py`

```python
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
```

---

## File: rapport graphs\plot_filtered_signal1.py
**Path:** `C:\stage\stage\rapport graphs\plot_filtered_signal1.py`

```python
# Save this as plot_notch_filter_effect.py
# This script specifically isolates and visualizes the effect of *only*
# the 50Hz notch filter on a small slice of the raw signal.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv" # Use the same file path as before

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

# --- ⚙️ TIME SLICE TO PLOT (ZOOMED IN) ---
# We'll find a "noisy" section, often during rest
START_SAMPLE = 30000 
# We'll plot a tiny 0.2-second slice (400 samples)
DURATION_SAMPLES = 400 

# --- ⚙️ FILTER PARAMETERS ---
FS = 2000.0 # 2000 Hz
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
# We are *NOT* using the bandpass filter in this script

# --- 🛠️ HELPER FUNCTION: Filter Design ---
def design_notch_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Creates *only* the 50Hz notch filter."""
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs)
    return b_notch, a_notch

# --- Main Plotting Function ---
def plot_filter_comparison(file_path):
    print(f"Loading raw data for {CHANNEL_TO_PLOT} from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the single channel we want to inspect
        df = pd.read_csv(file_path, usecols=[CHANNEL_TO_PLOT])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal
    signal_raw = df[CHANNEL_TO_PLOT].values
    
    # 3. Create the filtered signal (NOTCH ONLY)
    print(f"   ... Applying ONLY the 50Hz notch filter...")
    b_notch, a_notch = design_notch_filter(FS)
    signal_notched = signal.filtfilt(b_notch, a_notch, signal_raw)
    print("   ... Filtering complete.")

    # 4. Get the *slice* of both signals
    stop_sample = START_SAMPLE + DURATION_SAMPLES
    raw_slice = signal_raw[START_SAMPLE:stop_sample]
    notched_slice = signal_notched[START_SAMPLE:stop_sample]
    
    # Create a time axis in milliseconds
    time_axis_ms = (np.arange(len(raw_slice)) / FS) * 1000.0 

    # 5. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # --- Plot 1: Raw Signal (should show 50Hz "fuzz") ---
    ax1.plot(time_axis_ms, raw_slice, label="Raw Signal", color='r', alpha=0.9)
    ax1.set_title(f"BEFORE: Raw {CHANNEL_TO_PLOT} (Note the 50Hz 'fuzz')", fontsize=14)
    ax1.set_ylabel("Raw EMG Value")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Notched Signal (should be smooth) ---
    ax2.plot(time_axis_ms, notched_slice, label="Notch Filtered Signal", color='b')
    ax2.set_title(f"AFTER: {CHANNEL_TO_PLOT} with 50Hz Hum Removed", fontsize=14)
    ax2.set_ylabel("Filtered EMG Value")
    ax2.set_xlabel(f"Time (milliseconds) - A {DURATION_SAMPLES/FS:.2f} second slice", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = "notch_filter_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... This plot *only* shows the removal of the 50Hz hum.")

if __name__ == "__main__":
    plot_filter_comparison(RAW_FILE_TO_INSPECT)
```

---

## File: rapport graphs\plot_psd_effect.py
**Path:** `C:\stage\stage\rapport graphs\plot_psd_effect.py`

```python
# Save this as plot_psd_effect.py
# This script visualizes the *frequency domain* (PSD)
# to show the effect of the filters.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

# --- ⚙️ FILTER PARAMETERS ---
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
def plot_psd_comparison(file_path):
    print(f"Loading raw data for {CHANNEL_TO_PLOT} from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the single channel we want to inspect
        df = pd.read_csv(file_path, usecols=[CHANNEL_TO_PLOT, 'restimulus'])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal *during movement* (more signal to see)
    signal_raw = df[df['restimulus'] != 0][CHANNEL_TO_PLOT].values
    if len(signal_raw) == 0:
        print("   ⚠️ Warning: No movement data found, using 'rest' data.")
        signal_raw = df[CHANNEL_TO_PLOT].values

    # 3. Create the filtered signal
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    signal_filtered = signal.filtfilt(b_notch, a_notch, signal_raw)
    signal_filtered = signal.filtfilt(b_band, a_band, signal_filtered)
    print("   ... Filtering complete.")

    # 4. Calculate the Power Spectral Density (PSD) for both signals
    # This is the "frequency graph" you were asking for.
    print("   ... Calculating Power Spectral Density (PSD)...")
    # Using signal.welch() is a standard method to get a clean PSD
    f_raw, Pxx_raw = signal.welch(signal_raw, FS, nperseg=2048)
    f_filtered, Pxx_filtered = signal.welch(signal_filtered, FS, nperseg=2048)

    # 5. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.suptitle("Filter Effect in the Frequency Domain (Power Spectral Density)", fontsize=16)
    
    # --- Plot 1: Raw Signal PSD ---
    ax1.semilogy(f_raw, Pxx_raw, label="Raw Signal", color='r', alpha=0.8)
    ax1.set_title(f"BEFORE: Raw {CHANNEL_TO_PLOT} (Note the 50Hz spike)")
    ax1.set_ylabel("Power / Frequency (dB/Hz) - Log Scale")
    ax1.axvline(NOTCH_FREQ, color='k', linestyle='--', label="50Hz Hum")
    ax1.axvspan(0, BAND_LOW, color='gray', alpha=0.3, label="Low-freq Noise")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Filtered Signal PSD ---
    ax2.semilogy(f_filtered, Pxx_filtered, label="Filtered Signal", color='b')
    ax2.set_title(f"AFTER: {CHANNEL_TO_PLOT} (Noise successfully removed)")
    ax2.set_ylabel("Power / Frequency (dB/Hz) - Log Scale")
    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.axvline(NOTCH_FREQ, color='k', linestyle='--', label="50Hz Hum (Removed)")
    ax2.axvspan(0, BAND_LOW, color='gray', alpha=0.3, label="Low-freq Noise (Removed)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Limit x-axis to 0-500Hz to see the details
    ax2.set_xlim(0, 500)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "psd_filter_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... This plot *proves* the filters are working. Look for the 50Hz spike in the top plot and see it's gone in the bottom plot.")

if __name__ == "__main__":
    plot_psd_comparison(RAW_FILE_TO_INSPECT)
```

---

## File: rapport graphs\plot_raw_glove_targets.py
**Path:** `C:\stage\stage\rapport graphs\plot_raw_glove_targets.py`

```python
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
```

---

## File: rapport graphs\plot_raw_signal.py
**Path:** `C:\stage\stage\rapport graphs\plot_raw_signal.py`

```python
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
```

---

## File: rapport graphs\plot_rectification_effect.py
**Path:** `C:\stage\stage\rapport graphs\plot_rectification_effect.py`

```python
# Save this as plot_rectification_effect.py
# This script visualizes the effect of Rectification (absolute value)
# on a filtered and Z-scored signal.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The single channel we want to inspect
CHANNEL_TO_PLOT = "emg_1"

# --- ⚙️ TIME SLICE TO PLOT (ZOOMED IN) ---
# We'll find a "noisy" section, often during rest
START_SAMPLE = 30000 
# We'll plot a 0.5-second slice (1000 samples) to see the wave clearly
DURATION_SAMPLES = 1000 

# --- ⚙️ FILTER PARAMETERS ---
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
def plot_rectification_comparison(file_path):
    print(f"Loading raw data for {CHANNEL_TO_PLOT} from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load only the single channel we want to inspect
        df = pd.read_csv(file_path, usecols=[CHANNEL_TO_PLOT])
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal
    signal_raw = df[CHANNEL_TO_PLOT].values.reshape(-1, 1) # Reshape for scaler
    
    # 3. Create the filtered signal
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    
    signal_filtered = signal.filtfilt(b_notch, a_notch, signal_raw, axis=0)
    signal_filtered = signal.filtfilt(b_band, a_band, signal_filtered, axis=0)

    # 4. Create the Z-scored signal ("BEFORE" data)
    print("   ... Applying Z-Score (StandardScaler)...")
    scaler = StandardScaler()
    signal_zscored = scaler.fit_transform(signal_filtered)
    print("   ... Z-Score complete.")
    
    # 5. Create the Rectified signal ("AFTER" data)
    print("   ... Applying Rectification (Absolute Value)...")
    signal_rectified = np.abs(signal_zscored)

    # 6. Get the *slice* of both signals for plotting
    stop_sample = START_SAMPLE + DURATION_SAMPLES
    zscored_slice = signal_zscored[START_SAMPLE:stop_sample]
    rectified_slice = signal_rectified[START_SAMPLE:stop_sample]
    
    # Create a time axis in milliseconds
    time_axis_ms = (np.arange(len(zscored_slice)) / FS) * 1000.0 

    # 7. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.suptitle("Effect of Rectification (Absolute Value)", fontsize=16)
    
    # --- Plot 1: Z-Scored Signal (Before) ---
    ax1.plot(time_axis_ms, zscored_slice, label="Z-Scored Signal", color='b')
    ax1.axhline(0, color='k', linestyle='--', label="Zero line")
    ax1.set_title(f"BEFORE: Z-Scored {CHANNEL_TO_PLOT} (Positive and Negative Values)", fontsize=14)
    ax1.set_ylabel("Normalized Value (Std. Dev)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Rectified Signal (After) ---
    ax2.plot(time_axis_ms, rectified_slice, label="Rectified Signal", color='g')
    ax2.axhline(0, color='k', linestyle='--', label="Zero line")
    ax2.set_title(f"AFTER: Rectified {CHANNEL_TO_PLOT} (All Non-Negative)", fontsize=14)
    ax2.set_ylabel("Normalized Value (Absolute)")
    ax2.set_xlabel(f"Time (milliseconds) - A {DURATION_SAMPLES/FS:.2f} second slice", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "rectification_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... The top plot shows the wave centered at 0. The bottom plot shows all negative parts 'flipped' to positive.")

if __name__ == "__main__":
    plot_rectification_comparison(RAW_FILE_TO_INSPECT)
```

---

## File: rapport graphs\plot_zscore_effect.py
**Path:** `C:\stage\stage\rapport graphs\plot_zscore_effect.py`

```python
# Save this as plot_zscore_effect.py
# This script visualizes the statistical distribution of the 12 EMG channels
# *before* and *after* Z-Score normalization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# --- ⚙️ CONFIGURATION ---
# TODO: Paste the *full path* to ONE of your raw data files here
RAW_FILE_TO_INSPECT = r"D:\DB\DB2\E1\S1_E1_A1.csv"

# The 12 channels we will analyze
EMG_COLS = [f"emg_{i}" for i in range(1, 13)]

# We'll use a large chunk of data for a good statistical sample
# Let's take 50,000 samples (25 seconds) from a movement section
START_SAMPLE = 30000 
SAMPLE_COUNT = 50000

# --- ⚙️ FILTER PARAMETERS ---
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
def plot_zscore_comparison(file_path):
    print(f"Loading raw data from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return

    try:
        # 1. Load *only* the EMG data
        df = pd.read_csv(file_path, usecols=EMG_COLS)
        print(f"   ... Loaded {len(df):,} samples.")
    except Exception as e:
        print(f"❌ FATAL: Could not read CSV file. Error: {e}")
        return

    # 2. Get the full raw signal
    signal_raw = df.values
    
    # 3. Create the filtered signal ("BEFORE" data)
    print("   ... Applying 50Hz Notch and 20-450Hz Bandpass filters...")
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    
    signal_filtered = np.zeros_like(signal_raw)
    for i in range(signal_raw.shape[1]): # Filter each channel
        signal_filtered[:, i] = signal.filtfilt(b_notch, a_notch, signal_raw[:, i])
        signal_filtered[:, i] = signal.filtfilt(b_band, a_band, signal_filtered[:, i])
    print("   ... Filtering complete.")

    # 4. Create the Z-scored signal ("AFTER" data)
    print("   ... Applying Z-Score (StandardScaler)...")
    scaler = StandardScaler()
    signal_zscored = scaler.fit_transform(signal_filtered)
    print("   ... Z-Score complete.")

    # 5. Get the *slice* of both signals for plotting
    stop_sample = START_SAMPLE + SAMPLE_COUNT
    filtered_slice = signal_filtered[START_SAMPLE:stop_sample]
    zscored_slice = signal_zscored[START_SAMPLE:stop_sample]
    
    # 6. Create the "Before vs. After" plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.suptitle("Effect of Z-Score Normalization on EMG Channel Distributions", fontsize=16)
    
    # --- Plot 1: BEFORE Z-Score ---
    ax1.set_title("BEFORE: Filtered Data (12 Channels)")
    ax1.set_xlabel("Signal Value (Amplitude)")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)
    # This will plot 12 *different* density curves
    for i in range(filtered_slice.shape[1]):
        sns.kdeplot(filtered_slice[:, i], ax=ax1, label=f'emg_{i+1}', warn_singular=False)
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    # This plot should look "messy" with 12 different peaks and widths

    # --- Plot 2: AFTER Z-Score ---
    ax2.set_title("AFTER: Z-Scored Data (12 Channels)")
    ax2.set_xlabel("Normalized Value (Standard Deviations)")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)
    # This will plot 12 *identical* density curves
    for i in range(zscored_slice.shape[1]):
        sns.kdeplot(zscored_slice[:, i], ax=ax2, warn_singular=False)
    # This plot should show one single, clean bell curve
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    output_file = "zscore_effect_visualization.png" # <-- New filename
    plt.savefig(output_file, dpi=150)
    print(f"\n🎉 Success! Plot saved to {output_file}")
    print("   ... The 'BEFORE' plot shows 12 different distributions.")
    print("   ... The 'AFTER' plot should show one single, centered distribution (a perfect bell curve).")

if __name__ == "__main__":
    sns.set_theme() # Use Seaborn's nice default styling
    plot_zscore_comparison(RAW_FILE_TO_INSPECT)
```

---

## File: rapport graphs\test.py
**Path:** `C:\stage\stage\rapport graphs\test.py`

```python

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## File: v1.0\analyze_predictions.py
**Path:** `C:\stage\stage\v1.0\analyze_predictions.py`

```python
import matplotlib.pyplot as plt
import numpy as np
from data_loader import DB2DataLoader
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('lstm_model.keras')

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# Create sequence dataset
X, y = data_loader.create_sequence_dataset(subject=1, exercise=1, sequence_length=10, target_shift=1)

# Make predictions
y_pred = model.predict(X)

# Select a few representative sensors to plot
sensors_to_plot = [0, 5, 10, 15, 20]  # First, middle, and last sensors
n_plot = 500  # Number of samples to plot

plt.figure(figsize=(15, 10))
for i, sensor_idx in enumerate(sensors_to_plot):
    plt.subplot(len(sensors_to_plot), 1, i+1)
    plt.plot(y[:n_plot, sensor_idx], label='Actual', alpha=0.7)
    plt.plot(y_pred[:n_plot, sensor_idx], label='Predicted', alpha=0.7)
    plt.ylabel(f'Sensor {sensor_idx+1}')
    plt.legend()
    if i == 0:
        plt.title('Actual vs Predicted Joint Angles')

plt.xlabel('Time Step')
plt.tight_layout()
plt.savefig('prediction_analysis.png')
plt.show()

# Calculate R² per sensor
from sklearn.metrics import r2_score
r2_scores = []
for i in range(y.shape[1]):
    r2_scores.append(r2_score(y[:, i], y_pred[:, i]))

plt.figure(figsize=(10, 5))
plt.bar(range(1, 23), r2_scores)
plt.xlabel('Sensor Index')
plt.ylabel('R² Score')
plt.title('R² Score per Glove Sensor')
plt.axhline(y=np.mean(r2_scores), color='r', linestyle='--', label=f'Mean R²: {np.mean(r2_scores):.4f}')
plt.legend()
plt.savefig('r2_per_sensor.png')
plt.show()

print(f"Mean R² across all sensors: {np.mean(r2_scores):.4f}")
print(f"Std R² across all sensors: {np.std(r2_scores):.4f}")
```

---

## File: v1.0\data_loader.py
**Path:** `C:\stage\stage\v1.0\data_loader.py`

```python
import pandas as pd
import numpy as np
from scipy import signal
import os
from sklearn.decomposition import NMF
from tensorflow.keras.utils import Sequence

class DB2DataLoader:
    def __init__(self, base_path, subject_list, exercises=[1]):
        self.base_path = base_path
        self.subject_list = subject_list
        self.exercises = exercises
        self.nmf_model = None

    def load_subject_exercise(self, subject, exercise):
        """
        Loads a specific subject's exercise data from a CSV file.
        
        Args:
            subject (int): Subject ID (e.g., 1, 2, 3...)
            exercise (int): Exercise number (1, 2, or 3)
        
        Returns:
            pandas.DataFrame: The loaded data.
        """
        # Construct the file name based on your files
        file_name = f"S{subject}_E{exercise}_A1.csv"
        file_path = os.path.join(self.base_path, f"E{exercise}", file_name)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find the data file: {file_path}")
        
        # Load the CSV. This might take a few seconds because it's a large file.
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        print("Loading complete.")
        return df

    def get_processed_data(self, subject, exercise, filter=True, normalize=True):
        """
        Loads and preprocesses the data for a subject and exercise.
        Applies filtering and normalization to the EMG signals.

        Args:
            subject (int): Subject ID
            exercise (int): Exercise number
            filter (bool): Whether to apply filtering
            normalize (bool): Whether to apply normalization

        Returns:
            dict: A dictionary with processed EMG, glove data, and labels.
        """
        # Load the raw data
        df = self.load_subject_exercise(subject, exercise)
        
        # Extract the relevant data columns
        emg_data = df[[f'emg_{i}' for i in range(1, 13)]].values  # Shape: (n_samples, 12)
        glove_data = df[[f'glove_{i}' for i in range(1, 23)]].values  # Shape: (n_samples, 22)
        restimulus = df['restimulus'].values  # Shape: (n_samples,)
        
        # Apply preprocessing to EMG data
        if filter:
            emg_data = self._apply_filters(emg_data)
        if normalize:
            emg_data = self._apply_normalization(emg_data, restimulus)
        
        return {
            'emg': emg_data,
            'glove': glove_data,
            'labels': restimulus
        }
    
    def _apply_filters(self, emg_data):
        """Apply notch and bandpass filters to the EMG data."""
        sampling_rate = 2000  # Hz
        # Notch filter for 60 Hz interference
        notch_freq = 60.0
        Q = 30.0
        b_notch, a_notch = signal.iirnotch(notch_freq, Q, sampling_rate)
        # Bandpass filter (20-450 Hz)
        lowcut = 20.0
        highcut = 450.0
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b_band, a_band = signal.butter(4, [low, high], btype='band')
        
        # Apply filters using zero-phase filtering (filtfilt)
        filtered_data = np.zeros_like(emg_data)
        for i in range(emg_data.shape[1]):
            # Notch filter
            channel_data = emg_data[:, i]
            filtered_channel = signal.filtfilt(b_notch, a_notch, channel_data)
            # Bandpass filter
            filtered_channel = signal.filtfilt(b_band, a_band, filtered_channel)
            filtered_data[:, i] = filtered_channel
        
        return filtered_data
    
    def _apply_normalization(self, emg_data, restimulus):
        """Normalize EMG data based on rest periods."""
        # Find rest periods (where restimulus == 0)
        rest_indices = np.where(restimulus == 0)[0]
        rest_data = emg_data[rest_indices, :]
        
        # Calculate mean and std from rest data
        rest_mean = np.mean(rest_data, axis=0)
        rest_std = np.std(rest_data, axis=0)
        
        # Avoid division by zero; set minimum std to a small value
        rest_std[rest_std < 1e-6] = 1e-6
        
        # Apply z-score normalization
        normalized_data = (emg_data - rest_mean) / rest_std
        
        return normalized_data

    def create_regression_dataset(self, subject, exercise, window_size=500, overlap=250, normalize_targets=True, use_nmf=False, n_components=4):
        """
        Creates a dataset for regression by windowing the data and extracting features.
        
        Args:
            subject (int): Subject ID
            exercise (int): Exercise number
            window_size (int): Number of samples per window (500 = 250ms at 2000Hz)
            overlap (int): Number of samples to overlap between windows (250 = 125ms)
            normalize_targets (bool): Whether to normalize the target values to [0,1]
            use_nmf (bool): Whether to use NMF features instead of RMS/MAV
            n_components (int): Number of NMF components if use_nmf is True
            
        Returns:
            tuple: (X, y) where:
                X: Feature matrix (n_windows, n_features)
                y: Target matrix (n_windows, 22) - glove values at the end of each window
        """
        # Get the processed data
        data = self.get_processed_data(subject, exercise)
        emg_data = data['emg']
        glove_data = data['glove']
        
        # If using NMF, fit the NMF model on the entire EMG data
        if use_nmf:
            # For NMF, we need non-negative data; use absolute value
            emg_non_negative = np.abs(emg_data)
            self.nmf_model = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
            self.nmf_model.fit(emg_non_negative)  # Fit on entire data
            # We will transform each window later
            n_features = n_components
        else:
            n_features = 24  # 12 RMS + 12 MAV features
        
        # Calculate number of windows
        n_samples = emg_data.shape[0]
        step_size = window_size - overlap
        n_windows = (n_samples - window_size) // step_size + 1
        
        # Initialize arrays for features and targets
        X = np.zeros((n_windows, n_features))
        y = np.zeros((n_windows, 22))
        
        # Extract features for each window
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            # Extract EMG window
            emg_window = emg_data[start_idx:end_idx, :]
            
            if use_nmf:
                # For NMF, transform the window using the pre-trained NMF model
                emg_window_non_negative = np.abs(emg_window)
                W_window = self.nmf_model.transform(emg_window_non_negative)  # Shape: (window_size, n_components)
                # We need to reduce the window to a feature vector: take the mean activation per synergy
                features = np.mean(W_window, axis=0)
                X[i, :] = features
            else:
                # Calculate traditional features
                rms_features = np.sqrt(np.mean(emg_window**2, axis=0))
                mav_features = np.mean(np.abs(emg_window), axis=0)
                # Combine features
                X[i, :] = np.concatenate([rms_features, mav_features])
            
            # Use the glove data at the END of the window as target
            y[i, :] = glove_data[end_idx - 1, :]
        
        # Normalize targets if requested
        if normalize_targets:
            min_vals = np.min(glove_data, axis=0)
            max_vals = np.max(glove_data, axis=0)
            range_vals = max_vals - min_vals
            # Avoid division by zero
            range_vals[range_vals == 0] = 1
            y = (y - min_vals) / range_vals
        
        return X, y

    def create_sequence_dataset(self, subject, exercise, sequence_length=10, window_size=500, overlap=250, normalize_targets=True, use_nmf=False, n_components=4, target_shift=0):
            """
            Creates a dataset for sequence-based models (e.g., LSTM) by grouping consecutive windows into sequences.
            
            Args:
                subject (int): Subject ID
                exercise (int): Exercise number
                sequence_length (int): Number of windows in each sequence
                window_size (int): Number of samples per window
                overlap (int): Number of samples to overlap between windows
                normalize_targets (bool): Whether to normalize the target values to [0,1]
                use_nmf (bool): Whether to use NMF features instead of RMS/MAV
                n_components (int): Number of NMF components if use_nmf is True
                target_shift (int): Number of windows to shift the target into the future. For example, if 1, the target for a sequence ending at window t will be the target at window t+1.
                
            Returns:
                tuple: (X, y) where:
                    X: Sequence feature tensor (n_sequences, sequence_length, n_features)
                    y: Target matrix (n_sequences, 22) - glove values at the shifted window
            """
            # First, create the regression dataset without sequences
            X_windows, y_windows = self.create_regression_dataset(subject, exercise, window_size, overlap, normalize_targets, use_nmf, n_components)
            
            n_windows = X_windows.shape[0]
            n_features = X_windows.shape[1]
            
            # Calculate number of sequences, accounting for target_shift
            n_sequences = n_windows - sequence_length - target_shift + 1
            
            if n_sequences <= 0:
                raise ValueError("Not enough windows to create sequences with the given sequence_length and target_shift")
            
            # Initialize arrays for sequences and targets
            X_sequences = np.zeros((n_sequences, sequence_length, n_features))
            y_sequences = np.zeros((n_sequences, 22))
            
            # Create sequences
            for i in range(n_sequences):
                X_sequences[i] = X_windows[i:i+sequence_length, :]
                y_sequences[i] = y_windows[i+sequence_length-1 + target_shift]  # shift the target
            
            return X_sequences, y_sequences

    # Add this method to your DB2DataLoader class
    def get_subject_specific_normalization_params(self, subject, exercise):
        """
        Calculate normalization parameters for a specific subject and exercise.
        Returns mean and std for EMG channels based on rest periods.
        """
        df = self.load_subject_exercise(subject, exercise)
        emg_data = df[[f'emg_{i}' for i in range(1, 13)]].values
        restimulus = df['restimulus'].values
        
        # Find rest periods
        rest_indices = np.where(restimulus == 0)[0]
        rest_data = emg_data[rest_indices, :]
        
        # Calculate mean and std from rest data
        rest_mean = np.mean(rest_data, axis=0)
        rest_std = np.std(rest_data, axis=0)
        
        # Avoid division by zero
        rest_std[rest_std < 1e-6] = 1e-6
        
        return rest_mean, rest_std

    # Modify the _apply_normalization method to accept external parameters
    def _apply_normalization(self, emg_data, rest_mean, rest_std):
        """Normalize EMG data using provided mean and std."""
        normalized_data = (emg_data - rest_mean) / rest_std
        return normalized_data


class DataGenerator(Sequence):
    def __init__(self, data_loader, subjects, exercise, sequence_length=10, 
                 batch_size=32, shuffle=True, target_shift=1):
        self.data_loader = data_loader
        self.subjects = subjects
        self.exercise = exercise
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_shift = target_shift
        self.subject_nmf_models = {}
        self.subject_normalization_params = {}
        
        # Precompute for each subject
        for subject in subjects:
            # Get normalization parameters
            rest_mean, rest_std = data_loader.get_subject_specific_normalization_params(subject, exercise)
            self.subject_normalization_params[subject] = (rest_mean, rest_std)
            
            # Get NMF model
            data = data_loader.get_processed_data(subject, exercise, filter=True, normalize=False)
            emg_data = data['emg']
            emg_normalized = data_loader._apply_normalization(emg_data, rest_mean, rest_std)
            emg_abs = np.abs(emg_normalized)
            
            nmf = NMF(n_components=6, init='random', random_state=42, max_iter=500)
            nmf.fit(emg_abs)
            self.subject_nmf_models[subject] = nmf
            
        # Calculate total number of sequences across all subjects
        self.total_sequences = 0
        self.sequence_indices = []  # (subject_idx, start_idx)
        
        for subject_idx, subject in enumerate(subjects):
            data = data_loader.get_processed_data(subject, exercise, filter=True, normalize=False)
            emg_data = data['emg']
            n_samples = emg_data.shape[0]
            n_sequences = n_samples - sequence_length - target_shift
            
            self.total_sequences += n_sequences
            self.sequence_indices.extend([(subject_idx, i) for i in range(n_sequences)])
        
        if shuffle:
            np.random.shuffle(self.sequence_indices)
    
    def __len__(self):
        return int(np.ceil(self.total_sequences / self.batch_size))
    
    def __getitem__(self, idx):
        batch_X = []
        batch_y = []
        
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.total_sequences)
        
        for i in range(start_idx, end_idx):
            subject_idx, start_pos = self.sequence_indices[i]
            subject = self.subjects[subject_idx]
            
            # Load subject data
            data = self.data_loader.get_processed_data(subject, self.exercise, filter=True, normalize=False)
            emg_data = data['emg']
            glove_data = data['glove']
            
            # Apply subject-specific normalization
            rest_mean, rest_std = self.subject_normalization_params[subject]
            emg_normalized = self.data_loader._apply_normalization(emg_data, rest_mean, rest_std)
            
            # Extract synergy features for the sequence
            emg_window = emg_normalized[start_pos:start_pos+self.sequence_length]
            emg_abs = np.abs(emg_window)
            synergy_features = self.subject_nmf_models[subject].transform(emg_abs)
            
            # Get target
            target_idx = start_pos + self.sequence_length + self.target_shift - 1
            target = glove_data[target_idx]
            
            batch_X.append(synergy_features)
            batch_y.append(target)
        
        return np.array(batch_X), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sequence_indices)
```

---

## File: v1.0\test_data_loader.py
**Path:** `C:\stage\stage\v1.0\test_data_loader.py`

```python
from data_loader import DB2DataLoader

# 1. Initialize the DataLoader
# Point it to the folder that contains the E1, E2, E3 folders.
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# 2. Test loading one file
subject_data = data_loader.load_subject_exercise(subject=1, exercise=1)

# 3. Basic inspection
print(f"DataFrame Shape: {subject_data.shape}")
print(f"Columns: {subject_data.columns.tolist()}")
print(f"Subject unique values: {subject_data['subject'].unique()}")
```

---

## File: v1.0\test_nmf_feature.py
**Path:** `C:\stage\stage\v1.0\test_nmf_feature.py`

```python
from data_loader import DB2DataLoader
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# Test with traditional features
print("Creating dataset with traditional features...")
X_trad, y_trad = data_loader.create_regression_dataset(subject=1, exercise=1, normalize_targets=True, use_nmf=False)
X_train_trad, X_test_trad, y_train_trad, y_test_trad = train_test_split(X_trad, y_trad, test_size=0.2, random_state=42)
model_trad = MultiOutputRegressor(Ridge(alpha=1.0))
model_trad.fit(X_train_trad, y_train_trad)
y_pred_trad = model_trad.predict(X_test_trad)
mse_trad = mean_squared_error(y_test_trad, y_pred_trad)
r2_trad = r2_score(y_test_trad, y_pred_trad)
print(f"Traditional features - MSE: {mse_trad:.4f}, R²: {r2_trad:.4f}")

# Test with NMF features
print("Creating dataset with NMF features...")
X_nmf, y_nmf = data_loader.create_regression_dataset(subject=1, exercise=1, normalize_targets=True, use_nmf=True, n_components=6)
X_train_nmf, X_test_nmf, y_train_nmf, y_test_nmf = train_test_split(X_nmf, y_nmf, test_size=0.2, random_state=42)
model_nmf = MultiOutputRegressor(Ridge(alpha=1.0))
model_nmf.fit(X_train_nmf, y_train_nmf)
y_pred_nmf = model_nmf.predict(X_test_nmf)
mse_nmf = mean_squared_error(y_test_nmf, y_pred_nmf)
r2_nmf = r2_score(y_test_nmf, y_pred_nmf)
print(f"NMF features - MSE: {mse_nmf:.4f}, R²: {r2_nmf:.4f}")
```

---

## File: v1.0\test_normalized_targets.py
**Path:** `C:\stage\stage\v1.0\test_normalized_targets.py`

```python
from data_loader import DB2DataLoader
import numpy as np

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# Create the regression dataset with normalized targets
print("Creating regression dataset with normalized targets...")
X, y = data_loader.create_regression_dataset(subject=1, exercise=1, normalize_targets=True)

print(f"Feature matrix X shape: {X.shape}")
print(f"Target matrix y shape: {y.shape}")
print(f"Number of windows: {X.shape[0]}")

# Check the first few samples
print("\nFirst 5 target vectors (normalized, first 5 glove sensors):")
for i in range(5):
    print(f"Window {i}: {y[i, :5]}")

# Check the range of values
print(f"\nX value range: [{np.min(X):.4f}, {np.max(X):.4f}]")
print(f"y value range: [{np.min(y):.4f}, {np.max(y):.4f}]")
```

---

## File: v1.0\test_preprocessing.py
**Path:** `C:\stage\stage\v1.0\test_preprocessing.py`

```python
import matplotlib.pyplot as plt
from data_loader import DB2DataLoader
import numpy as np

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# Get processed data for subject 1, exercise 1
data = data_loader.get_processed_data(subject=1, exercise=1)

# Extract processed EMG and labels
processed_emg = data['emg']
labels = data['labels']

# Plot a small segment of raw vs. processed EMG for comparison
# First, load raw data again to get the raw EMG
raw_df = data_loader.load_subject_exercise(1, 1)
raw_emg = raw_df['emg_1'].values

# Choose a segment where movement occurs (label != 0)
# Find the first non-rest index
non_rest_indices = np.where(labels != 0)[0]
start_index = non_rest_indices[1000]  # Start after a few movement samples
end_index = start_index + 400  # 400 samples = 200 ms

# Create time axis in milliseconds
time_ms = np.arange(end_index - start_index) * 0.5  # 0.5 ms per sample

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_ms, raw_emg[start_index:end_index], 'b-', label='Raw EMG')
plt.title('Raw EMG Signal (emg_1)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (V)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_ms, processed_emg[start_index:end_index, 0], 'r-', label='Processed EMG')
plt.title('Processed EMG Signal (emg_1)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (SD from rest mean)')
plt.legend()

plt.tight_layout()
plt.savefig('emg_processing_verification.png')
plt.show()
```

---

## File: v1.0\test_regression_dataset.py
**Path:** `C:\stage\stage\v1.0\test_regression_dataset.py`

```python
from data_loader import DB2DataLoader
import numpy as np

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# Create the regression dataset
print("Creating regression dataset...")
X, y = data_loader.create_regression_dataset(subject=1, exercise=1)

print(f"Feature matrix X shape: {X.shape}")
print(f"Target matrix y shape: {y.shape}")
print(f"Number of windows: {X.shape[0]}")

# Check the first few samples
print("\nFirst 5 feature vectors (first 12 are RMS, last 12 are MAV):")
for i in range(5):
    print(f"Window {i}: {X[i, :]}")

print("\nFirst 5 target vectors (first 5 glove sensors):")
for i in range(5):
    print(f"Window {i}: {y[i, :5]}")

# Check for any NaN values
print(f"\nNaN values in X: {np.any(np.isnan(X))}")
print(f"NaN values in y: {np.any(np.isnan(y))}")

# Check the range of values
print(f"\nX value range: [{np.min(X):.4f}, {np.max(X):.4f}]")
print(f"y value range: [{np.min(y):.4f}, {np.max(y):.4f}]")
```

---

## File: v1.0\test_subject2.py
**Path:** `C:\stage\stage\v1.0\test_subject2.py`

```python
from data_loader import DB2DataLoader
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('lstm_model.keras')

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[2], exercises=[1])

# Create sequence dataset for Subject 2
X_s2, y_s2 = data_loader.create_sequence_dataset(subject=2, exercise=1, sequence_length=10, target_shift=1)

# Make predictions
y_pred_s2 = model.predict(X_s2)

# Calculate overall R² score
r2_s2 = r2_score(y_s2, y_pred_s2)
print(f"R² Score on Subject 2: {r2_s2:.4f}")

# Calculate R² per sensor
r2_scores_s2 = []
for i in range(y_s2.shape[1]):
    r2_scores_s2.append(r2_score(y_s2[:, i], y_pred_s2[:, i]))
    print(f"Sensor {i+1}: R² = {r2_scores_s2[-1]:.4f}")

print(f"Mean R² across all sensors for Subject 2: {np.mean(r2_scores_s2):.4f}")

# Add this to your training script after model.fit()
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_history.png')
plt.show()

# Add this to your analysis script
sensor_21_actual = y[:, 20]  # Sensor 21 is index 20 (0-indexed)
sensor_21_pred = y_pred[:, 20]

plt.figure(figsize=(12, 4))
plt.plot(sensor_21_actual[:1000], label='Actual Sensor 21', alpha=0.7)
plt.plot(sensor_21_pred[:1000], label='Predicted Sensor 21', alpha=0.7)
plt.title('Sensor 21: Actual vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Normalized Value')
plt.legend()
plt.savefig('sensor_21_analysis.png')
plt.show()

# Check the range and distribution of Sensor 21 values
print(f"Sensor 21 - Min: {np.min(sensor_21_actual):.4f}, Max: {np.max(sensor_21_actual):.4f}")
print(f"Sensor 21 - Mean: {np.mean(sensor_21_actual):.4f}, Std: {np.std(sensor_21_actual):.4f}")
```

---

## File: v1.0\train_baseline.py
**Path:** `C:\stage\stage\v1.0\train_baseline.py`

```python
from data_loader import DB2DataLoader
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# Create the regression dataset with normalized targets
print("Creating regression dataset...")
X, y = data_loader.create_regression_dataset(subject=1, exercise=1, normalize_targets=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
print("Training the Ridge regression model...")
model = MultiOutputRegressor(Ridge(alpha=1.0))
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate overall metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Overall Mean Squared Error: {mse:.4f}")
print(f"Overall R² Score: {r2:.4f}")

# Calculate metrics per output
print("\nMetrics per glove sensor:")
for i in range(y.shape[1]):
    mse_i = mean_squared_error(y_test[:, i], y_pred[:, i])
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    print(f"Sensor {i+1}: MSE = {mse_i:.4f}, R² = {r2_i:.4f}")
```

---

## File: v1.0\train_lstm.py
**Path:** `C:\stage\stage\v1.0\train_lstm.py`

```python
from data_loader import DB2DataLoader
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# Create sequence dataset with target_shift=1 (125 ms into the future)
print("Creating sequence dataset...")
X, y = data_loader.create_sequence_dataset(subject=1, exercise=1, sequence_length=10, target_shift=1)

print(f"X shape: {X.shape}")  # (n_sequences, sequence_length, n_features)
print(f"y shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='linear'))  # Output layer for 22 targets

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
print("Training the LSTM model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Calculate R² score
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Optional: Save the model
model.save('lstm_model.keras', save_format='keras')
print("Model saved as lstm_model.h5")
```

---

## File: v1.0\train_multiple_subjects.py
**Path:** `C:\stage\stage\v1.0\train_multiple_subjects.py`

```python
from data_loader import DB2DataLoader
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Subjects for training and testing
train_subjects = [1, 2, 3, 4, 5, 6, 7]  # Train on these subjects
test_subject = 8  # Test on this subject

# Initialize the DataLoader for training subjects
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=train_subjects, exercises=[1])

# Collect data from all training subjects
X_train_all = []
y_train_all = []

for subject in train_subjects:
    print(f"Processing subject {subject}...")
    X, y = data_loader.create_sequence_dataset(subject=subject, exercise=1, sequence_length=10, target_shift=1)
    X_train_all.append(X)
    y_train_all.append(y)

# Concatenate all training data
X_train_all = np.concatenate(X_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0)

print(f"Combined training data shape: X={X_train_all.shape}, y={y_train_all.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

# Build the LSTM model with increased dropout
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))  # Increased dropout for regularization
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
print("Training the LSTM model on multiple subjects...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate on the test subject
data_loader_test = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[test_subject], exercises=[1])
X_test, y_test = data_loader_test.create_sequence_dataset(subject=test_subject, exercise=1, sequence_length=10, target_shift=1)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score on Subject {test_subject}: {r2:.4f}")

# Calculate R² per sensor
r2_scores = []
for i in range(y_test.shape[1]):
    r2_scores.append(r2_score(y_test[:, i], y_pred[:, i]))
    print(f"Sensor {i+1}: R² = {r2_scores[-1]:.4f}")

print(f"Mean R² across all sensors for Subject {test_subject}: {np.mean(r2_scores):.4f}")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_history_multiple.png')
plt.show()

# Save the model
model.save('lstm_model_multiple.keras', save_format='keras')
print("Model saved as lstm_model_multiple.keras")
```

---

## File: v1.0\train_with_generator.py
**Path:** `C:\stage\stage\v1.0\train_with_generator.py`

```python
from data_loader import DB2DataLoader, DataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Subjects for training and testing
train_subjects = [1, 2, 3, 4, 5, 6, 7]
test_subject = 8

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=train_subjects + [test_subject], exercises=[1])

# Create data generators
train_generator = DataGenerator(
    data_loader=data_loader,
    subjects=train_subjects,
    exercise=1,
    sequence_length=10,
    batch_size=64,
    shuffle=True,
    target_shift=1
)

# Build the LSTM model
model = Sequential([
    LSTM(64, input_shape=(10, 6), return_sequences=True,
         kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(22, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Train the model using the generator
print("Training the LSTM model with data generator...")
history = model.fit(
    train_generator,
    epochs=10,  # Start with fewer epochs
    verbose=1
)

# Prepare test data
print(f"Preparing test data for subject {test_subject}...")
# (We'll need to create a similar generator for test data or use a different approach)

# For now, let's use a simple evaluation on a small subset
test_generator = DataGenerator(
    data_loader=data_loader,
    subjects=[test_subject],
    exercise=1,
    sequence_length=10,
    batch_size=64,
    shuffle=False,
    target_shift=1
)

# Evaluate on the test subject
test_loss, test_mae = model.evaluate(test_generator, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Save the model
model.save('lstm_model_generator.keras')
print("Model saved as lstm_model_generator.keras")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_history_generator.png')
plt.show()
```

---

## File: v1.0\train_with_synergies.py
**Path:** `C:\stage\stage\v1.0\train_with_synergies.py`

```python
from data_loader import DB2DataLoader
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import NMF

# Subjects for training and testing
train_subjects = [1, 2, 3, 4, 5, 6, 7]
test_subject = 8

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=train_subjects + [test_subject], exercises=[1])

# Step 1: Extract muscle synergies for each subject
n_components = 6  # Number of muscle synergies
subject_synergies = {}
subject_nmf_models = {}

for subject in train_subjects + [test_subject]:
    print(f"Extracting synergies for subject {subject}...")
    
    # Get subject-specific normalization parameters
    rest_mean, rest_std = data_loader.get_subject_specific_normalization_params(subject, 1)
    
    # Load and preprocess data
    data = data_loader.get_processed_data(subject, 1, filter=True, normalize=False)
    emg_data = data['emg']
    
    # Apply subject-specific normalization
    emg_normalized = data_loader._apply_normalization(emg_data, rest_mean, rest_std)
    
    # Use absolute values for NMF (since NMF requires non-negative inputs)
    emg_abs = np.abs(emg_normalized)
    
    # Apply NMF to extract muscle synergies
    nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
    W = nmf.fit_transform(emg_abs)  # Muscle activation patterns
    H = nmf.components_  # Muscle synergies
    
    subject_synergies[subject] = H
    subject_nmf_models[subject] = nmf
    
    print(f"Subject {subject}: Extracted {n_components} synergies")

# Step 2: Create a common synergy set by averaging across subjects
common_synergies = np.mean([subject_synergies[sub] for sub in train_subjects], axis=0)
print("Created common synergy set")

# Step 3: Create dataset using the common synergies
def extract_synergy_features(emg_data, nmf_model):
    """Extract synergy activation features from EMG data."""
    emg_abs = np.abs(emg_data)
    W = nmf_model.transform(emg_abs)
    return W

X_train_all = []
y_train_all = []

for subject in train_subjects:
    print(f"Processing subject {subject}...")
    
    # Get subject-specific normalization parameters
    rest_mean, rest_std = data_loader.get_subject_specific_normalization_params(subject, 1)
    
    # Load and preprocess data
    data = data_loader.get_processed_data(subject, 1, filter=True, normalize=False)
    emg_data = data['emg']
    glove_data = data['glove']
    
    # Apply subject-specific normalization
    emg_normalized = data_loader._apply_normalization(emg_data, rest_mean, rest_std)
    
    # Extract synergy features
    synergy_features = extract_synergy_features(emg_normalized, subject_nmf_models[subject])
    
    # Create sequences
    sequence_length = 10
    n_samples = synergy_features.shape[0]
    n_sequences = n_samples - sequence_length
    
    X_subject = np.zeros((n_sequences, sequence_length, n_components))
    y_subject = np.zeros((n_sequences, 22))
    
    for i in range(n_sequences):
        X_subject[i] = synergy_features[i:i+sequence_length]
        y_subject[i] = glove_data[i+sequence_length]  # Predict next state
    
    X_train_all.append(X_subject)
    y_train_all.append(y_subject)

# Concatenate all training data
X_train_all = np.concatenate(X_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0)

print(f"Combined training data shape: X={X_train_all.shape}, y={y_train_all.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

# Build a more robust LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True,
         kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Train the model
print("Training the LSTM model with muscle synergies...")
history = model.fit(X_train, y_train, epochs=100, batch_size=64, 
                    validation_data=(X_val, y_val), verbose=1)

# Prepare test data
print(f"Preparing test data for subject {test_subject}...")
rest_mean, rest_std = data_loader.get_subject_specific_normalization_params(test_subject, 1)
data_test = data_loader.get_processed_data(test_subject, 1, filter=True, normalize=False)
emg_test = data_test['emg']
glove_test = data_test['glove']

# Apply subject-specific normalization
emg_test_normalized = data_loader._apply_normalization(emg_test, rest_mean, rest_std)

# Extract synergy features
synergy_features_test = extract_synergy_features(emg_test_normalized, subject_nmf_models[test_subject])

# Create sequences for test data
n_samples_test = synergy_features_test.shape[0]
n_sequences_test = n_samples_test - sequence_length

X_test = np.zeros((n_sequences_test, sequence_length, n_components))
y_test = np.zeros((n_sequences_test, 22))

for i in range(n_sequences_test):
    X_test[i] = synergy_features_test[i:i+sequence_length]
    y_test[i] = glove_test[i+sequence_length]

# Evaluate on the test subject
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score on Subject {test_subject}: {r2:.4f}")

# Calculate R² per sensor
r2_scores = []
for i in range(y_test.shape[1]):
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    r2_scores.append(r2_i)
    print(f"Sensor {i+1}: R² = {r2_i:.4f}")

print(f"Mean R² across all sensors for Subject {test_subject}: {np.mean(r2_scores):.4f}")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_history_synergies.png')
plt.show()

# Save the model
model.save('lstm_model_synergies.keras')
print("Model saved as lstm_model_synergies.keras")
```

---

## File: v1.0\train_with_time.py
**Path:** `C:\stage\stage\v1.0\train_with_time.py`

```python
from data_loader import DB2DataLoader
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Initialize the DataLoader
data_loader = DB2DataLoader(base_path="E:/DB/DB2/", subject_list=[1], exercises=[1])

# Create the regression dataset with traditional features and normalized targets
print("Creating dataset with traditional features...")
X, y = data_loader.create_regression_dataset(subject=1, exercise=1, normalize_targets=True, use_nmf=False)

# Apply time shift: shift targets by 150ms (300 samples) forward
# Since we have a window size of 500 samples (250ms) and step of 250 samples (125ms),
# we need to shift the target indices by 300/250 = 1.2 windows. Since we can't shift by fractional windows,
# we'll adjust the data accordingly.

# Calculate the shift in terms of windows
window_size = 500
overlap = 250
step_size = window_size - overlap
shift_ms = 150  # desired shift in milliseconds
shift_samples = int(shift_ms * 2)  # since 2000Hz, 1ms = 2 samples
shift_windows = shift_samples / step_size  # shift in terms of windows

# Since we can't have fractional windows, we'll round to the nearest integer and adjust
shift_windows_int = int(round(shift_windows))
print(f"Shifting targets by {shift_windows_int} windows ({shift_windows_int * step_size / 2} ms)")

# Shift the targets by removing the first shift_windows_int rows from y and the last shift_windows_int rows from X
X_shifted = X[:-shift_windows_int, :]
y_shifted = y[shift_windows_int:, :]

# Check the new shapes
print(f"Original X shape: {X.shape}, y shape: {y.shape}")
print(f"Shifted X shape: {X_shifted.shape}, y shape: {y_shifted.shape}")

# Split the shifted data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_shifted, y_shifted, test_size=0.2, random_state=42)

# Create and train the model
print("Training the Ridge regression model with time shift...")
model = MultiOutputRegressor(Ridge(alpha=1.0))
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate overall metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Overall Mean Squared Error: {mse:.4f}")
print(f"Overall R² Score: {r2:.4f}")

# Calculate metrics per output
print("\nMetrics per glove sensor:")
for i in range(y_shifted.shape[1]):
    mse_i = mean_squared_error(y_test[:, i], y_pred[:, i])
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    print(f"Sensor {i+1}: MSE = {mse_i:.4f}, R² = {r2_i:.4f}")
```

---

## File: v10.0\analyze_10hz_data.py
**Path:** `C:\stage\stage\v10.0\analyze_10hz_data.py`

```python
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
```

---

## File: v10.0\convert_to_10hz_csv.py
**Path:** `C:\stage\stage\v10.0\convert_to_10hz_csv.py`

```python
# Save this as convert_to_10hz_csv.py
# This script converts raw 2000Hz CSVs into a single 10Hz feature-engineered CSV
# for Exploratory Data Analysis (EDA), with verbose logging.

import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # We will define the file list in the main() function.
    OUTPUT_CSV_PATH = "10hz_feature_data.csv" # The SINGLE output CSV
    
    # Signal Processing
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # Feature Engineering (Downsampling)
    WINDOW_MS = 100 # 100ms window
    WINDOW_SAMPLES = int(FS * (WINDOW_MS / 1000.0)) # = 200 samples

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def calculate_hudgins_features(window: np.ndarray) -> np.ndarray:
    """
    Calculates the 4 Hudgins' features (RMS, WL, ZC, SSC) for a 12-channel window.
    Input window shape: (200, 12)
    Output feature vector shape: (48,)
    """
    n_samples, n_channels = window.shape
    features = []
    
    # 1. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2, axis=0))
    features.append(rms)
    
    # 2. Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    features.append(wl)
    
    # 3. Zero Crossings (ZC) (with a small threshold for noise)
    zc = np.sum((window[:-1] * window[1:] < 0) & (np.abs(window[:-1] - window[1:]) > 1e-6), axis=0)
    features.append(zc)
    
    # 4. Slope Sign Changes (SSC) (with a small threshold for noise)
    ssc = np.sum((window[1:-1] - window[:-2]) * (window[1:-1] - window[2:]) > 1e-6, axis=0)
    features.append(ssc)
    
    return np.concatenate(features, axis=0).astype(np.float32)

# --- 1. DATA GATHERING (2000Hz -> 10Hz) ---
def build_10hz_list_of_rows(file_list: List[str]) -> List[Dict]:
    """
    Processes all raw CSVs, extracts features, and returns a list of rows
    (as dictionaries) for the 10Hz data.
    """
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_rows = [] # List to hold all our new 10Hz rows (as dicts)
    group_id_counter = 0 # This is the critical ID for preventing sequence breaks
    
    print("Processing Raw Files (2000Hz -> 10Hz)...")
    
    for file_index, fp in enumerate(tqdm(file_list, desc="Overall Progress", unit="file", total=len(file_list))):
        print(f"\nProcessing File {file_index+1}/{len(file_list)}: {os.path.basename(fp)}")
        try:
            # Load all necessary columns
            use_cols = [f"emg_{i}" for i in range(1, 13)] + [f"glove_{i}" for i in range(1, 23)] + ["restimulus"]
            df = pd.read_csv(fp, usecols=lambda c: c in use_cols)
            
            emg_cols = [f"emg_{i}" for i in range(1, 13)]
            glove_cols = [f"glove_{i}" for i in range(1, 23)]
            
            emg_data = df[emg_cols].values
            glove_data = df[glove_cols].values
            labels = df['restimulus'].values.astype(int)
            
            # Filter the raw EMG data
            for i in range(emg_data.shape[1]):
                emg_data[:, i] = signal.filtfilt(b_notch, a_notch, emg_data[:, i])
                emg_data[:, i] = signal.filtfilt(b_band, a_band, emg_data[:, i])
            
            # Find boundaries of "pure" chunks
            boundaries = np.where(np.diff(labels) != 0)[0] + 1
            chunk_starts = np.concatenate(([0], boundaries))
            chunk_stops = np.concatenate((boundaries, [len(labels)]))
            
            print(f"   ... Found {len(chunk_starts)} 'pure' movement/rest chunks.")
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                if start >= stop: continue
                
                chunk_emg = emg_data[start:stop]
                chunk_glove = glove_data[start:stop]
                chunk_label = labels[start] # All labels in this chunk are the same
                
                n_samples = len(chunk_emg)
                # Use non-overlapping windows, discard partial remainder
                n_windows = n_samples // Config.WINDOW_SAMPLES 
                remainder = n_samples % Config.WINDOW_SAMPLES
                
                if n_windows == 0:
                    print(f"   LOG: Discarding chunk (rows {start}-{stop}, label {chunk_label}) - too short ({n_samples} samples) for one window.")
                    continue
                
                # *** YOUR REQUESTED LOGGING ***
                if remainder > 0:
                    print(f"   LOG: Discarding last {remainder} samples from chunk (rows {start}-{stop}, label {chunk_label}) to ensure pure windows.")
                
                # Process all *full* non-overlapping windows in this chunk
                for i in range(n_windows):
                    win_start = start + (i * Config.WINDOW_SAMPLES)
                    win_stop = win_start + Config.WINDOW_SAMPLES
                    
                    # We reference the *original* dataframe slices
                    emg_window = emg_data[win_start:win_stop]
                    glove_window = glove_data[win_start:win_stop]
                    
                    features_48 = calculate_hudgins_features(emg_window)
                    target_22 = glove_window[-1, :].astype(np.float32)
                    
                    # Create a dictionary for this 10Hz row
                    row_data = {
                        'group_id': group_id_counter, # This ID marks the 'pure' chunk
                        'restimulus': chunk_label
                    }
                    # Add features
                    for f_idx in range(48):
                        row_data[f'feat_{f_idx+1}'] = features_48[f_idx] # Use 1-indexing
                    # Add targets
                    for t_idx in range(22):
                        row_data[f'glove_{t_idx+1}'] = target_22[t_idx] # Use 1-indexing
                        
                    all_rows.append(row_data)
                
                group_id_counter += 1 # Increment for each new, pure chunk
                
        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process file {fp}. Error: {e}")
            continue
            
    print(f"\n... 10Hz data extraction complete. Found {len(all_rows)} total 10Hz samples.")
    return all_rows

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING CONVERSION TO 10Hz CSV (PHASE 2a)")
    print("=============================================================")
    
    # ==================== EDIT THIS LIST ====================
    # Paste your file paths here for the test run.
    # Use 'r' before the string to handle Windows backslashes '\'.
    FILES_TO_PROCESS = [
        r"D:\stage\v5.0\data\DB2\s1\E1_A1.csv",
        r"DS:\stage\v5.0\data\DB2\s1\E2_A1.csv",
        r"D:\stage\v5.0\data\DB2\s2\E1_A1.csv"
        # Add more files here when you are ready for the full run
    ]
    # ========================================================
    
    print(f"Starting processing for {len(FILES_TO_PROCESS)} manually specified files...")

    # --- STEP 1: Process 2000Hz -> 10Hz ---
    list_of_rows = build_10hz_list_of_rows(FILES_TO_PROCESS)
    
    if not list_of_rows:
        print("❌ FATAL: No 10Hz data could be extracted. Check file paths and data.")
        return

    # --- STEP 2: Convert list of dicts to DataFrame and save ---
    print(f"\n... Converting to DataFrame and saving to {Config.OUTPUT_CSV_PATH}...")
    try:
        final_df = pd.DataFrame(list_of_rows)
        # We will use CSV as you requested, not Parquet.
        final_df.to_csv(Config.OUTPUT_CSV_PATH, index=False)
        print(f"   ... Saved as {Config.OUTPUT_CSV_PATH}.")

    except Exception as e:
        print(f"❌ FATAL: Failed to save DataFrame to CSV. Error: {e}")
        return

    print("\n" + "=" * 60)
    print(f"🎉 CONVERSION TO 10Hz CSV COMPLETE!")
    print(f"   Total 10Hz samples created: {len(list_of_rows):,}")
    print(f"   Output file: {Config.OUTPUT_CSV_PATH}")
    print("\n   ✅ YOU CAN NOW OPEN THIS CSV FILE TO SEE THE DATA.")
    print("\nNext step: Run the 'Phase 2b' analysis script on this new file.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v10.0\convert_to_tfrecord.py
**Path:** `C:\stage\stage\v10.0\convert_to_tfrecord.py`

```python
# Save this as convert_to_tfrecord.py
# This is the FINAL data conversion script (Phase 2c).
# It converts the 10Hz CSV into a balanced, normalized, sequential TFRecord dataset.

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Input Files ---
    DATA_CSV_FILE = "10hz_feature_data.csv"
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    # --- Output ---
    OUTPUT_DIR = "tfrecord_dataset_v2"
    FEATURE_SCALER_FILE = "feature_scaler.pkl"
    TARGET_SCALER_FILE = "target_scaler.pkl" # We save the Min/Max as a scaler
    
    # --- Clustering & Balancing ---
    NUM_MOVEMENT_CLUSTERS = 6
    
    # --- Sequencing ---
    SEQUENCE_LENGTH = 50 # 50 samples * 100ms/sample = 5 seconds
    
    # --- TFRecord ---
    SEQUENCES_PER_SHARD = 500
    VALIDATION_SPLIT = 0.15 # 15% of our balanced sequences will be for validation

# --- 🛠️ HELPER FUNCTIONS ---

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the 10Hz CSV data."""
    print(f"Loading 10Hz data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        raise FileNotFoundError
    
    df = pd.read_csv(file_path)
    print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
    return df

def create_virtual_motors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Creates the 7 Virtual Motor columns based on the sdata201453.pdf paper."""
    print("--- 1. Creating 7 Virtual Motors ---")
    
    # Sensor map based on sdata201453.pdf
    VIRTUAL_MOTORS: Dict[str, List[str]] = {
        "motor_thumb_flex": [f"glove_{i}" for i in [3, 4]],
        "motor_index_flex": [f"glove_{i}" for i in [6, 7, 8]],
        "motor_middle_flex": [f"glove_{i}" for i in [10, 11, 12]],
        "motor_ring_flex": [f"glove_{i}" for i in [14, 15, 16]],
        "motor_pinky_flex": [f"glove_{i}" for i in [18, 19, 20]],
        "motor_thumb_abduct": ["glove_2"],
        "motor_wrist_flex": ["glove_21"]
    }
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
    
    # Get the input feature column names (feat_1 to feat_48)
    feature_cols = [f'feat_{i+1}' for i in range(48)]
    
    print(f"   ... Created {len(motor_cols)} motor columns.")
    return df, feature_cols, motor_cols

def normalize_data(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], norm_params_file: str, out_dir: str) -> pd.DataFrame:
    """
    Applies Min-Max normalization to motors (targets) and Z-Score to features (inputs).
    """
    print("--- 2. Normalizing Data ---")
    
    # --- A. Normalize Motors (Targets) ---
    print(f"   ... Applying Min-Max (0-1) normalization to {len(m_cols)} motors...")
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    for motor in m_cols:
        p = params[motor]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        
        # Ensure range is not zero to avoid division by zero
        if motor_range == 0: motor_range = 1.0 
        
        # Apply normalization and clip between 0 and 1
        df[motor] = (df[motor] - motor_min) / motor_range
        df[motor] = np.clip(df[motor], 0.0, 1.0)
    
    # Save the Min/Max params as our "target scaler" for the model
    # We will need to "un-normalize" the model's output later
    joblib.dump(params, os.path.join(out_dir, Config.TARGET_SCALER_FILE))

    # --- B. Normalize Features (Inputs) ---
    print(f"   ... Applying Z-Score normalization to {len(f_cols)} features...")
    feature_data = df[f_cols].values
    feature_scaler = StandardScaler()
    
    # Fit and transform the features
    features_scaled = feature_scaler.fit_transform(feature_data)
    
    # Save the scaler for the model to use on new data
    joblib.dump(feature_scaler, os.path.join(out_dir, Config.FEATURE_SCALER_FILE))
    
    # Put the normalized features back into the DataFrame
    df[f_cols] = features_scaled
    print("   ... Normalization complete. Scalers saved.")
    return df

def balance_data(df: pd.DataFrame, m_cols: List[str]) -> pd.DataFrame:
    """
    Performs stratified sampling using 'rest' + K-Means movement clusters.
    """
    print("--- 3. Balancing Data (Stratified Sampling) ---")
    
    # Separate Rest and Movement data
    rest_df = df[df['restimulus'] == 0].copy()
    move_df = df[df['restimulus'] != 0].copy()
    
    if rest_df.empty or move_df.empty:
        print("   ⚠️ Warning: Missing rest or movement data. Balancing may be skewed.")
        return df # Return original df if we can't balance

    print(f"   ... Found {len(rest_df)} 'rest' samples and {len(move_df)} 'movement' samples.")
    
    # --- A. Cluster Movements ---
    print(f"   ... Finding {Config.NUM_MOVEMENT_CLUSTERS} movement archetypes (K-Means)...")
    # We cluster on the *normalized* motor data
    kmeans = KMeans(n_clusters=Config.NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_df['cluster'] = kmeans.fit_predict(move_df[m_cols])
    
    # --- B. Stratified Sampling ---
    all_balanced_dfs = []
    
    # Find the count of each cluster
    cluster_counts = move_df['cluster'].value_counts()
    
    # Find the "minority class" (smallest movement cluster)
    min_move_count = cluster_counts.min()
    print(f"   ... Minority movement cluster has {min_move_count} samples.")
    
    # Sample from movement clusters
    for i in range(Config.NUM_MOVEMENT_CLUSTERS):
        cluster_df = move_df[move_df['cluster'] == i]
        all_balanced_dfs.append(cluster_df.sample(n=min_move_count, random_state=42, replace=False))
        
    # Sample from "Rest"
    # We will take an equal amount of "rest" samples
    rest_sample_size = min(len(rest_df), min_move_count)
    print(f"   ... Sampling {rest_sample_size} 'rest' samples.")
    all_balanced_dfs.append(rest_df.sample(n=rest_sample_size, random_state=42, replace=False))
    
    # Combine all balanced data
    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} samples.")
    return balanced_df

def write_sequences_to_tfrecord(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], output_dir: str):
    """
    Builds sequences from the balanced/normalized 10Hz data and saves to TFRecord.
    """
    print("--- 4. Creating Sequences & Saving to TFRecord ---")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Group by the 'group_id' to prevent sequences from crossing boundaries
    grouped = df.groupby('group_id')
    
    train_writer = None
    val_writer = None
    shard_counts = {'train': 0, 'val': 0}
    seq_counts = {'train': 0, 'val': 0}

    # Iterate over each "pure" chunk
    for group_id, group_df in tqdm(grouped, desc="   ... Processing groups into sequences", unit="group"):
        if len(group_df) < Config.SEQUENCE_LENGTH:
            continue # This chunk is too short to make a sequence
            
        features = group_df[f_cols].values
        targets = group_df[m_cols].values.astype(np.float32) # Ensure targets are float32
        
        # Slide our 50-sample window over this pure chunk
        for i in range(len(group_df) - Config.SEQUENCE_LENGTH + 1):
            
            # --- Assign to Train or Validation ---
            # We use the group_id to make the split. This ensures
            # ALL sequences from one chunk go to *either* train *or* val.
            # This is a robust way to prevent data leakage.
            if group_id % 100 < (Config.VALIDATION_SPLIT * 100):
                split = 'val'
            else:
                split = 'train'

            # --- Handle Shard Rollover ---
            if seq_counts[split] % Config.SEQUENCES_PER_SHARD == 0:
                if split == 'train' and train_writer: train_writer.close()
                if split == 'val' and val_writer: val_writer.close()
                
                path = os.path.join(output_dir, split, f'data_{shard_counts[split]:04d}.tfrecord')
                if split == 'train':
                    train_writer = tf.io.TFRecordWriter(path)
                else:
                    val_writer = tf.io.TFRecordWriter(path)
                shard_counts[split] += 1
            
            # Extract the 50-sample sequences
            seq_features = features[i : i + Config.SEQUENCE_LENGTH]
            seq_targets = targets[i : i + Config.SEQUENCE_LENGTH]
            
            # Serialize and create the TFExample
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(seq_features)),
                'targets': _bytes_feature(tf.io.serialize_tensor(seq_targets))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Write to the correct file
            if split == 'train':
                train_writer.write(example.SerializeToString())
            else:
                val_writer.write(example.SerializeToString())
            
            seq_counts[split] += 1

    if train_writer: train_writer.close()
    if val_writer: val_writer.close()
    
    print(f"\n   ... Sequencing complete.")
    print(f"   ... Wrote {seq_counts['train']:,} training sequences to {shard_counts['train']} shards.")
    print(f"   ... Wrote {seq_counts['val']:,} validation sequences to {shard_counts['val']} shards.")
    return seq_counts['train'] + seq_counts['val']

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING FINAL CONVERSION (PHASE 2c)")
    print("=============================================================")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load 10Hz CSV
    df = load_data(Config.DATA_CSV_FILE)
    if df is None: return

    # 2. Create 7 Virtual Motors
    df, feature_cols, motor_cols = create_virtual_motors(df)
    
    # 3. Normalize Inputs (Z-Score) and Outputs (Min-Max)
    df = normalize_data(df, feature_cols, motor_cols, Config.NORMALIZATION_PARAMS_FILE, Config.OUTPUT_DIR)

    # 4. Balance the data using Stratified Sampling
    balanced_df = balance_data(df, motor_cols)
    
    # 5. Create Sequences and Save to TFRecord
    total_seqs = write_sequences_to_tfrecord(balanced_df, feature_cols, motor_cols, Config.OUTPUT_DIR)

    # --- FINAL METADATA ---
    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_balanced_10hz_samples": len(balanced_df),
        "total_sequences_written": total_seqs,
        "feature_scaler": Config.FEATURE_SCALER_FILE,
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 FINAL CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {total_seqs:,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("   Scalers saved: feature_scaler.pkl, target_scaler.pkl")
    print("\n   ✅ THE DATASET IS NOW 100% READY FOR TRAINING.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v10.0\convert_to_tfrecord_v2.py
**Path:** `C:\stage\stage\v10.0\convert_to_tfrecord_v2.py`

```python
# Save this as convert_to_tfrecord_v2.py
# This is the FINAL data conversion script (Phase 2c).
# It converts the 10Hz CSV into a balanced, normalized, sequential TFRecord dataset.
# *** VERSION 2.1: Includes the float32 cast fix ***

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    DATA_CSV_FILE = "10hz_feature_data.csv"
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    OUTPUT_DIR = "tfrecord_dataset_v2"
    FEATURE_SCALER_FILE = "feature_scaler.pkl"
    TARGET_SCALER_FILE = "target_scaler.pkl" 
    
    NUM_MOVEMENT_CLUSTERS = 6
    SEQUENCE_LENGTH = 50 # 5 seconds
    
    SEQUENCES_PER_SHARD = 500
    VALIDATION_SPLIT = 0.15 

# --- 🛠️ HELPER FUNCTIONS ---

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the 10Hz CSV data."""
    print(f"Loading 10Hz data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        raise FileNotFoundError
    
    df = pd.read_csv(file_path)
    print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
    return df

def create_virtual_motors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Creates the 7 Virtual Motor columns."""
    print("--- 1. Creating 7 Virtual Motors ---")
    
    VIRTUAL_MOTORS: Dict[str, List[str]] = {
        "motor_thumb_flex": [f"glove_{i}" for i in [3, 4]],
        "motor_index_flex": [f"glove_{i}" for i in [6, 7, 8]],
        "motor_middle_flex": [f"glove_{i}" for i in [10, 11, 12]],
        "motor_ring_flex": [f"glove_{i}" for i in [14, 15, 16]],
        "motor_pinky_flex": [f"glove_{i}" for i in [18, 19, 20]],
        "motor_thumb_abduct": ["glove_2"],
        "motor_wrist_flex": ["glove_21"]
    }
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
    
    feature_cols = [f'feat_{i+1}' for i in range(48)]
    return df, feature_cols, motor_cols

def normalize_data(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], norm_params_file: str, out_dir: str) -> pd.DataFrame:
    """
    Applies Min-Max to motors (targets) and Z-Score to features (inputs).
    This is applied to the *entire* 10Hz dataframe.
    """
    print("--- 2. Normalizing All 10Hz Data ---")
    
    # --- A. Normalize Motors (Targets) ---
    print(f"   ... Applying Min-Max (0-1) to {len(m_cols)} motors...")
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    for motor in m_cols:
        p = params[motor]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        if motor_range == 0: motor_range = 1.0 
        
        df[motor] = (df[motor] - motor_min) / motor_range
        df[motor] = np.clip(df[motor], 0.0, 1.0)
    
    joblib.dump(params, os.path.join(out_dir, Config.TARGET_SCALER_FILE))

    # --- B. Normalize Features (Inputs) ---
    print(f"   ... Applying Z-Score to {len(f_cols)} features...")
    feature_data = df[f_cols].values
    feature_scaler = StandardScaler()
    
    # Fit and transform the features
    # Save as float32 to prevent type mismatch error
    features_scaled = feature_scaler.fit_transform(feature_data).astype(np.float32)
    
    joblib.dump(feature_scaler, os.path.join(out_dir, Config.FEATURE_SCALER_FILE))
    
    # Put the normalized features back into the DataFrame
    df[f_cols] = features_scaled
    print("   ... Normalization complete. Scalers saved.")
    return df

def create_all_sequences(df: pd.DataFrame, f_cols: List[str], m_cols: List[str]) -> pd.DataFrame:
    """
    Builds all possible sequences from the full, normalized 10Hz dataframe.
    """
    print("--- 3. Creating All Possible Sequences ---")
    
    all_sequences = [] # This will be a list of dicts
    
    # Group by the 'group_id' to prevent sequences from crossing boundaries
    grouped = df.groupby('group_id')
    
    for group_id, group_df in tqdm(grouped, desc="   ... Processing groups", unit="group"):
        if len(group_df) < Config.SEQUENCE_LENGTH:
            continue # This chunk is too short
            
        # ==================== THE FIX (Alternative Location) ====================
        # We ensure features are float32 before sequencing.
        # This is the one-line fix.
        features = group_df[f_cols].values.astype(np.float32) 
        # ======================================================================
        
        targets = group_df[m_cols].values.astype(np.float32)
        labels = group_df['restimulus'].values
        
        # Slide our 50-sample window over this pure chunk
        for i in range(len(group_df) - Config.SEQUENCE_LENGTH + 1):
            
            seq_label = labels[i + Config.SEQUENCE_LENGTH - 1]
            seq_features = features[i : i + Config.SEQUENCE_LENGTH]
            seq_targets = targets[i : i + Config.SEQUENCE_LENGTH]
            
            all_sequences.append({
                "features": seq_features,
                "targets": seq_targets,
                "label": seq_label
            })

    print(f"   ... Found {len(all_sequences):,} total valid sequences.")
    return pd.DataFrame(all_sequences)

def balance_sequences(seq_df: pd.DataFrame, m_cols: List[str]) -> pd.DataFrame:
    """
    Performs stratified sampling on the SEQUENCES, not the 10Hz data.
    """
    print("--- 4. Balancing Sequences (Stratified Sampling) ---")
    
    rest_seq_df = seq_df[seq_df['label'] == 0].copy()
    move_seq_df = seq_df[seq_df['label'] != 0].copy()
    
    if rest_seq_df.empty or move_seq_df.empty:
        print("   ⚠️ Warning: Missing rest or movement sequences. Balancing may be skewed.")
        return seq_df

    print(f"   ... Found {len(rest_seq_df)} 'rest' sequences and {len(move_seq_df)} 'movement' sequences.")
    
    print(f"   ... Finding {Config.NUM_MOVEMENT_CLUSTERS} movement archetypes (K-Means)...")
    
    avg_poses = np.array([seq.mean(axis=0) for seq in move_seq_df['targets']])
    
    kmeans = KMeans(n_clusters=Config.NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_seq_df['cluster'] = kmeans.fit_predict(avg_poses)
    
    all_balanced_dfs = []
    
    cluster_counts = move_seq_df['cluster'].value_counts()
    
    min_move_count = cluster_counts.min()
    print(f"   ... Minority movement cluster has {min_move_count} sequences.")
    
    for i in range(Config.NUM_MOVEMENT_CLUSTERS):
        cluster_df = move_seq_df[move_seq_df['cluster'] == i]
        all_balanced_dfs.append(cluster_df.sample(n=min_move_count, random_state=42, replace=False))
        
    rest_sample_size = min(len(rest_seq_df), min_move_count)
    print(f"   ... Sampling {rest_sample_size} 'rest' sequences.")
    all_balanced_dfs.append(rest_seq_df.sample(n=rest_sample_size, random_state=42, replace=False))
    
    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} sequences.")
    return balanced_df

def write_sequences_to_tfrecord(seq_df: pd.DataFrame, output_dir: str):
    """
    Saves the final, balanced DataFrame of sequences to TFRecord files.
    """
    print("--- 5. Saving Sequences to TFRecord ---")
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    val_size = int(len(seq_df) * Config.VALIDATION_SPLIT)
    val_df = seq_df.iloc[:val_size]
    train_df = seq_df.iloc[val_size:]
    
    print(f"   ... Splitting into {len(train_df)} train and {len(val_df)} val sequences.")
    
    shard_counts = {'train': 0, 'val': 0}

    # --- Write Training Data ---
    with tqdm(total=len(train_df), desc="   ... Writing train shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(train_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(train_dir, f'data_{shard_counts["train"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['train'] += 1
            
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    # --- Write Validation Data ---
    with tqdm(total=len(val_df), desc="   ... Writing val shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(val_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(val_dir, f'data_{shard_counts["val"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['val'] += 1
                
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    print(f"\n   ... Wrote {len(train_df):,} training sequences to {shard_counts['train']} shards.")
    print(f"   ... Wrote {len(val_df):,} validation sequences to {shard_counts['val']} shards.")
    return len(seq_df)

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING FINAL CONVERSION (PHASE 2c) - V2 (Corrected Order)")
    print("=============================================================")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = load_data(Config.DATA_CSV_FILE)
    if df is None: return

    df, feature_cols, motor_cols = create_virtual_motors(df)
    
    df = normalize_data(df, feature_cols, motor_cols, Config.NORMALIZATION_PARAMS_FILE, Config.OUTPUT_DIR)

    all_sequences_df = create_all_sequences(df, feature_cols, motor_cols)
    
    balanced_seq_df = balance_sequences(all_sequences_df, motor_cols)
    
    total_seqs = write_sequences_to_tfrecord(balanced_seq_df, Config.OUTPUT_DIR)

    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_unbalanced_sequences": len(all_sequences_df),
        "total_balanced_sequences": total_seqs,
        "feature_scaler": Config.FEATURE_SCALER_FILE,
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 FINAL CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {total_seqs:,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("   Scalers saved: feature_scaler.pkl, target_scaler.pkl")
    print("\n   ✅ THE DATASET IS NOW 100% READY FOR TRAINING.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v10.0\train_sequential_v2.py
**Path:** `C:\stage\stage\v10.0\train_sequential_v2.py`

```python
# Save this as train_sequential_v2.py
# This script trains our new sequential models (v4, v5)
# on the final TFRecord dataset.

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import glob
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Data ---
    TRAIN_TFRECORD_PATH = "tfrecord_dataset_v2/train/*.tfrecord"
    VAL_TFRECORD_PATH = "tfrecord_dataset_v2/val/*.tfrecord"
    
    # --- Model Hyperparameters (from our data conversion) ---
    SEQUENCE_LENGTH = 50 # 50 steps (5 seconds)
    NUM_FEATURES = 48    # 48 Hudgins' features (12 channels * 4)
    NUM_TARGETS = 7      # 7 Virtual Motors
    
    # --- Training ---
    BATCH_SIZE = 16 # Small dataset, so small batch size
    EPOCHS = 150    # Let's train for a while; EarlyStopping will find the best
    
# --- 1. DATA LOADER ---

def parse_sequence_tfrecord(example_proto):
    """
    Parses the TFRecord examples created by convert_to_tfrecord_v2.py.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'targets': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    features = tf.io.parse_tensor(parsed['features'], out_type=tf.float32)
    targets = tf.io.parse_tensor(parsed['targets'], out_type=tf.float32)
    
    # Set the shapes explicitly
    features = tf.reshape(features, [Config.SEQUENCE_LENGTH, Config.NUM_FEATURES])
    targets = tf.reshape(targets, [Config.SEQUENCE_LENGTH, Config.NUM_TARGETS])
    
    return features, targets

def create_dataset(tfrecord_pattern: str, batch_size: int, shuffle=True):
    """Creates a high-performance TensorFlow dataset from TFRecord files."""
    
    file_list = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    dataset = file_list.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        # Shuffle *before* mapping for better performance
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.map(parse_sequence_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    if shuffle:
        # Repeat indefinitely for training
        dataset = dataset.repeat()
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. MODEL ARCHITECTURES ---

def create_v4_lstm_baseline(model_name="v4_lstm_baseline"):
    """
    Model v4: A simple LSTM-only model. This is our sequential baseline.
    This is a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # LSTM layer that processes the sequence.
    # return_sequences=True is ESSENTIAL for Seq2Seq. It outputs
    # a prediction for *every* of the 50 time steps.
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    
    # A final Dense layer, wrapped in TimeDistributed.
    # This applies the *same* dense layer to *each* of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear') # 'linear' for regression
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse', # Mean Squared Error is standard for regression
        metrics=['mae'] # Mean Absolute Error is easier to interpret
    )
    return model

def create_v5_cnn_lstm(model_name="v5_cnn_lstm"):
    """
    Model v5: The SOTA-informed RCNN (CNN-LSTM) Hybrid.
    This is also a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # 1. CNN Feature Extraction Block
    # 1D Conv layers to find local patterns (motifs) in the 50-step sequence.
    # We use 'causal' padding to prevent the model from "cheating" by
    # looking at future time steps to predict the present.
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(x)
    
    # 2. LSTM Temporal Processing Block
    # The LSTM layer learns the long-term relationships between the
    # features extracted by the CNN.
    # We must use return_sequences=True.
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.4)(x)
    
    # 3. TimeDistributed Prediction Head
    # Applies a Dense layer to each of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear')
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

# --- 3. TRAINING & EVALUATION ---

def setup_experiment_directory(experiment_name):
    """Creates a unique directory for storing all artifacts of a training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def train_model(model_creator, model_name, train_dataset, val_dataset, train_steps, val_steps):
    """The main function to train a single model and save all results."""
    print(f"\n{'='*60}\n🚀 STARTING TRAINING FOR {model_name}\n{'='*60}")
    
    exp_dir = setup_experiment_directory(f"series_{model_name}")
    model = model_creator()
    model.summary()
    
    # Callbacks
    best_model_path = os.path.join(exp_dir, "models", f"best_{model_name}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1), # 25 patience
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
    ]

    print(f"\nTraining {model_name}...")
    start_time = datetime.datetime.now()

    history = model.fit(
        train_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    
    # Load best model and get its validation MAE
    best_val_mae = min(history.history['val_mae'])
    
    # Save results
    results = {
        'model_version': model_name,
        'best_validation_mae': float(best_val_mae),
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_mae']) + 1),
        'training_time_seconds': training_time.total_seconds(),
        'model_architecture': json.loads(model.to_json())
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create and save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE'); plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f'{model_name}_training_history.png'))
    plt.close()

    print(f"\n🎯 {model_name} - FINAL RESULTS:")
    print(f"   Best Validation MAE: {results['best_validation_mae']:.4f}")
    print(f"   Training Time: {str(training_time).split('.')[0]}")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"💾 Results and artifacts saved to: {exp_dir}")
    return results

# --- 4. MAIN EXECUTION ---
def main():
    print("🚀 STARTING SEQUENTIAL MODEL TRAINING (PHASE 2d)")
    print("=============================================================")
    
    # Find the data we just created
    train_files = glob.glob(Config.TRAIN_TFRECORD_PATH)
    val_files = glob.glob(Config.VAL_TFRECORD_PATH)
    
    if not train_files or not val_files:
        print(f"❌ FATAL: Could not find TFRecord files.")
        print(f"   Checked for train: {Config.TRAIN_TFRECORD_PATH}")
        print(f"   Checked for val: {Config.VAL_TFRECORD_PATH}")
        return

    # Get total number of samples from our log file
    # This is a bit of a hack, but it's the easiest way
    # In a real-world scenario, we'd read this from metadata.json
    total_train_seqs = 405
    total_val_seqs = 71
    
    # Calculate steps per epoch
    train_steps = total_train_seqs // Config.BATCH_SIZE
    val_steps = total_val_seqs // Config.BATCH_SIZE
    
    # Ensure steps are at least 1
    if train_steps == 0: train_steps = 1
    if val_steps == 0: val_steps = 1

    print(f"   Found {total_train_seqs} train sequences and {total_val_seqs} val sequences.")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Steps/Epoch: {train_steps}")
    print(f"   Validation Steps/Epoch: {val_steps}")

    # Create the datasets
    train_dataset = create_dataset(Config.TRAIN_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(Config.VAL_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=False)

    # Define models to train
    models_to_train = {
        'v4_lstm_baseline': create_v4_lstm_baseline,
        'v5_cnn_lstm': create_v5_cnn_lstm,
    }
    
    all_results = []
    
    for model_name, model_creator in models_to_train.items():
        results = train_model(
            model_creator, 
            model_name, 
            train_dataset, 
            val_dataset, 
            train_steps, 
            val_steps
        )
        all_results.append(results)

    print(f"\n{'='*70}\n📊 SEQUENTIAL EXPERIMENTS COMPLETE 📊\n{'='*70}")
    
    # Print summary table
    print(f"{'Model':<20} {'Best Val MAE':<15} {'Best Epoch':<12}")
    print('-'*70)
    for res in sorted(all_results, key=lambda x: x['best_validation_mae']):
        print(f"{res['model_version']:<20} {res['best_validation_mae']:.4f}{'':<11} {res['best_epoch']:<12}")
    
    best_model_results = min(all_results, key=lambda x: x['best_validation_mae'])
    print(f"\n🏆 BEST SEQUENTIAL MODEL: {best_model_results['model_version']} (Val MAE: {best_model_results['best_validation_mae']:.4f})")
    print(f"\n   Recall: Our 'Point-in-Time' (v2) model had a Test MAE of 0.1610")
    print(f"   ... We will need to run a final test, but this validation MAE is our new benchmark.")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
```

---

## File: v11.0\analyze_10hz_data.py
**Path:** `C:\stage\stage\v11.0\analyze_10hz_data.py`

```python
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

# --- ⚙️ CONFIGURATION (Defined as Global Constants) ---
DATA_CSV_FILE = "10hz_feature_data.csv"  # CRITICAL: Must be the final CSV file
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
    glove_cols = [f'glove_{i}' for i in range(1, NUM_GLOVE_SENSORS + 1)]  # Use global constant
    
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
    print(f"\n--- 🔬 Generating Figure 3.13 (k={NUM_MOVEMENT_CLUSTERS} Clustering) ---")  # Use global constant
    
    # 1. Apply PCA to reduce from 22 dimensions to 2
    pca = PCA(n_components=2)
    glove_pca = pca.fit_transform(glove_data_scaled)
    
    print(f"   ... PCA Explained Variance (2 components): {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # 2. Apply K-Means clustering to find the archetypal movements
    print(f"   ... Applying K-Means to find {NUM_MOVEMENT_CLUSTERS} movement groups...")  # Use global constant
    kmeans = KMeans(n_clusters=NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_df['cluster'] = kmeans.fit_predict(glove_data_scaled)
    
    # 3. Plot the clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=glove_pca[:, 0], 
        y=glove_pca[:, 1], 
        hue=move_df['cluster'],
        palette=sns.color_palette("hsv", n_colors=NUM_MOVEMENT_CLUSTERS),  # Use global constant
        alpha=0.5, 
        s=10,
        legend='full'
    )
    plt.title(f"Movement Groups (PCA + K-Means Clustering, k={NUM_MOVEMENT_CLUSTERS})", fontsize=16)  # Use global constant
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
    
    # ✅ FIXED: Use global constant directly (NO 'Config' class)
    move_df, glove_data_scaled = load_and_prepare_data(DATA_CSV_FILE) 
    
    if move_df is not None:
        analyze_movement_clusters(move_df, glove_data_scaled)
        
    print("\n" + "=" * 60)
    print("🎉 PLOTTING COMPLETE!")
    print("   Ensure the new 'movement_clusters_pca.png' is used in Figure 3.13.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v11.0\analyze_10hz_data_v0.py
**Path:** `C:\stage\stage\v11.0\analyze_10hz_data_v0.py`

```python
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
    
    # A. Primary Flexion/Abduction (7 Motors)
    "motor_thumb_flex": ["glove_3", "glove_4"],
    "motor_index_flex": ["glove_5", "glove_6", "glove_7"],
    "motor_middle_flex": ["glove_8", "glove_9", "glove_10"],
    "motor_ring_flex": ["glove_12", "glove_13", "glove_14"],
    "motor_pinky_flex": ["glove_16", "glove_17", "glove_18"],
    "motor_thumb_abduct": ["glove_2"],
    "motor_wrist_flex_ext": ["glove_21"], # Corrected motor name
    
    # B. Secondary Adduction/Waving (5 Motors - Enhanced)
    "motor_index_adduct": ["glove_11"],
    "motor_middle_adduct": ["glove_15"],
    "motor_ring_adduct": ["glove_19"],
    "motor_pinky_adduct": ["glove_20"],
    "motor_hand_waving": ["glove_1", "glove_22"] # Glove 1 and 22 averaged
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
    print("\n--- 🔬 1. Creating 12 Virtual Motors ---")
    
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
```

---

## File: v11.0\analyze_movement_clusters.py
**Path:** `C:\stage\stage\v11.0\analyze_movement_clusters.py`

```python
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
```

---

## File: v11.0\convert_to_10hz_csv.py
**Path:** `C:\stage\stage\v11.0\convert_to_10hz_csv.py`

```python
# Save this as convert_to_10hz_csv.py
# This script converts raw 2000Hz CSVs into a single 10Hz feature-engineered CSV
# for Exploratory Data Analysis (EDA), with verbose logging.

import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # We will define the file list in the main() function.
    OUTPUT_CSV_PATH = "10hz_feature_data.csv" # The SINGLE output CSV
    
    # Signal Processing
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # Feature Engineering (Downsampling)
    WINDOW_MS = 100 # 100ms window
    WINDOW_SAMPLES = int(FS * (WINDOW_MS / 1000.0)) # = 200 samples

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def calculate_hudgins_features(window: np.ndarray) -> np.ndarray:
    """
    Calculates the 4 Hudgins' features (RMS, WL, ZC, SSC) for a 12-channel window.
    Input window shape: (200, 12)
    Output feature vector shape: (48,)
    """
    n_samples, n_channels = window.shape
    features = []
    
    # 1. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2, axis=0))
    features.append(rms)
    
    # 2. Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    features.append(wl)
    
    # 3. Zero Crossings (ZC) (with a small threshold for noise)
    zc = np.sum((window[:-1] * window[1:] < 0) & (np.abs(window[:-1] - window[1:]) > 1e-6), axis=0)
    features.append(zc)
    
    # 4. Slope Sign Changes (SSC) (with a small threshold for noise)
    ssc = np.sum((window[1:-1] - window[:-2]) * (window[1:-1] - window[2:]) > 1e-6, axis=0)
    features.append(ssc)
    
    return np.concatenate(features, axis=0).astype(np.float32)

# --- 1. DATA GATHERING (2000Hz -> 10Hz) ---
def build_10hz_list_of_rows(file_list: List[str]) -> List[Dict]:
    """
    Processes all raw CSVs, extracts features, and returns a list of rows
    (as dictionaries) for the 10Hz data.
    """
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_rows = [] # List to hold all our new 10Hz rows (as dicts)
    group_id_counter = 0 # This is the critical ID for preventing sequence breaks
    
    print("Processing Raw Files (2000Hz -> 10Hz)...")
    
    for file_index, fp in enumerate(tqdm(file_list, desc="Overall Progress", unit="file", total=len(file_list))):
        print(f"\nProcessing File {file_index+1}/{len(file_list)}: {os.path.basename(fp)}")
        try:
            # Load all necessary columns
            use_cols = [f"emg_{i}" for i in range(1, 13)] + [f"glove_{i}" for i in range(1, 23)] + ["restimulus"]
            df = pd.read_csv(fp, usecols=lambda c: c in use_cols)
            
            emg_cols = [f"emg_{i}" for i in range(1, 13)]
            glove_cols = [f"glove_{i}" for i in range(1, 23)]
            
            emg_data = df[emg_cols].values
            glove_data = df[glove_cols].values
            labels = df['restimulus'].values.astype(int)
            
            # Filter the raw EMG data
            for i in range(emg_data.shape[1]):
                emg_data[:, i] = signal.filtfilt(b_notch, a_notch, emg_data[:, i])
                emg_data[:, i] = signal.filtfilt(b_band, a_band, emg_data[:, i])
            
            # Find boundaries of "pure" chunks
            boundaries = np.where(np.diff(labels) != 0)[0] + 1
            chunk_starts = np.concatenate(([0], boundaries))
            chunk_stops = np.concatenate((boundaries, [len(labels)]))
            
            print(f"   ... Found {len(chunk_starts)} 'pure' movement/rest chunks.")
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                if start >= stop: continue
                
                chunk_emg = emg_data[start:stop]
                chunk_glove = glove_data[start:stop]
                chunk_label = labels[start] # All labels in this chunk are the same
                
                n_samples = len(chunk_emg)
                # Use non-overlapping windows, discard partial remainder
                n_windows = n_samples // Config.WINDOW_SAMPLES 
                remainder = n_samples % Config.WINDOW_SAMPLES
                
                if n_windows == 0:
                    print(f"   LOG: Discarding chunk (rows {start}-{stop}, label {chunk_label}) - too short ({n_samples} samples) for one window.")
                    continue
                
                # *** YOUR REQUESTED LOGGING ***
                if remainder > 0:
                    print(f"   LOG: Discarding last {remainder} samples from chunk (rows {start}-{stop}, label {chunk_label}) to ensure pure windows.")
                
                # Process all *full* non-overlapping windows in this chunk
                for i in range(n_windows):
                    win_start = start + (i * Config.WINDOW_SAMPLES)
                    win_stop = win_start + Config.WINDOW_SAMPLES
                    
                    # We reference the *original* dataframe slices
                    emg_window = emg_data[win_start:win_stop]
                    glove_window = glove_data[win_start:win_stop]
                    
                    features_48 = calculate_hudgins_features(emg_window)
                    target_22 = glove_window[-1, :].astype(np.float32)
                    
                    # Create a dictionary for this 10Hz row
                    row_data = {
                        'group_id': group_id_counter, # This ID marks the 'pure' chunk
                        'restimulus': chunk_label
                    }
                    # Add features
                    for f_idx in range(48):
                        row_data[f'feat_{f_idx+1}'] = features_48[f_idx] # Use 1-indexing
                    # Add targets
                    for t_idx in range(22):
                        row_data[f'glove_{t_idx+1}'] = target_22[t_idx] # Use 1-indexing
                        
                    all_rows.append(row_data)
                
                group_id_counter += 1 # Increment for each new, pure chunk
                
        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process file {fp}. Error: {e}")
            continue
            
    print(f"\n... 10Hz data extraction complete. Found {len(all_rows)} total 10Hz samples.")
    return all_rows

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING CONVERSION TO 10Hz CSV (PHASE 2a)")
    print("=============================================================")
    
    # ==================== EDIT THIS LIST ====================
    # Paste your file paths here for the test run.
    # Use 'r' before the string to handle Windows backslashes '\'.
    FILES_TO_PROCESS = [
        r"D:/DB\\DB2\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S12_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S13_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S14_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S15_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S16_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S17_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S18_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S19_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S20_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S21_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S22_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S23_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S24_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S25_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S26_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S27_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S28_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S29_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S30_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S31_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S32_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S33_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S34_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S35_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S36_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S37_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S38_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S39_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S40_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB2\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S12_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S13_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S14_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S15_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S16_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S17_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S18_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S19_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S20_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S21_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S22_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S23_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S24_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S25_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S26_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S27_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S28_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S29_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S30_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S31_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S32_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S33_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S34_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S35_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S36_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S37_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S38_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S39_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S40_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S9_E2_A1.csv",
        r"D:/DB\\DB3\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB3\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S9_E2_A1.csv"
    ]
    # ========================================================
    
    print(f"Starting processing for {len(FILES_TO_PROCESS)} manually specified files...")

    # --- STEP 1: Process 2000Hz -> 10Hz ---
    list_of_rows = build_10hz_list_of_rows(FILES_TO_PROCESS)
    
    if not list_of_rows:
        print("❌ FATAL: No 10Hz data could be extracted. Check file paths and data.")
        return

    # --- STEP 2: Convert list of dicts to DataFrame and save ---
    print(f"\n... Converting to DataFrame and saving to {Config.OUTPUT_CSV_PATH}...")
    try:
        final_df = pd.DataFrame(list_of_rows)
        # We will use CSV as you requested, not Parquet.
        final_df.to_csv(Config.OUTPUT_CSV_PATH, index=False)
        print(f"   ... Saved as {Config.OUTPUT_CSV_PATH}.")

    except Exception as e:
        print(f"❌ FATAL: Failed to save DataFrame to CSV. Error: {e}")
        return

    print("\n" + "=" * 60)
    print(f"🎉 CONVERSION TO 10Hz CSV COMPLETE!")
    print(f"   Total 10Hz samples created: {len(list_of_rows):,}")
    print(f"   Output file: {Config.OUTPUT_CSV_PATH}")
    print("\n   ✅ YOU CAN NOW OPEN THIS CSV FILE TO SEE THE DATA.")
    print("\nNext step: Run the 'Phase 2b' analysis script on this new file.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v11.0\convert_to_tfrecord_v2.py
**Path:** `C:\stage\stage\v11.0\convert_to_tfrecord_v2.py`

```python
# Save this as convert_to_tfrecord_v2.py
# This is the FINAL data conversion script (Phase 2c).
# It converts the 10Hz CSV into a balanced, normalized, sequential TFRecord dataset.
# *** VERSION 2.1: Includes the float32 cast fix ***

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    DATA_CSV_FILE = "10hz_feature_data.csv"
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    OUTPUT_DIR = "tfrecord_dataset_v2"
    FEATURE_SCALER_FILE = "feature_scaler.pkl"
    TARGET_SCALER_FILE = "target_scaler.pkl" 
    
    # MANDATORY FIX: Explicitly define the correct dimensions
    NUM_TARGETS = 12 # 12 Virtual Motors
    
    NUM_MOVEMENT_CLUSTERS = 12 # 12 movement archetypes
    SEQUENCE_LENGTH = 50 # 5 seconds
    
    SEQUENCES_PER_SHARD = 500
    VALIDATION_SPLIT = 0.15 

# --- 🛠️ HELPER FUNCTIONS ---

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the 10Hz CSV data."""
    print(f"Loading 10Hz data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        raise FileNotFoundError
    
    df = pd.read_csv(file_path)
    print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
    return df

def create_virtual_motors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Creates the 12 Virtual Motor columns based on the report's final grouping."""
    print("--- 1. Creating 12 Virtual Motors (Targets) ---")
    
    # MANDATORY FIX: Updated to 12-motor grouping (must match analyze_10hz_data.py)
    VIRTUAL_MOTORS: Dict[str, List[str]] = {
        # A. Primary Flexion/Abduction (7 Motors)
        "motor_thumb_flex": ["glove_3", "glove_4"],
        "motor_index_flex": ["glove_5", "glove_6", "glove_7"],
        "motor_middle_flex": ["glove_8", "glove_9", "glove_10"],
        "motor_ring_flex": ["glove_12", "glove_13", "glove_14"],
        "motor_pinky_flex": ["glove_16", "glove_17", "glove_18"],
        "motor_thumb_abduct": ["glove_2"],
        "motor_wrist_flex_ext": ["glove_21"], 
        
        # B. Secondary Adduction/Waving (5 Motors - Enhanced)
        "motor_index_adduct": ["glove_11"],
        "motor_middle_adduct": ["glove_15"],
        "motor_ring_adduct": ["glove_19"],
        "motor_pinky_adduct": ["glove_20"],
        "motor_hand_waving": ["glove_1", "glove_22"]
    }
    
    # Check for dimensional mismatch
    if len(VIRTUAL_MOTORS) != Config.NUM_TARGETS:
         print(f"❌ ERROR: Motor count ({len(VIRTUAL_MOTORS)}) does not match Config.NUM_TARGETS ({Config.NUM_TARGETS}).")
         raise ValueError("Dimensional Mismatch in VIRTUAL_MOTORS setup.")
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        df[motor_name] = df[sensor_list].mean(axis=1)
        motor_cols.append(motor_name)
    
    feature_cols = [f'feat_{i+1}' for i in range(48)]
    return df, feature_cols, motor_cols

def normalize_data(df: pd.DataFrame, f_cols: List[str], m_cols: List[str], norm_params_file: str, out_dir: str) -> pd.DataFrame:
    """
    Applies Min-Max to motors (targets) and Z-Score to features (inputs).
    This is applied to the *entire* 10Hz dataframe.
    """
    print("--- 2. Normalizing All 10Hz Data ---")
    
    # --- A. Normalize Motors (Targets) ---
    # The print statement is now correct: it prints the actual number of motors being processed
    print(f"   ... Applying Min-Max (0-1) to {len(m_cols)} motors...") 
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    for motor in m_cols:
        p = params[motor]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        if motor_range == 0: motor_range = 1.0 
        
        df[motor] = (df[motor] - motor_min) / motor_range
        df[motor] = np.clip(df[motor], 0.0, 1.0)
    
    joblib.dump(params, os.path.join(out_dir, Config.TARGET_SCALER_FILE))

    # --- B. Normalize Features (Inputs) ---
    print(f"   ... Applying Z-Score to {len(f_cols)} features...")
    feature_data = df[f_cols].values
    feature_scaler = StandardScaler()
    
    # Fit and transform the features
    # Save as float32 to prevent type mismatch error
    features_scaled = feature_scaler.fit_transform(feature_data).astype(np.float32)
    
    joblib.dump(feature_scaler, os.path.join(out_dir, Config.FEATURE_SCALER_FILE))
    
    # Put the normalized features back into the DataFrame
    df[f_cols] = features_scaled
    print("   ... Normalization complete. Scalers saved.")
    return df

def create_all_sequences(df: pd.DataFrame, f_cols: List[str], m_cols: List[str]) -> pd.DataFrame:
    """
    Builds all possible sequences from the full, normalized 10Hz dataframe.
    """
    print("--- 3. Creating All Possible Sequences ---")
    
    all_sequences = [] # This will be a list of dicts
    
    # Group by the 'group_id' to prevent sequences from crossing boundaries
    grouped = df.groupby('group_id')
    
    for group_id, group_df in tqdm(grouped, desc="   ... Processing groups", unit="group"):
        if len(group_df) < Config.SEQUENCE_LENGTH:
            continue # This chunk is too short
            
        # ==================== THE FIX (Alternative Location) ====================
        # We ensure features are float32 before sequencing.
        # This is the one-line fix.
        features = group_df[f_cols].values.astype(np.float32) 
        # ======================================================================
        
        targets = group_df[m_cols].values.astype(np.float32)
        labels = group_df['restimulus'].values
        
        # Slide our 50-sample window over this pure chunk
        for i in range(len(group_df) - Config.SEQUENCE_LENGTH + 1):
            
            seq_label = labels[i + Config.SEQUENCE_LENGTH - 1]
            seq_features = features[i : i + Config.SEQUENCE_LENGTH]
            seq_targets = targets[i : i + Config.SEQUENCE_LENGTH]
            
            all_sequences.append({
                "features": seq_features,
                "targets": seq_targets,
                "label": seq_label
            })

    print(f"   ... Found {len(all_sequences):,} total valid sequences.")
    return pd.DataFrame(all_sequences)

def balance_sequences(seq_df: pd.DataFrame, m_cols: List[str]) -> pd.DataFrame:
    """
    Performs stratified sampling on the SEQUENCES, not the 10Hz data.
    """
    print("--- 4. Balancing Sequences (Stratified Sampling) ---")
    
    rest_seq_df = seq_df[seq_df['label'] == 0].copy()
    move_seq_df = seq_df[seq_df['label'] != 0].copy()
    
    if rest_seq_df.empty or move_seq_df.empty:
        print("   ⚠️ Warning: Missing rest or movement sequences. Balancing may be skewed.")
        return seq_df

    print(f"   ... Found {len(rest_seq_df)} 'rest' sequences and {len(move_seq_df)} 'movement' sequences.")
    
    # The number of movement clusters must match the number of movement motors (12)
    # The stratified sampling will now reflect the 13 classes (12 movements + 1 rest)
    print(f"   ... Finding {Config.NUM_MOVEMENT_CLUSTERS} movement archetypes (K-Means)...") 
    
    avg_poses = np.array([seq.mean(axis=0) for seq in move_seq_df['targets']])
    
    kmeans = KMeans(n_clusters=Config.NUM_MOVEMENT_CLUSTERS, random_state=42, n_init=10)
    move_seq_df['cluster'] = kmeans.fit_predict(avg_poses)
    
    all_balanced_dfs = []
    
    cluster_counts = move_seq_df['cluster'].value_counts()
    
    min_move_count = cluster_counts.min()
    print(f"   ... Minority movement cluster has {min_move_count} sequences.")
    
    for i in range(Config.NUM_MOVEMENT_CLUSTERS):
        cluster_df = move_seq_df[move_seq_df['cluster'] == i]
        all_balanced_dfs.append(cluster_df.sample(n=min_move_count, random_state=42, replace=False))
        
    # The rest class is balanced against the size of the smallest movement cluster
    rest_sample_size = min(len(rest_seq_df), min_move_count) 
    print(f"   ... Sampling {rest_sample_size} 'rest' sequences.")
    all_balanced_dfs.append(rest_seq_df.sample(n=rest_sample_size, random_state=42, replace=False))
    
    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} sequences.")
    return balanced_df

def write_sequences_to_tfrecord(seq_df: pd.DataFrame, output_dir: str):
    """
    Saves the final, balanced DataFrame of sequences to TFRecord files.
    """
    print("--- 5. Saving Sequences to TFRecord ---")
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    val_size = int(len(seq_df) * Config.VALIDATION_SPLIT)
    val_df = seq_df.iloc[:val_size]
    train_df = seq_df.iloc[val_size:]
    
    print(f"   ... Splitting into {len(train_df)} train and {len(val_df)} val sequences.")
    
    shard_counts = {'train': 0, 'val': 0}

    # --- Write Training Data ---
    with tqdm(total=len(train_df), desc="   ... Writing train shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(train_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(train_dir, f'data_{shard_counts["train"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['train'] += 1
            
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    # --- Write Validation Data ---
    with tqdm(total=len(val_df), desc="   ... Writing val shards", unit="seq") as pbar:
        writer = None
        for i, (idx, row) in enumerate(val_df.iterrows()):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(val_dir, f'data_{shard_counts["val"]:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_counts['val'] += 1
                
            feature = {
                'features': _bytes_feature(tf.io.serialize_tensor(row['features'])),
                'targets': _bytes_feature(tf.io.serialize_tensor(row['targets']))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            pbar.update(1)
        if writer: writer.close()

    print(f"\n   ... Wrote {len(train_df):,} training sequences to {shard_counts['train']} shards.")
    print(f"   ... Wrote {len(val_df):,} validation sequences to {shard_counts['val']} shards.")
    return len(seq_df)

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING FINAL CONVERSION (PHASE 2c) - V2 (Corrected Order)")
    print("=============================================================")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = load_data(Config.DATA_CSV_FILE)
    if df is None: return

    df, feature_cols, motor_cols = create_virtual_motors(df)
    
    df = normalize_data(df, feature_cols, motor_cols, Config.NORMALIZATION_PARAMS_FILE, Config.OUTPUT_DIR)

    all_sequences_df = create_all_sequences(df, feature_cols, motor_cols)
    
    balanced_seq_df = balance_sequences(all_sequences_df, motor_cols)
    
    total_seqs = write_sequences_to_tfrecord(balanced_seq_df, Config.OUTPUT_DIR)

    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_unbalanced_sequences": len(all_sequences_df),
        "total_balanced_sequences": total_seqs,
        "feature_scaler": Config.FEATURE_SCALER_FILE,
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 FINAL CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {total_seqs:,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("   Scalers saved: feature_scaler.pkl, target_scaler.pkl")
    print("\n   ✅ THE DATASET IS NOW 100% READY FOR TRAINING.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v11.0\train_sequential_v2.py
**Path:** `C:\stage\stage\v11.0\train_sequential_v2.py`

```python
# Save this as train_sequential_v2.py
# This script trains our new sequential models (v4, v5)
# on the final TFRecord dataset.

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import glob
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Data ---
    TRAIN_TFRECORD_PATH = "tfrecord_dataset_v2/train/*.tfrecord"
    VAL_TFRECORD_PATH = "tfrecord_dataset_v2/val/*.tfrecord"
    
    # --- Model Hyperparameters (from our data conversion) ---
    SEQUENCE_LENGTH = 50 # 50 steps (5 seconds)
    NUM_FEATURES = 48    # 48 Hudgins' features (12 channels * 4)
    NUM_TARGETS = 12      # 7 Virtual Motors
    
    # --- Training ---
    BATCH_SIZE = 16 # Small dataset, so small batch size
    EPOCHS = 150    # Let's train for a while; EarlyStopping will find the best
    
# --- 1. DATA LOADER ---

def parse_sequence_tfrecord(example_proto):
    """
    Parses the TFRecord examples created by convert_to_tfrecord_v2.py.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'targets': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    features = tf.io.parse_tensor(parsed['features'], out_type=tf.float32)
    targets = tf.io.parse_tensor(parsed['targets'], out_type=tf.float32)
    
    # Set the shapes explicitly
    features = tf.reshape(features, [Config.SEQUENCE_LENGTH, Config.NUM_FEATURES])
    targets = tf.reshape(targets, [Config.SEQUENCE_LENGTH, Config.NUM_TARGETS])
    
    return features, targets

def create_dataset(tfrecord_pattern: str, batch_size: int, shuffle=True):
    """Creates a high-performance TensorFlow dataset from TFRecord files."""
    
    file_list = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    dataset = file_list.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        # Shuffle *before* mapping for better performance
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.map(parse_sequence_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    if shuffle:
        # Repeat indefinitely for training
        dataset = dataset.repeat()
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. MODEL ARCHITECTURES ---

def create_v4_lstm_baseline(model_name="v4_lstm_baseline"):
    """
    Model v4: A simple LSTM-only model. This is our sequential baseline.
    This is a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # LSTM layer that processes the sequence.
    # return_sequences=True is ESSENTIAL for Seq2Seq. It outputs
    # a prediction for *every* of the 50 time steps.
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    
    # A final Dense layer, wrapped in TimeDistributed.
    # This applies the *same* dense layer to *each* of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear') # 'linear' for regression
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse', # Mean Squared Error is standard for regression
        metrics=['mae'] # Mean Absolute Error is easier to interpret
    )
    return model

def create_v5_cnn_lstm(model_name="v5_cnn_lstm"):
    """
    Model v5: The SOTA-informed RCNN (CNN-LSTM) Hybrid.
    This is also a Seq2Seq model.
    """
    inputs = layers.Input(shape=(Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), name="emg_input_sequence")
    
    # 1. CNN Feature Extraction Block
    # 1D Conv layers to find local patterns (motifs) in the 50-step sequence.
    # We use 'causal' padding to prevent the model from "cheating" by
    # looking at future time steps to predict the present.
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(x)
    
    # 2. LSTM Temporal Processing Block
    # The LSTM layer learns the long-term relationships between the
    # features extracted by the CNN.
    # We must use return_sequences=True.
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.4)(x)
    
    # 3. TimeDistributed Prediction Head
    # Applies a Dense layer to each of the 50 time steps.
    outputs = layers.TimeDistributed(
        layers.Dense(Config.NUM_TARGETS, activation='linear')
    , name="motor_output_sequence")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

# --- 3. TRAINING & EVALUATION ---

def setup_experiment_directory(experiment_name):
    """Creates a unique directory for storing all artifacts of a training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def train_model(model_creator, model_name, train_dataset, val_dataset, train_steps, val_steps):
    """The main function to train a single model and save all results."""
    print(f"\n{'='*60}\n🚀 STARTING TRAINING FOR {model_name}\n{'='*60}")
    
    exp_dir = setup_experiment_directory(f"series_{model_name}")
    model = model_creator()
    model.summary()
    
    # Callbacks
    best_model_path = os.path.join(exp_dir, "models", f"best_{model_name}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1), # 25 patience
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
    ]

    print(f"\nTraining {model_name}...")
    start_time = datetime.datetime.now()

    history = model.fit(
        train_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    
    # Load best model and get its validation MAE
    best_val_mae = min(history.history['val_mae'])
    
    # Save results
    results = {
        'model_version': model_name,
        'best_validation_mae': float(best_val_mae),
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_mae']) + 1),
        'training_time_seconds': training_time.total_seconds(),
        'model_architecture': json.loads(model.to_json())
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create and save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE'); plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f'{model_name}_training_history.png'))
    plt.close()

    print(f"\n🎯 {model_name} - FINAL RESULTS:")
    print(f"   Best Validation MAE: {results['best_validation_mae']:.4f}")
    print(f"   Training Time: {str(training_time).split('.')[0]}")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"💾 Results and artifacts saved to: {exp_dir}")
    return results

# --- 4. MAIN EXECUTION ---
def main():
    print("🚀 STARTING SEQUENTIAL MODEL TRAINING (PHASE 2d)")
    print("=============================================================")
    
    # Find the data we just created
    train_files = glob.glob(Config.TRAIN_TFRECORD_PATH)
    val_files = glob.glob(Config.VAL_TFRECORD_PATH)
    
    if not train_files or not val_files:
        print(f"❌ FATAL: Could not find TFRecord files.")
        print(f"   Checked for train: {Config.TRAIN_TFRECORD_PATH}")
        print(f"   Checked for val: {Config.VAL_TFRECORD_PATH}")
        return

    # Get total number of samples from our log file
    # This is a bit of a hack, but it's the easiest way
    # In a real-world scenario, we'd read this from metadata.json
    total_train_seqs = 405
    total_val_seqs = 71
    
    # Calculate steps per epoch
    train_steps = total_train_seqs // Config.BATCH_SIZE
    val_steps = total_val_seqs // Config.BATCH_SIZE
    
    # Ensure steps are at least 1
    if train_steps == 0: train_steps = 1
    if val_steps == 0: val_steps = 1

    print(f"   Found {total_train_seqs} train sequences and {total_val_seqs} val sequences.")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Steps/Epoch: {train_steps}")
    print(f"   Validation Steps/Epoch: {val_steps}")

    # Create the datasets
    train_dataset = create_dataset(Config.TRAIN_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(Config.VAL_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=False)

    # Define models to train
    models_to_train = {
        'v4_lstm_baseline': create_v4_lstm_baseline,
        'v5_cnn_lstm': create_v5_cnn_lstm,
    }
    
    all_results = []
    
    for model_name, model_creator in models_to_train.items():
        results = train_model(
            model_creator, 
            model_name, 
            train_dataset, 
            val_dataset, 
            train_steps, 
            val_steps
        )
        all_results.append(results)

    print(f"\n{'='*70}\n📊 SEQUENTIAL EXPERIMENTS COMPLETE 📊\n{'='*70}")
    
    # Print summary table
    print(f"{'Model':<20} {'Best Val MAE':<15} {'Best Epoch':<12}")
    print('-'*70)
    for res in sorted(all_results, key=lambda x: x['best_validation_mae']):
        print(f"{res['model_version']:<20} {res['best_validation_mae']:.4f}{'':<11} {res['best_epoch']:<12}")
    
    best_model_results = min(all_results, key=lambda x: x['best_validation_mae'])
    print(f"\n🏆 BEST SEQUENTIAL MODEL: {best_model_results['model_version']} (Val MAE: {best_model_results['best_validation_mae']:.4f})")
    print(f"\n   Recall: Our 'Point-in-Time' (v2) model had a Test MAE of 0.1610")
    print(f"   ... We will need to run a final test, but this validation MAE is our new benchmark.")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
```

---

## File: v12.0\convert_to_spectrogram.py
**Path:** `C:\stage\stage\v12.0\convert_to_spectrogram.py`

```python
# Save this as convert_to_spectrogram.py
# This script converts raw 2000Hz CSVs into a sequential
# dataset of 2D Spectrograms (images) and normalized 7-motor targets.

import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, List, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Input Files ---
    NORMALIZATION_PARAMS_FILE = "motor_normalization_params.json"
    
    # --- Output ---
    OUTPUT_DIR = "tfrecord_dataset_v3_spectrogram"
    TARGET_SCALER_FILE = "target_scaler.pkl" # We'll re-save our Min/Max scaler
    
    # --- Signal Processing ---
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # --- Sequencing ---
    # We want to match the 5-second movement, as per the dataset guide
    SEQUENCE_S = 5.0
    SEQUENCE_SAMPLES = int(FS * SEQUENCE_S) # 5.0s * 2000 Hz = 10,000 samples
    
    # --- Spectrogram (STFT) Parameters ---
    # We'll use the same 100ms (200-sample) window as our 10Hz features
    STFT_WINDOW_SAMPLES = 200
    # We'll overlap windows by 50% for a smoother image
    STFT_OVERLAP_SAMPLES = 100 
    
    # --- Balancing & TFRecord ---
    SEQUENCES_PER_SHARD = 200 # Spectrograms are large
    VALIDATION_SPLIT = 0.20 # 20% for validation

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def create_virtual_motors(glove_data: np.ndarray) -> np.ndarray:
    """
    Converts a (N, 22) glove array into a (N, 7) motor array.
    """
    # This is our known, fact-based mapping
    VIRTUAL_MOTORS: Dict[str, List[int]] = {
        "motor_thumb_flex": [2, 3], # glove_3, glove_4 (0-indexed)
        "motor_index_flex": [5, 6, 7],
        "motor_middle_flex": [9, 10, 11],
        "motor_ring_flex": [13, 14, 15],
        "motor_pinky_flex": [17, 18, 19],
        "motor_thumb_abduct": [1], # glove_2
        "motor_wrist_flex": [20]  # glove_21
    }
    
    # Create the new 7-motor array
    motor_data = np.zeros((glove_data.shape[0], 7), dtype=np.float32)
    for i, (motor_name, sensor_indices) in enumerate(VIRTUAL_MOTORS.items()):
        motor_data[:, i] = glove_data[:, sensor_indices].mean(axis=1)
        
    return motor_data

def normalize_motors(motor_data: np.ndarray, norm_params_file: str) -> np.ndarray:
    """
    Applies Min-Max (0-1) normalization to the 7 motor columns.
    """
    with open(norm_params_file, 'r') as f:
        params = json.load(f)
    
    normalized_data = np.zeros_like(motor_data)
    
    # This list *must* match the order in VIRTUAL_MOTORS
    motor_names = [
        "motor_thumb_flex", "motor_index_flex", "motor_middle_flex",
        "motor_ring_flex", "motor_pinky_flex", "motor_thumb_abduct", "motor_wrist_flex"
    ]
    
    for i, motor_name in enumerate(motor_names):
        p = params[motor_name]
        motor_min = p["Rest_Value_Min"]
        motor_max = p["Move_Value_Max_99pct"]
        motor_range = motor_max - motor_min
        if motor_range == 0: motor_range = 1.0 
        
        normalized_data[:, i] = (motor_data[:, i] - motor_min) / motor_range
    
    # Clip to our target 0-1 range
    return np.clip(normalized_data, 0.0, 1.0).astype(np.float32)

def compute_spectrograms(emg_chunk: np.ndarray) -> np.ndarray:
    """
    Converts a (10000, 12) EMG chunk into a (12, 101, 99) spectrogram tensor.
    """
    n_channels = emg_chunk.shape[1]
    all_spectrograms = []
    
    for i in range(n_channels):
        f, t, Sxx = signal.stft(
            emg_chunk[:, i],
            fs=Config.FS,
            nperseg=Config.STFT_WINDOW_SAMPLES,
            noverlap=Config.STFT_OVERLAP_SAMPLES
        )
        # Sxx shape is (n_freqs, n_times), e.g., (101, 99)
        # We use np.abs() to get magnitude and add a small epsilon for log stability
        Sxx_mag = np.abs(Sxx)
        all_spectrograms.append(Sxx_mag)

    # Stack into a (12, 101, 99) tensor
    # We use log-magnitude, which is standard for audio/EMG "images"
    # This compresses the dynamic range
    spectrogram_stack = np.stack(all_spectrograms, axis=0)
    return np.log(spectrogram_stack + 1e-6).astype(np.float32)

def make_tfexample(spectrogram: np.ndarray, target_vector: np.ndarray, label: int) -> tf.train.Example:
    """Serializes a (12, 101, 99) spectrogram and (7,) target vector."""
    
    spectrogram_bytes = tf.io.serialize_tensor(spectrogram).numpy()
    target_bytes = tf.io.serialize_tensor(target_vector).numpy()
    
    feature_dict = {
        'spectrogram': tf.train.Feature(bytes_list=tf.train.BytesList(value=[spectrogram_bytes])),
        'target_vector': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING SPECTROGRAM CONVERSION (PHASE 3a)")
    print("=============================================================")
    
    # ==================== EDIT THIS LIST ====================
    # This should be the FULL list of your raw data files
    FILES_TO_PROCESS = [
        r"D:/DB\\DB2\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S12_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S13_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S14_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S15_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S16_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S17_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S18_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S19_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S20_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S21_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S22_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S23_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S24_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S25_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S26_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S27_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S28_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S29_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S30_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S31_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S32_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S33_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S34_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S35_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S36_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S37_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S38_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S39_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S40_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB2\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB2\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S12_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S13_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S14_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S15_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S16_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S17_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S18_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S19_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S20_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S21_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S22_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S23_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S24_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S25_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S26_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S27_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S28_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S29_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S30_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S31_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S32_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S33_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S34_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S35_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S36_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S37_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S38_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S39_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S40_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB2\\E2\\S9_E2_A1.csv",
        r"D:/DB\\DB3\\E1\\S10_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S11_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S1_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S2_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S3_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S4_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S5_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S6_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S7_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S8_E1_A1.csv",
        r"D:/DB\\DB3\\E1\\S9_E1_A1.csv",
        r"D:/DB\\DB3\\E2\\S10_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S11_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S1_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S2_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S3_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S4_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S5_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S6_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S7_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S8_E2_A1.csv",
        r"D:/DB\\DB3\\E2\\S9_E2_A1.csv"
    ]
    # ========================================================
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_sequences = [] # This will be a list of (spectrogram, target_vector, label)
    
    print(f"Starting processing for {len(FILES_TO_PROCESS)} files...")
    
    for fp in tqdm(FILES_TO_PROCESS, desc="Processing Files", unit="file"):
        try:
            use_cols = [f"emg_{i}" for i in range(1, 13)] + [f"glove_{i}" for i in range(1, 23)] + ["restimulus"]
            df = pd.read_csv(fp, usecols=lambda c: c in use_cols)
            
            emg_cols = [f"emg_{i}" for i in range(1, 13)]
            glove_cols = [f"glove_{i}" for i in range(1, 23)]
            
            emg_data = df[emg_cols].values
            glove_data = df[glove_cols].values
            labels = df['restimulus'].values.astype(int)
            
            # Filter the raw EMG data
            for i in range(emg_data.shape[1]):
                emg_data[:, i] = signal.filtfilt(b_notch, a_notch, emg_data[:, i])
                emg_data[:, i] = signal.filtfilt(b_band, a_band, emg_data[:, i])
            
            # Find boundaries of "pure" chunks
            boundaries = np.where(np.diff(labels) != 0)[0] + 1
            chunk_starts = np.concatenate(([0], boundaries))
            chunk_stops = np.concatenate((boundaries, [len(labels)]))
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                n_samples = stop - start
                chunk_label = labels[start]
                
                # How many 5-second (10,000-sample) sequences can we make from this chunk?
                n_sequences = n_samples // Config.SEQUENCE_SAMPLES
                
                if n_sequences == 0: continue
                    
                # Process all *full* non-overlapping 5-second sequences
                for i in range(n_sequences):
                    seq_start = start + (i * Config.SEQUENCE_SAMPLES)
                    seq_stop = seq_start + Config.SEQUENCE_SAMPLES
                    
                    emg_chunk = emg_data[seq_start:seq_stop]
                    glove_chunk = glove_data[seq_start:seq_stop]
                    
                    # 1. Create the Spectrogram "Image"
                    spectrogram = compute_spectrograms(emg_chunk)
                    
                    # 2. Create the Target Vector
                    # We only care about the hand pose at the *end* of the 5s movement
                    # We'll average the *last 100ms* of glove data for a stable target
                    target_glove_window = glove_chunk[-Config.STFT_WINDOW_SAMPLES:]
                    # Convert (200, 22) -> (22,) by averaging
                    target_glove_vector = target_glove_window.mean(axis=0)
                    # Convert (22,) -> (7,) by grouping
                    target_motor_vector = create_virtual_motors(target_glove_vector.reshape(1, -1))[0]
                    
                    # Add this (input, output, label) pair to our big list
                    all_sequences.append((spectrogram, target_motor_vector, chunk_label))

        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process file {fp}. Error: {e}")
            continue

    print(f"\n... Found {len(all_sequences)} total 5-second sequences.")
    
    # --- 4. Balance the Sequences ---
    print("--- 2. Balancing Sequences (Stratified Sampling) ---")
    
    # Put our sequences into a DataFrame to balance
    seq_df = pd.DataFrame(all_sequences, columns=['spectrogram', 'target_vector', 'label'])
    
    rest_df = seq_df[seq_df['label'] == 0]
    move_df = seq_df[seq_df['label'] != 0]
    
    if rest_df.empty or move_df.empty:
        print("❌ FATAL: Dataset is missing 'rest' or 'movement' sequences.")
        return

    print(f"   ... Found {len(rest_df)} 'rest' sequences and {len(move_df)} 'movement' sequences.")
    
    # Find minority class
    min_move_count = move_df.groupby('label').size().min()
    min_class_count = min(len(rest_df), min_move_count)
    
    print(f"   ... Minority class has {min_class_count} sequences. Balancing to this number.")
    
    all_balanced_dfs = []
    
    # Sample from "Rest"
    all_balanced_dfs.append(rest_df.sample(n=min_class_count, random_state=42))
    
    # Sample from each movement
    movement_labels = move_df['label'].unique()
    for label in movement_labels:
        move_label_df = move_df[move_df['label'] == label]
        all_balanced_dfs.append(move_label_df.sample(n=min_class_count, random_state=42, replace=True)) # Oversample if needed

    balanced_df = pd.concat(all_balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   ... Balancing complete. New dataset size: {len(balanced_df)} sequences.")

    # --- 5. Normalize Targets and Save TFRecords ---
    print("--- 3. Normalizing Targets and Saving to TFRecord ---")
    
    # Stack all target vectors for normalization
    target_vectors_stacked = np.stack(balanced_df['target_vector'].values)
    
    # Normalize the 7-motor targets using our JSON file
    normalized_targets = normalize_motors(target_vectors_stacked, Config.NORMALIZATION_PARAMS_FILE)
    
    # Put them back in the DataFrame
    balanced_df['target_vector'] = list(normalized_targets)
    
    # Save the normalization params file we used
    joblib.dump(json.load(open(Config.NORMALIZATION_PARAMS_FILE)), os.path.join(Config.OUTPUT_DIR, Config.TARGET_SCALER_FILE))

    # --- 6. Write TFRecords ---
    train_dir = os.path.join(Config.OUTPUT_DIR, "train")
    val_dir = os.path.join(Config.OUTPUT_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    val_size = int(len(balanced_df) * Config.VALIDATION_SPLIT)
    val_df = balanced_df.iloc[:val_size]
    train_df = balanced_df.iloc[val_size:]
    
    print(f"   ... Splitting into {len(train_df)} train and {len(val_df)} val sequences.")
    
    # --- Write Shards ---
    for split_name, df_split in [('train', train_df), ('val', val_df)]:
        shard_count = 0
        writer = None
        for i, (idx, row) in enumerate(tqdm(df_split.iterrows(), total=len(df_split), desc=f"   ... Writing {split_name} shards")):
            if i % Config.SEQUENCES_PER_SHARD == 0:
                if writer: writer.close()
                path = os.path.join(Config.OUTPUT_DIR, split_name, f'data_{shard_count:04d}.tfrecord')
                writer = tf.io.TFRecordWriter(path)
                shard_count += 1
            
            example = make_tfexample(row['spectrogram'], row['target_vector'], row['label'])
            writer.write(example.SerializeToString())
        if writer: writer.close()
        print(f"   ... Wrote {len(df_split)} {split_name} sequences to {shard_count} shards.")

    # --- FINAL METADATA ---
    metadata = {
        "params": {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k.isupper()},
        "total_balanced_sequences": len(balanced_df),
        "target_scaler": Config.TARGET_SCALER_FILE
    }
    meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 SPECTROGRAM CONVERSION COMPLETE!")
    print(f"   Total balanced sequences: {len(balanced_df):,}")
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print("\n   ✅ THE DATASET IS NOW READY FOR TRAINING (PHASE 3b).")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v12.0\plot_spectrogram_example.py
**Path:** `C:\stage\stage\v12.0\plot_spectrogram_example.py`

```python
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
```

---

## File: v12.0\train_spectrogram_v3.py
**Path:** `C:\stage\stage\v12.0\train_spectrogram_v3.py`

```python
# Save this as train_spectrogram_v3.py
# This script trains our new 2D-CNN model (v6)
# on the final spectrogram dataset (tfrecord_dataset_v3_spectrogram).

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import glob
import random

# --- ⚙️ CONFIGURATION ---
class Config:
    # --- Data ---
    TRAIN_TFRECORD_PATH = "tfrecord_dataset_v3_spectrogram/train/*.tfrecord"
    VAL_TFRECORD_PATH = "tfrecord_dataset_v3_spectrogram/val/*.tfrecord"
    
    # --- Model Hyperparameters (from our data conversion) ---
    # These shapes are based on the STFT parameters in the conversion script
    # Input shape: (n_freqs, n_times, n_channels)
    # (101, 99, 12)
    INPUT_SHAPE = (101, 101, 12)
    NUM_TARGETS = 7 # 7 Virtual Motors
    
    # --- Training ---
    BATCH_SIZE = 16 
    EPOCHS = 150    
    
    # --- From Log File ---
    TOTAL_TRAIN_SEQS = 1903
    TOTAL_VAL_SEQS = 475

# --- 1. DATA LOADER ---

def parse_spectrogram_tfrecord(example_proto):
    """
    Parses the TFRecord examples created by convert_to_spectrogram.py.
    """
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        'target_vector': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    # Spectrogram was saved as (12, 101, 99) -> (C, H, W)
    spectrogram = tf.io.parse_tensor(parsed['spectrogram'], out_type=tf.float32)
    
    # Target was saved as (7,)
    target = tf.io.parse_tensor(parsed['target_vector'], out_type=tf.float32)
    
    # *** CRITICAL: Transpose the spectrogram ***
    # Keras Conv2D expects (Height, Width, Channels)
    # We convert (C, H, W) -> (H, W, C)
    # (12, 101, 99) -> (101, 99, 12)
    spectrogram = tf.transpose(spectrogram, (1, 2, 0))
    
    # Set the shapes explicitly
    spectrogram = tf.reshape(spectrogram, Config.INPUT_SHAPE)
    target = tf.reshape(target, [Config.NUM_TARGETS])
    
    return spectrogram, target

def create_dataset(tfrecord_pattern: str, batch_size: int, shuffle=True):
    """Creates a high-performance TensorFlow dataset from TFRecord files."""
    
    file_list = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    dataset = file_list.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000) # Buffer size matches our dataset
        
    dataset = dataset.map(parse_spectrogram_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    if shuffle:
        dataset = dataset.repeat() # Repeat indefinitely for training
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. MODEL ARCHITECTURE (v6) ---

def create_v6_cnn_spectrogram(model_name="v6_cnn_spectrogram"):
    """
    Model v6: A 2D-CNN for End-to-End Spectrogram Regression.
    This is a Seq2Vec model: (5-sec image) -> (1 final prediction)
    """
    inputs = layers.Input(shape=Config.INPUT_SHAPE, name="emg_spectrogram_input")
    
    # 1. CNN Feature Extraction Block
    # We treat the (101, 99, 12) input as an "image"
    
    # Initial "stem"
    x = layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 2. Fully-Connected Prediction Head
    # Flatten the 2D features into a 1D vector
    x = layers.Flatten()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Final output layer
    outputs = layers.Dense(Config.NUM_TARGETS, activation='sigmoid', name="motor_output_vector")(x)
    # We use 'sigmoid' because our targets are all normalized to 0-1
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Start with a low LR
        loss='mse', # Mean Squared Error
        metrics=['mae'] # Mean Absolute Error
    )
    return model

# --- 3. TRAINING & EVALUATION ---

def setup_experiment_directory(experiment_name):
    """Creates a unique directory for storing all artifacts of a training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def train_model(model_creator, model_name, train_dataset, val_dataset, train_steps, val_steps):
    """The main function to train a single model and save all results."""
    print(f"\n{'='*60}\n🚀 STARTING TRAINING FOR {model_name}\n{'='*60}")
    
    exp_dir = setup_experiment_directory(f"series_{model_name}")
    model = model_creator()
    model.summary()
    
    # Callbacks
    best_model_path = os.path.join(exp_dir, "models", f"best_{model_name}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
    ]

    print(f"\nTraining {model_name}...")
    start_time = datetime.datetime.now()

    history = model.fit(
        train_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    
    # Load best model and get its validation MAE
    best_val_mae = min(history.history['val_mae'])
    
    # Save results
    results = {
        'model_version': model_name,
        'best_validation_mae': float(best_val_mae),
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_mae']) + 1),
        'training_time_seconds': training_time.total_seconds(),
        'model_architecture': json.loads(model.to_json())
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create and save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE'); plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f'{model_name}_training_history.png'))
    plt.close()

    print(f"\n🎯 {model_name} - FINAL RESULTS:")
    print(f"   Best Validation MAE: {results['best_validation_mae']:.4f}")
    print(f"   Training Time: {str(training_time).split('.')[0]}")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"💾 Results and artifacts saved to: {exp_dir}")
    return results

# --- 4. MAIN EXECUTION ---
def main():
    print("🚀 STARTING SPECTROGRAM MODEL TRAINING (PHASE 3b)")
    print("=============================================================")
    
    # Find the data we just created
    train_files = glob.glob(Config.TRAIN_TFRECORD_PATH)
    val_files = glob.glob(Config.VAL_TFRECORD_PATH)
    
    if not train_files or not val_files:
        print(f"❌ FATAL: Could not find TFRecord files.")
        print(f"   Checked for train: {Config.TRAIN_TFRECORD_PATH}")
        print(f"   Checked for val: {Config.VAL_TFRECORD_PATH}")
        return
    
    # Calculate steps per epoch
    train_steps = Config.TOTAL_TRAIN_SEQS // Config.BATCH_SIZE
    val_steps = Config.TOTAL_VAL_SEQS // Config.BATCH_SIZE
    
    if train_steps == 0: train_steps = 1
    if val_steps == 0: val_steps = 1

    print(f"   Found {Config.TOTAL_TRAIN_SEQS} train sequences and {Config.TOTAL_VAL_SEQS} val sequences.")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Steps/Epoch: {train_steps}")
    print(f"   Validation Steps/Epoch: {val_steps}")

    # Create the datasets
    train_dataset = create_dataset(Config.TRAIN_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(Config.VAL_TFRECORD_PATH, Config.BATCH_SIZE, shuffle=False)

    # Train the single v6 model
    results = train_model(
        create_v6_cnn_spectrogram, 
        "v6_cnn_spectrogram", 
        train_dataset, 
        val_dataset, 
        train_steps, 
        val_steps
    )

    print(f"\n{'='*70}\n📊 END-TO-END EXPERIMENT COMPLETE 📊\n{'='*70}")
    
    print(f"{'Model':<25} {'Best Val MAE':<15}")
    print('-'*70)
    print(f"{results['model_version']:<25} {results['best_validation_mae']:.4f}")
    
    print(f"\n   Recall: Our 'Sequential' (v4) model had a Val MAE of 0.1149")
    
    if results['best_validation_mae'] < 0.1149:
        print(f"\n   🏆🏆🏆 NEW CHAMPION! The End-to-End model is the best one so far! 🏆🏆🏆")
    else:
        print(f"\n   ... The Sequential (v4) model remains the champion.")


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
```

---

## File: v2.0\compute_normalization_params.py
**Path:** `C:\stage\stage\v2.0\compute_normalization_params.py`

```python
import h5py
import numpy as np

def compute_emg_stats(x_file, restimulus_file):
    """Compute EMG mean and std using only rest periods"""
    with h5py.File(x_file, 'r') as f_x, h5py.File(restimulus_file, 'r') as f_r:
        # Use the correct dataset name 'data' for EMG
        emg_data = f_x['data']
        restimulus = f_r['restimulus'][:]
        
        # Find rest periods (where restimulus != 0)
        rest_indices = np.where(restimulus == 0)[0]
        
        n_channels = emg_data.shape[1]
        rest_mean = np.zeros(n_channels)
        rest_std = np.zeros(n_channels)
        
        # Read rest data in chunks
        chunk_size = 10000
        for i in range(0, len(rest_indices), chunk_size):
            indices_chunk = rest_indices[i:i+chunk_size]
            emg_chunk = emg_data[indices_chunk, :]
            rest_mean += np.sum(emg_chunk, axis=0)
            rest_std += np.sum(emg_chunk**2, axis=0)
        
        rest_mean /= len(rest_indices)
        rest_std = np.sqrt(rest_std / len(rest_indices) - rest_mean**2)
        
        return rest_mean, rest_std


# Use raw strings for Windows paths
x_file_path = r'D:\stage\data\cleaned_x.h5'
restimulus_file_path = r'D:\stage\data\cleaned_restimulus.h5'

try:
    # Compute and save EMG normalization parameters
    print("Computing EMG normalization parameters...")
    emg_mean, emg_std = compute_emg_stats(x_file_path, restimulus_file_path)
    np.save('emg_rest_mean.npy', emg_mean)
    np.save('emg_rest_std.npy', emg_std)
    print("EMG normalization parameters saved")

    print("All normalization parameters computed and saved successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

---

## File: v2.0\hdf5_data_loader.py
**Path:** `C:\stage\stage\v2.0\hdf5_data_loader.py`

```python
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence
import joblib

class HDF5DataGenerator(Sequence):
    def __init__(self, x_file, y_file, restimulus_file, batch_size, sequence_length, target_shift, shuffle=True):
        self.x_file = x_file
        self.y_file = y_file
        self.restimulus_file = restimulus_file
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_shift = target_shift
        self.shuffle = shuffle
        
        # Load preprocessing parameters
        self.emg_mean = np.load('emg_rest_mean.npy')
        self.emg_std = np.load('emg_rest_std.npy')
        self.glove_min = np.load('glove_min.npy')
        self.glove_range = np.load('glove_range.npy')
        self.nmf = joblib.load('nmf_model.pkl')
        
        # Open HDF5 files
        self.x_f = h5py.File(x_file, 'r')
        self.y_f = h5py.File(y_file, 'r')
        self.restimulus_f = h5py.File(restimulus_file, 'r')
        
        # Get dataset dimensions
        self.n_samples = self.x_f['data'].shape[0]
        
        # Precompute valid indices (movement periods)
        self.valid_indices = self._get_valid_indices()
        
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
        
    def _get_valid_indices(self):
        """Get indices where sequences are entirely in movement periods"""
        restimulus = self.restimulus_f['restimulus'][:]
        movement_indices = np.where(restimulus != 0)[0]
        
        # Only keep indices where the entire sequence is within movement periods
        valid_indices = []
        for i in movement_indices:
            if i + self.sequence_length + self.target_shift < len(restimulus):
                # Check if the entire sequence is in movement period
                if np.all(restimulus[i:i+self.sequence_length] != 0):
                    valid_indices.append(i)
        
        return np.array(valid_indices)
    
    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.valid_indices))
        batch_indices = self.valid_indices[start_idx:end_idx]
        
        batch_X = []
        batch_y = []
        
        for i in batch_indices:
            # Read EMG sequence
            emg_seq = self.x_f['data'][i:i+self.sequence_length, :]
            # Normalize EMG
            emg_normalized = (emg_seq - self.emg_mean) / self.emg_std
            # Take absolute value for NMF
            emg_abs = np.abs(emg_normalized)
            synergy_features = self.nmf.transform(emg_abs)
            batch_X.append(synergy_features)
            
            # Read target: glove data at future time point
            target_index = i + self.sequence_length + self.target_shift
            glove_data = self.y_f['labels'][target_index, :]  # Changed from 'data' to 'labels'
            # Normalize glove data
            glove_normalized = (glove_data - self.glove_min) / self.glove_range
            batch_y.append(glove_normalized)
        
        return np.array(batch_X), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
    
    def __del__(self):
        if hasattr(self, 'x_f') and self.x_f:
            self.x_f.close()
        if hasattr(self, 'y_f') and self.y_f:
            self.y_f.close()
        if hasattr(self, 'restimulus_f') and self.restimulus_f:
            self.restimulus_f.close()
```

---

## File: v2.0\recompute_glove_stats.py
**Path:** `C:\stage\stage\v2.0\recompute_glove_stats.py`

```python
import h5py
import numpy as np

def compute_glove_stats_with_nan_handling(y_file):
    """Compute min and range for glove data, handling NaN values"""
    with h5py.File(y_file, 'r') as f:
        glove_data = f['labels']
        n_samples = glove_data.shape[0]
        n_sensors = glove_data.shape[1]
        
        min_vals = np.full(n_sensors, np.inf)
        max_vals = np.full(n_sensors, -np.inf)
        
        chunk_size = 10000
        for i in range(0, n_samples, chunk_size):
            chunk = glove_data[i:i+chunk_size, :]
            
            # Compute min and max while ignoring NaN values
            chunk_min = np.nanmin(chunk, axis=0)
            chunk_max = np.nanmax(chunk, axis=0)
            
            # Update overall min and max
            min_vals = np.minimum(min_vals, chunk_min)
            max_vals = np.maximum(max_vals, chunk_max)
            
        return min_vals, max_vals

# Use raw string for Windows path
y_file_path = r'D:\stage\data\cleaned_y.h5'

# Compute and save glove normalization parameters with NaN handling
print("Computing glove normalization parameters with NaN handling...")
glove_min, glove_max = compute_glove_stats_with_nan_handling(y_file_path)
glove_range = glove_max - glove_min
glove_range[glove_range == 0] = 1  # Avoid division by zero

# Check for any remaining NaN values
if np.any(np.isnan(glove_min)) or np.any(np.isnan(glove_range)):
    print("Warning: Still found NaN values in normalization parameters")
    # Replace any remaining NaN values with reasonable defaults
    glove_min = np.nan_to_num(glove_min, nan=0.0)
    glove_range = np.nan_to_num(glove_range, nan=1.0)

np.save('glove_min.npy', glove_min)
np.save('glove_range.npy', glove_range)
print("Glove normalization parameters saved with NaN handling")
```

---

## File: v2.0\train_model.py
**Path:** `C:\stage\stage\v2.0\train_model.py`

```python
from hdf5_data_loader import HDF5DataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Parameters
batch_size = 64
sequence_length = 500  # 250ms window at 2000Hz
target_shift = 300     # 150ms prediction horizon

# Create data generator
train_generator = HDF5DataGenerator(
    x_file=r'D:\stage\x.h5',
    y_file=r'D:\stage\y.h5',
    restimulus_file=r'D:\stage\restimulus.h5',
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    shuffle=True
)

# Build model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 6), return_sequences=True,
         kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(18, activation='sigmoid')  # 18 glove sensors for DB7
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Train model
print("Training the LSTM model with HDF5 data generator...")
history = model.fit(
    train_generator,
    epochs=100,
    verbose=1
)

# Save model
model.save('prosthetic_hand_model.h5')
print("Model saved as prosthetic_hand_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')

plt.savefig('training_history.png')
plt.show()
```

---

## File: v2.0\train_nmf_model.py
**Path:** `C:\stage\stage\v2.0\train_nmf_model.py`

```python
import h5py
import numpy as np
import joblib
from sklearn.decomposition import MiniBatchNMF

# Load normalization parameters
emg_mean = np.load('emg_rest_mean.npy')
emg_std = np.load('emg_rest_std.npy')

# File paths
x_file_path = r'F:\DB\x.h5'
restimulus_file_path = r'F:\DB\restimulus.h5'

# Load restimulus data to find movement periods (where restimulus == 0)
with h5py.File(restimulus_file_path, 'r') as f:
    restimulus = f['restimulus'][:]

movement_indices = np.where(restimulus != 0)[0]

# We'll use a subset of the movement data to train the NMF model
n_samples_for_nmf = 100000  # Adjust as needed
if len(movement_indices) > n_samples_for_nmf:
    indices = np.random.choice(movement_indices, n_samples_for_nmf, replace=False)
    # Sort indices to avoid HDF5 reading error
    indices = np.sort(indices)
else:
    indices = movement_indices

# Load the corresponding EMG data in chunks to avoid memory issues
chunk_size = 10000
emg_chunks = []

with h5py.File(x_file_path, 'r') as f:
    dataset = f['data']
    for i in range(0, len(indices), chunk_size):
        chunk_indices = indices[i:i+chunk_size]
        emg_chunk = dataset[chunk_indices, :]
        emg_chunks.append(emg_chunk)

# Combine all chunks
emg_data = np.vstack(emg_chunks)

# Normalize the EMG data using the rest period statistics
emg_normalized = (emg_data - emg_mean) / emg_std

# Take absolute value for NMF (NMF requires non-negative data)
emg_abs = np.abs(emg_normalized)

# Train NMF model
nmf_components = 6  # Number of muscle synergies
nmf = MiniBatchNMF(n_components=nmf_components, random_state=42, batch_size=1000, max_iter=100)
nmf.fit(emg_abs)

# Save the NMF model
joblib.dump(nmf, 'nmf_model.pkl')
print("NMF model trained and saved successfully!")
```

---

## File: v3.0\check_restimulus.py
**Path:** `C:\stage\stage\v3.0\check_restimulus.py`

```python
import h5py
import numpy as np

# Check the restimulus data
with h5py.File(r'D:\stage\data\cleaned_restimulus.h5', 'r') as f:
    restimulus = f['restimulus'][:]
    
    # Check the distribution of values
    unique, counts = np.unique(restimulus, return_counts=True)
    print("Restimulus value distribution:")
    for val, count in zip(unique, counts):
        print(f"Value {val}: {count} occurrences")
    
    # Check if there are any zeros
    zero_indices = np.where(restimulus == 0)[0]
    print(f"\nFound {len(zero_indices)} rest periods (value 0) in the entire dataset")
    
    if len(zero_indices) > 0:
        # Check if these zeros are in the training indices
        train_indices = np.load('train_indices.npy')
        train_rest_indices = np.intersect1d(zero_indices, train_indices)
        print(f"Found {len(train_rest_indices)} rest periods in training data")
        
        if len(train_rest_indices) == 0:
            print("WARNING: No rest periods in training data!")
            print("This suggests your training indices only include movement periods")
```

---

## File: v3.0\compute_normalization_params.py
**Path:** `C:\stage\stage\v3.0\compute_normalization_params.py`

```python
import h5py
import numpy as np

def compute_emg_stats(x_file, restimulus_file):
    """Compute EMG mean and std using rest periods from entire dataset"""
    with h5py.File(x_file, 'r') as f_x, h5py.File(restimulus_file, 'r') as f_r:
        emg_data = f_x['data']
        restimulus = f_r['restimulus'][:]
        
        # Find rest periods in entire dataset
        rest_indices = np.where(restimulus == 0)[0]
        
        if len(rest_indices) == 0:
            raise ValueError("No rest periods found in the entire dataset!")
        
        # Sort indices to satisfy HDF5 requirements
        rest_indices.sort()
        
        n_channels = emg_data.shape[1]
        rest_mean = np.zeros(n_channels)
        rest_std = np.zeros(n_channels)
        
        # Read rest data in chunks
        chunk_size = 10000
        for i in range(0, len(rest_indices), chunk_size):
            indices_chunk = rest_indices[i:i+chunk_size]
            emg_chunk = emg_data[indices_chunk, :]
            rest_mean += np.sum(emg_chunk, axis=0)
            rest_std += np.sum(emg_chunk**2, axis=0)
        
        rest_mean /= len(rest_indices)
        rest_std = np.sqrt(rest_std / len(rest_indices) - rest_mean**2)
        
        return rest_mean, rest_std

# Use raw strings for Windows paths
x_file_path = r'D:\stage\data\cleaned_x.h5'
restimulus_file_path = r'D:\stage\data\cleaned_restimulus.h5'

try:
    # Compute and save EMG normalization parameters using entire dataset
    print("Computing EMG normalization parameters using entire dataset...")
    emg_mean, emg_std = compute_emg_stats(x_file_path, restimulus_file_path)
    np.save('emg_rest_mean.npy', emg_mean)
    np.save('emg_rest_std.npy', emg_std)
    print("EMG normalization parameters saved")

    print("All normalization parameters computed successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

---

## File: v3.0\create_balanced_splits.py
**Path:** `C:\stage\stage\v3.0\create_balanced_splits.py`

```python
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import os

# Load the subject master list
master_list = pd.read_csv('subject_master_list.csv')

# Get unique subjects
unique_subjects = master_list[['unique_id', 'subject_type']].drop_duplicates()

# Separate amputee and intact subjects
amputee_subjects = unique_subjects[unique_subjects['subject_type'] == 'amputee']
intact_subjects = unique_subjects[unique_subjects['subject_type'] == 'intact']

print(f"Total unique subjects: {len(unique_subjects)}")
print(f"Amputee subjects: {len(amputee_subjects)}")
print(f"Intact subjects: {len(intact_subjects)}")

# Set random seed for reproducibility
np.random.seed(42)

# Split amputee subjects: 7 train, 2 validation, 2 test
amputee_ids = amputee_subjects['unique_id'].values
np.random.shuffle(amputee_ids)
amputee_train = amputee_ids[:7]
amputee_val = amputee_ids[7:9]
amputee_test = amputee_ids[9:11]

# Split intact subjects: 42 train, 9 validation, 9 test
intact_ids = intact_subjects['unique_id'].values
np.random.shuffle(intact_ids)
intact_train = intact_ids[:42]
intact_val = intact_ids[42:51]
intact_test = intact_ids[51:60]

# Combine subject lists for each set
train_subjects = list(amputee_train) + list(intact_train)
val_subjects = list(amputee_val) + list(intact_val)
test_subjects = list(amputee_test) + list(intact_test)

print(f"Training subjects: {len(train_subjects)}")
print(f"Validation subjects: {len(val_subjects)}")
print(f"Test subjects: {len(test_subjects)}")

# Open the restimulus.h5 file
restimulus_file_path = r'D:\stage\data\cleaned_restimulus.h5'
f = h5py.File(restimulus_file_path, 'r')
restimulus_dataset = f['restimulus']

# Create a dictionary to map each subject to its segments
subject_segments = {}
for unique_id in unique_subjects['unique_id']:
    segments = master_list[master_list['unique_id'] == unique_id]
    subject_segments[unique_id] = segments

# Initialize lists for indices
train_movement_indices = []
train_rest_indices = []
val_movement_indices = []
val_rest_indices = []
test_movement_indices = []
test_rest_indices = []

# Function to process subjects and collect indices
def process_subjects(subject_list, movement_indices_list, rest_indices_list):
    for subject in tqdm(subject_list):
        segments = subject_segments[subject]
        for _, segment in segments.iterrows():
            start = segment['global_start_index']
            end = segment['global_end_index']
            
            # Read restimulus data for this segment
            restimulus_chunk = restimulus_dataset[start:end+1]
            
            # Find movement indices (restimulus != 0)
            movement_indices = np.where(restimulus_chunk != 0)[0]
            movement_indices_global = movement_indices + start
            movement_indices_list.extend(movement_indices_global)
            
            # Find rest indices (restimulus == 0) but limit to balance the dataset
            rest_indices = np.where(restimulus_chunk == 0)[0]
            
            # Sample a balanced number of rest indices (same as movement indices)
            if len(rest_indices) > 0:
                n_rest_to_sample = min(len(movement_indices), len(rest_indices))
                sampled_rest_indices = np.random.choice(rest_indices, n_rest_to_sample, replace=False)
                rest_indices_global = sampled_rest_indices + start
                rest_indices_list.extend(rest_indices_global)

# Process training subjects
print("Processing training subjects...")
process_subjects(train_subjects, train_movement_indices, train_rest_indices)

# Process validation subjects
print("Processing validation subjects...")
process_subjects(val_subjects, val_movement_indices, val_rest_indices)

# Process test subjects
print("Processing test subjects...")
process_subjects(test_subjects, test_movement_indices, test_rest_indices)

# Close the HDF5 file
f.close()

# Combine movement and rest indices for each set
train_indices = np.concatenate([train_movement_indices, train_rest_indices])
val_indices = np.concatenate([val_movement_indices, val_rest_indices])
test_indices = np.concatenate([test_movement_indices, test_rest_indices])

# Shuffle each set
np.random.shuffle(train_indices)
np.random.shuffle(val_indices)
np.random.shuffle(test_indices)

# Save indices to files
np.save('train_indices.npy', train_indices)
np.save('val_indices.npy', val_indices)
np.save('test_indices.npy', test_indices)

# Also save the separate indices for debugging
np.save('train_movement_indices.npy', train_movement_indices)
np.save('train_rest_indices.npy', train_rest_indices)
np.save('val_movement_indices.npy', val_movement_indices)
np.save('val_rest_indices.npy', val_rest_indices)
np.save('test_movement_indices.npy', test_movement_indices)
np.save('test_rest_indices.npy', test_rest_indices)

print("Balanced indices saved successfully!")
print(f"Training set: {len(train_movement_indices)} movement + {len(train_rest_indices)} rest = {len(train_indices)} total")
print(f"Validation set: {len(val_movement_indices)} movement + {len(val_rest_indices)} rest = {len(val_indices)} total")
print(f"Test set: {len(test_movement_indices)} movement + {len(test_rest_indices)} rest = {len(test_indices)} total")
```

---

## File: v3.0\create_data_splits.py
**Path:** `C:\stage\stage\v3.0\create_data_splits.py`

```python
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm  # For progress bars; install with `pip install tqdm` if needed

# Load the subject master list
master_list = pd.read_csv('subject_master_list.csv')

# Get unique subjects (each subject has a unique 'unique_id')
unique_subjects = master_list[['unique_id', 'subject_type']].drop_duplicates()

# Separate amputee and intact subjects
amputee_subjects = unique_subjects[unique_subjects['subject_type'] == 'amputee']
intact_subjects = unique_subjects[unique_subjects['subject_type'] == 'intact']

print(f"Total unique subjects: {len(unique_subjects)}")
print(f"Amputee subjects: {len(amputee_subjects)}")
print(f"Intact subjects: {len(intact_subjects)}")

# Set random seed for reproducibility
np.random.seed(42)

# Split amputee subjects: 7 train, 2 validation, 2 test
amputee_ids = amputee_subjects['unique_id'].values
np.random.shuffle(amputee_ids)
amputee_train = amputee_ids[:7]
amputee_val = amputee_ids[7:9]
amputee_test = amputee_ids[9:11]

# Split intact subjects: 42 train, 9 validation, 9 test
intact_ids = intact_subjects['unique_id'].values
np.random.shuffle(intact_ids)
intact_train = intact_ids[:42]
intact_val = intact_ids[42:51]
intact_test = intact_ids[51:60]

# Combine subject lists for each set
train_subjects = list(amputee_train) + list(intact_train)
val_subjects = list(amputee_val) + list(intact_val)
test_subjects = list(amputee_test) + list(intact_test)

print(f"Training subjects: {len(train_subjects)}")
print(f"Validation subjects: {len(val_subjects)}")
print(f"Test subjects: {len(test_subjects)}")

# Open the restimulus.h5 file
restimulus_file_path = r'D:\stage\data\cleaned_restimulus.h5'  # Update path if necessary
f = h5py.File(restimulus_file_path, 'r')
restimulus_dataset = f['restimulus']  # Ensure the dataset name is correct

# Create a dictionary to map each subject to its segments
subject_segments = {}
for unique_id in unique_subjects['unique_id']:
    segments = master_list[master_list['unique_id'] == unique_id]
    subject_segments[unique_id] = segments

# Initialize lists for indices
train_indices = []
val_indices = []
test_indices = []

# Function to process subjects and collect indices
def process_subjects(subject_list, indices_list):
    for subject in tqdm(subject_list):
        segments = subject_segments[subject]
        for _, segment in segments.iterrows():
            start = segment['global_start_index']
            end = segment['global_end_index']
            # Read restimulus data for this segment
            restimulus_chunk = restimulus_dataset[start:end+1]
            # Find indices where restimulus != 0 (movement periods)
            movement_indices = np.where(restimulus_chunk != 0)[0]
            # Convert to global indices
            movement_indices_global = movement_indices + start
            indices_list.extend(movement_indices_global)

# Process training subjects
print("Processing training subjects...")
process_subjects(train_subjects, train_indices)

# Process validation subjects
print("Processing validation subjects...")
process_subjects(val_subjects, val_indices)

# Process test subjects
print("Processing test subjects...")
process_subjects(test_subjects, test_indices)

# Close the HDF5 file
f.close()

# Convert lists to numpy arrays
train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
test_indices = np.array(test_indices)

# Save indices to files
np.save('train_indices.npy', train_indices)
np.save('val_indices.npy', val_indices)
np.save('test_indices.npy', test_indices)

# Save subject lists for verification
np.save('train_subjects.npy', train_subjects)
np.save('val_subjects.npy', val_subjects)
np.save('test_subjects.npy', test_subjects)

print("Indices saved successfully.")
```

---

## File: v3.0\create_subject_master.py
**Path:** `C:\stage\stage\v3.0\create_subject_master.py`

```python
import pandas as pd
import numpy as np
import h5py

# Load your cleaned map file
map_df = pd.read_csv('cleaned_map.csv')

# Let's create a list to hold all our subject information
subjects_list = []

# Iterate through each row in the map file
for index, row in map_df.iterrows():
    filename = row['filename']
    start = row['start_line'] - 1  # Convert to 0-based indexing
    end = row['end_line'] - 1      # Convert to 0-based indexing
    
    # Extract database and subject ID from the filename
    # Example: 'E:\DB\DB2\E2\S10_E2_A1.csv'
    parts = filename.split('\\')
    db_source = parts[2]  # 'DB2', 'DB3', or 'DB7'
    subject_file = parts[-1]  # 'S10_E2_A1.csv'
    subject_id = subject_file.split('_')[0]  # 'S10'
    
    # Determine subject type based on database source
    if db_source == 'DB3':
        subject_type = 'amputee'
        unique_id = f"DB3_{subject_id}"
    else:  # DB2 or DB7
        subject_type = 'intact'
        unique_id = f"{db_source}_{subject_id}"
    
    # Add to our master list
    subjects_list.append({
        'unique_id': unique_id,
        'db_source': db_source,
        'original_subject_id': subject_id,
        'subject_type': subject_type,
        'global_start_index': start,
        'global_end_index': end,
        'filename': filename
    })

# Convert to a DataFrame for easier manipulation
subjects_df = pd.DataFrame(subjects_list)

# Let's see what we have
print(f"Total subjects found: {len(subjects_df)}")
print(f"Amputee subjects: {len(subjects_df[subjects_df['subject_type'] == 'amputee'])}")
print(f"Intact subjects: {len(subjects_df[subjects_df['subject_type'] == 'intact'])}")

# Display first few entries
print("\nFirst 5 subjects:")
print(subjects_df.head())

# Save this master list for reference
subjects_df.to_csv('subject_master_list.csv', index=False)
print("\nSubject master list saved to 'subject_master_list.csv'")
```

---

## File: v3.0\debug.py
**Path:** `C:\stage\stage\v3.0\debug.py`

```python
import numpy as np
import h5py

# Load training indices
train_indices = np.load('train_indices.npy')

# Load restimulus data
with h5py.File(r'D:\stage\data\cleaned_restimulus.h5', 'r') as f:
    restimulus = f['restimulus'][:]

# Check how many rest periods are in the training indices
rest_count = 0
for idx in train_indices[:10000]:  # Check first 10k for speed
    if restimulus[idx] == 0:
        rest_count += 1

print(f"Found {rest_count} rest periods in first 10,000 training indices")
print(f"Total training indices: {len(train_indices)}")

if rest_count == 0:
    print("WARNING: No rest periods found in training data!")
    print("This suggests a problem with your data splitting or restimulus data")
```

---

## File: v3.0\evaluate_model.py
**Path:** `C:\stage\stage\v3.0\evaluate_model.py`

```python
from hdf5_data_loader import HDF5DataGenerator
from tensorflow.keras.models import load_model
import numpy as np

# Load test indices
test_indices = np.load('test_indices.npy')

# Create test generator
test_generator = HDF5DataGenerator(
    x_file=r'D:\stage\data\cleaned_x.h5',
    y_file=r'D:\stage\data\cleaned_y.h5',
    restimulus_file=r'D:\stage\data\cleaned_restimulus.h5',
    batch_size=64,
    sequence_length=500,
    target_shift=300,
    indices_list=test_indices,
    shuffle=False
)

# Load trained model
model = load_model('prosthetic_hand_model.h5')

# Evaluate on test set
print("Evaluating model on test set...")
test_loss, test_mae = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Optional: Get predictions for further analysis
print("Generating predictions...")
predictions = model.predict(test_generator, verbose=1)
np.save('test_predictions.npy', predictions)
```

---

## File: v3.0\gpu_monitor.py
**Path:** `C:\stage\stage\v3.0\gpu_monitor.py`

```python
import subprocess
import time

def monitor_gpu():
    while True:
        try:
            # Run nvidia-smi command
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv'],
                                  capture_output=True, text=True)
            print(result.stdout)
            time.sleep(5)  # Check every 5 seconds
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error monitoring GPU: {e}")
            break

if __name__ == "__main__":
    monitor_gpu()
```

---

## File: v3.0\hdf5_data_loader.py
**Path:** `C:\stage\stage\v3.0\hdf5_data_loader.py`

```python
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence
import joblib

class HDF5DataGenerator(Sequence):
    def __init__(self, x_file, y_file, restimulus_file, batch_size, sequence_length, target_shift, indices_list=None, shuffle=True):
        self.x_file = x_file
        self.y_file = y_file
        self.restimulus_file = restimulus_file
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_shift = target_shift
        self.shuffle = shuffle
        
        # Load preprocessing parameters
        self.emg_mean = np.load('emg_rest_mean.npy')
        self.emg_std = np.load('emg_rest_std.npy')  # Fixed: changed from emg_epoch to emg_std
        self.glove_min = np.load('glove_min.npy')
        self.glove_range = np.load('glove_range.npy')
        self.nmf = joblib.load('nmf_model.pkl')
        
        # Open HDF5 files
        self.x_f = h5py.File(x_file, 'r')
        self.y_f = h5py.File(y_file, 'r')
        self.restimulus_f = h5py.File(restimulus_file, 'r')
        
        # Use provided indices or compute valid indices
        if indices_list is not None:
            self.valid_indices = indices_list
        else:
            self.valid_indices = self._get_valid_indices()
        
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
        
    def _get_valid_indices(self):
        """Get indices where sequences are entirely in movement periods"""
        restimulus = self.restimulus_f['restimulus'][:]
        movement_indices = np.where(restimulus != 0)[0]  # Corrected: != 0 for movement
        
        # Only keep indices where the entire sequence is within movement periods
        valid_indices = []
        for i in movement_indices:
            if i + self.sequence_length + self.target_shift < len(restimulus):
                # Check if the entire sequence is in movement period
                if np.all(restimulus[i:i+self.sequence_length] != 0):  # Corrected: != 0 for movement
                    valid_indices.append(i)
        
        return np.array(valid_indices)
    
    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.valid_indices))
        batch_indices = self.valid_indices[start_idx:end_idx]
        
        batch_X = []
        batch_y = []
        
        for i in batch_indices:
            # Read EMG sequence
            emg_seq = self.x_f['data'][i:i+self.sequence_length, :]
            # Normalize EMG
            emg_normalized = (emg_seq - self.emg_mean) / self.emg_std  # Fixed: using emg_std
            # Take absolute value for NMF
            emg_abs = np.abs(emg_normalized)
            synergy_features = self.nmf.transform(emg_abs)
            batch_X.append(synergy_features)
            
            # Read target: glove data at future time point
            target_index = i + self.sequence_length + self.target_shift
            glove_data = self.y_f['labels'][target_index, :]
            # Normalize glove data
            glove_normalized = (glove_data - self.glove_min) / self.glove_range
            batch_y.append(glove_normalized)
        
        return np.array(batch_X), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
    
    def __del__(self):
        if hasattr(self, 'x_f') and self.x_f:
            self.x_f.close()
        if hasattr(self, 'y_f') and self.y_f:
            self.y_f.close()
        if hasattr(self, 'restimulus_f') and self.restimulus_f:
            self.restimulus_f.close()
```

---

## File: v3.0\hdf5_data_loader_fast.py
**Path:** `C:\stage\stage\v3.0\hdf5_data_loader_fast.py`

```python
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

class HDF5DataGeneratorFast(Sequence):
    def __init__(self, x_nmf_file, y_file, batch_size, sequence_length, target_shift, indices_list=None, shuffle=True):
        self.x_nmf_file = x_nmf_file
        self.y_file = y_file
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_shift = target_shift
        self.shuffle = shuffle
        
        # Load preprocessing parameters for glove data only
        self.glove_min = np.load('glove_min.npy')
        self.glove_range = np.load('glove_range.npy')
        
        # Open HDF5 files
        self.x_nmf_f = h5py.File(x_nmf_file, 'r')
        self.y_f = h5py.File(y_file, 'r')
        
        # Use provided indices
        self.valid_indices = indices_list
        
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
    
    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.valid_indices))
        batch_indices = self.valid_indices[start_idx:end_idx]
        
        batch_X = []
        batch_y = []
        
        for i in batch_indices:
            # Read precomputed NMF sequence
            nmf_seq = self.x_nmf_f['nmf_features'][i:i+self.sequence_length, :]
            batch_X.append(nmf_seq)
            
            # Read target: glove data at future time point
            target_index = i + self.sequence_length + self.target_shift
            glove_data = self.y_f['labels'][target_index, :]
            # Normalize glove data
            glove_normalized = (glove_data - self.glove_min) / self.glove_range
            batch_y.append(glove_normalized)
        
        return np.array(batch_X), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
    
    def __del__(self):
        if hasattr(self, 'x_nmf_f') and self.x_nmf_f:
            self.x_nmf_f.close()
        if hasattr(self, 'y_f') and self.y_f:
            self.y_f.close()
```

---

## File: v3.0\hdf5_data_loader_optimized.py
**Path:** `C:\stage\stage\v3.0\hdf5_data_loader_optimized.py`

```python
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

class BalancedHDF5DataGenerator(Sequence):
    def __init__(self, x_file, y_file, batch_size, sequence_length, target_shift, indices_list=None, shuffle=True):
        self.x_file = x_file
        self.y_file = y_file
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_shift = target_shift
        self.shuffle = shuffle
        
        # Load preprocessing parameters
        self.emg_mean = np.load('emg_rest_mean.npy')
        self.emg_std = np.load('emg_rest_std.npy')
        self.glove_min = np.load('glove_min.npy')
        self.glove_range = np.load('glove_range.npy')
        
        # Open HDF5 files
        self.x_f = h5py.File(x_file, 'r')
        self.y_f = h5py.File(y_file, 'r')
        
        # Use provided indices
        self.valid_indices = indices_list
        
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
    
    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.valid_indices))
        batch_indices = self.valid_indices[start_idx:end_idx]
        
        batch_X = []
        batch_y = []
        
        for i in batch_indices:
            # Read EMG sequence
            emg_seq = self.x_f['data'][i:i+self.sequence_length, :]
            
            # Normalize EMG
            emg_normalized = (emg_seq - self.emg_mean) / self.emg_std
            
            # For rest periods, we might want to use raw features instead of NMF
            # Or we can still use NMF but be aware that it was trained on movement data
            # Let's use the same preprocessing for both for consistency
            emg_abs = np.abs(emg_normalized)
            
            # Read target: glove data at future time point
            target_index = i + self.sequence_length + self.target_shift
            glove_data = self.y_f['labels'][target_index, :]
            
            # For rest periods, we know the target should be zeros (or minimal values)
            # Normalize glove data
            glove_normalized = (glove_data - self.glove_min) / self.glove_range
            
            batch_X.append(emg_abs)  # Using absolute values instead of NMF for now
            batch_y.append(glove_normalized)
        
        return np.array(batch_X), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
    
    def __del__(self):
        if hasattr(self, 'x_f') and self.x_f:
            self.x_f.close()
        if hasattr(self, 'y_f') and self.y_f:
            self.y_f.close()
```

---

## File: v3.0\precompute_nmf_features.py
**Path:** `C:\stage\stage\v3.0\precompute_nmf_features.py`

```python
import h5py
import numpy as np
import joblib
from tqdm import tqdm

# Load the NMF model and normalization parameters
nmf = joblib.load('nmf_model.pkl')
emg_mean = np.load('emg_rest_mean.npy')
emg_std = np.load('emg_rest_std.npy')

# File paths
x_file_path = r'D:\stage\data\cleaned_x.h5'
output_file_path = r'D:\stage\data\x_nmf.h5'

print("Precomputing NMF features for entire dataset...")

with h5py.File(x_file_path, 'r') as f_in, h5py.File(output_file_path, 'w') as f_out:
    emg_data = f_in['data']
    n_samples = emg_data.shape[0]
    n_components = nmf.n_components_
    
    # Create dataset for precomputed NMF features
    nmf_dataset = f_out.create_dataset('nmf_features', 
                                      shape=(n_samples, n_components),
                                      dtype=np.float32,
                                      chunks=(1000, n_components),
                                      compression='gzip')
    
    # Process in chunks
    chunk_size = 10000
    for i in tqdm(range(0, n_samples, chunk_size)):
        end_idx = min(i + chunk_size, n_samples)
        
        # Read EMG chunk
        emg_chunk = emg_data[i:end_idx, :]
        
        # Normalize EMG
        emg_normalized = (emg_chunk - emg_mean) / emg_std
        
        # Take absolute value for NMF
        emg_abs = np.abs(emg_normalized)
        
        # Apply NMF transformation
        nmf_features = nmf.transform(emg_abs)
        
        # Save to output file
        nmf_dataset[i:end_idx, :] = nmf_features

print("NMF features precomputation complete!")
```

---

## File: v3.0\recompute_glove_stats.py
**Path:** `C:\stage\stage\v3.0\recompute_glove_stats.py`

```python
import h5py
import numpy as np

def compute_glove_stats_train_only(y_file, train_indices_path):
    """Compute min and range for glove data using only training data, handling NaN values"""
    train_indices = np.load(train_indices_path)
    
    # Sort indices to satisfy HDF5 requirements
    train_indices.sort()
    
    with h5py.File(y_file, 'r') as f:
        glove_data = f['labels']
        n_sensors = glove_data.shape[1]
        
        min_vals = np.full(n_sensors, np.inf)
        max_vals = np.full(n_sensors, -np.inf)
        
        # Process training indices in chunks
        chunk_size = 10000
        for i in range(0, len(train_indices), chunk_size):
            indices_chunk = train_indices[i:i+chunk_size]
            chunk = glove_data[indices_chunk, :]
            
            # Compute min and max while ignoring NaN values
            chunk_min = np.nanmin(chunk, axis=0)
            chunk_max = np.nanmax(chunk, axis=0)
            
            # Update overall min and max
            min_vals = np.minimum(min_vals, chunk_min)
            max_vals = np.maximum(max_vals, chunk_max)
            
        return min_vals, max_vals

# Use raw string for Windows path
y_file_path = r'D:\stage\data\cleaned_y.h5'
train_indices_path = 'train_indices.npy'

# Compute and save glove normalization parameters with NaN handling using only training data
print("Computing glove normalization parameters using training data only...")
glove_min, glove_max = compute_glove_stats_train_only(y_file_path, train_indices_path)
glove_range = glove_max - glove_min
glove_range[glove_range == 0] = 1  # Avoid division by zero

# Check for any remaining NaN values
if np.any(np.isnan(glove_min)) or np.any(np.isnan(glove_range)):
    print("Warning: Still found NaN values in normalization parameters")
    # Replace any remaining NaN values with reasonable defaults
    glove_min = np.nan_to_num(glove_min, nan=0.0)
    glove_range = np.nan_to_num(glove_range, nan=1.0)

np.save('glove_min.npy', glove_min)
np.save('glove_range.npy', glove_range)
print("Glove normalization parameters (training data only) saved")
```

---

## File: v3.0\tf_data_loader.py
**Path:** `C:\stage\stage\v3.0\tf_data_loader.py`

```python
import tensorflow as tf
import numpy as np
import h5py

def create_tf_dataset(x_nmf_file, y_file, indices, batch_size, sequence_length, target_shift, shuffle=True):
    """
    Creates a TensorFlow Dataset that efficiently loads precomputed NMF features and glove data
    """
    # Load normalization parameters
    glove_min = np.load('glove_min.npy')
    glove_range = np.load('glove_range.npy')
    
    # Convert indices to a TensorFlow constant for better performance
    indices = tf.constant(indices, dtype=tf.int64)
    
    def data_generator():
        # Open HDF5 files (will be opened once per worker)
        with h5py.File(x_nmf_file, 'r') as x_f, h5py.File(y_file, 'r') as y_f:
            nmf_features = x_f['nmf_features']
            glove_data = y_f['labels']
            
            # Iterate through indices
            for i in indices:
                # Read precomputed NMF sequence
                nmf_seq = nmf_features[i:i+sequence_length, :]
                
                # Read target: glove data at future time point
                target_index = i + sequence_length + target_shift
                glove_target = glove_data[target_index, :]
                
                # Normalize glove data
                glove_normalized = (glove_target - glove_min) / glove_range
                
                yield nmf_seq, glove_normalized
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(sequence_length, 6), dtype=tf.float32),
            tf.TensorSpec(shape=(18,), dtype=tf.float32)
        )
    )
    
    # Optimize the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.cache()
    
    return dataset
```

---

## File: v3.0\train_model.py
**Path:** `C:\stage\stage\v3.0\train_model.py`

```python
from hdf5_data_loader import HDF5DataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Parameters
batch_size = 64
sequence_length = 500  # 250ms window at 2000Hz
target_shift = 300     # 150ms prediction horizon

# Load indices
train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')

# Create data generators
train_generator = HDF5DataGenerator(
    x_file=r'D:\stage\data\cleaned_x.h5',
    y_file=r'D:\stage\data\cleaned_y.h5',
    restimulus_file=r'D:\stage\data\cleaned_restimulus.h5',
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    indices_list=train_indices,
    shuffle=True
)

val_generator = HDF5DataGenerator(
    x_file=r'D:\stage\data\cleaned_x.h5',
    y_file=r'D:\stage\data\cleaned_y.h5',
    restimulus_file=r'D:\stage\data\cleaned_restimulus.h5',
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    indices_list=val_indices,
    shuffle=False
)

# Build model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 6), return_sequences=True,
         kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(18, activation='sigmoid')  # 18 glove sensors for DB7
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train model
print("Training the LSTM model with HDF5 data generator...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save model
model.save('prosthetic_hand_model.h5')
print("Model saved as prosthetic_hand_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('training_history.png')
plt.show()
```

---

## File: v3.0\train_model_fast.py
**Path:** `C:\stage\stage\v3.0\train_model_fast.py`

```python
from hdf5_data_loader_fast import HDF5DataGeneratorFast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Parameters
batch_size = 256  # Increased batch size for better GPU utilization
sequence_length = 500
target_shift = 300

# Load indices
train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')

# Create data generators
train_generator = HDF5DataGeneratorFast(
    x_nmf_file=r'D:\stage\data\x_nmf.h5',  # Use precomputed NMF features
    y_file=r'D:\stage\data\cleaned_y.h5',
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    indices_list=train_indices,
    shuffle=True
)

val_generator = HDF5DataGeneratorFast(
    x_nmf_file=r'D:\stage\data\x_nmf.h5',
    y_file=r'D:\stage\data\cleaned_y.h5',
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    indices_list=val_indices,
    shuffle=False
)

# Build model (same as before)
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 6), return_sequences=True,
         kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(18, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train model
print("Training the LSTM model with precomputed NMF features...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save model
model.save('prosthetic_hand_model.h5')
print("Model saved as prosthetic_hand_model.h5")

# Plot training history (same as before)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('training_history.png')
plt.show()
```

---

## File: v3.0\train_model_optimized.py
**Path:** `C:\stage\stage\v3.0\train_model_optimized.py`

```python
from tf_data_loader import create_tf_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import numpy as np
import tensorflow as tf

# Enable mixed precision for faster training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Parameters
batch_size = 512  # Increased batch size
sequence_length = 500
target_shift = 300

# Load indices
train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')

# Create TensorFlow datasets
print("Creating TensorFlow datasets...")
train_dataset = create_tf_dataset(
    x_nmf_file=r'D:\stage\data\x_nmf.h5',
    y_file=r'D:\stage\data\cleaned_y.h5',
    indices=train_indices,
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    shuffle=True
)

val_dataset = create_tf_dataset(
    x_nmf_file=r'D:\stage\data\x_nmf.h5',
    y_file=r'D:\stage\data\cleaned_y.h5',
    indices=val_indices,
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    shuffle=False
)

# Build model with mixed precision compatibility
with tf.device('/GPU:0'):
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, 6), return_sequences=True,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        Dropout(0.4),
        BatchNormalization(),
        LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        Dropout(0.4),
        BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(18, activation='sigmoid')
    ])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train model
print("Training the optimized LSTM model...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save model
model.save('prosthetic_hand_model.h5')
print("Model saved as prosthetic_hand_model.h5")
```

---

## File: v3.0\train_model_optimized_final.py
**Path:** `C:\stage\stage\v3.0\train_model_optimized_final.py`

```python
from hdf5_data_loader_fast import HDF5DataGeneratorFast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# Reduced parameters for GTX 1050
batch_size = 32  # Reduced from 256 to 32
sequence_length = 500
target_shift = 300

# Load indices
train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')

# Create data generators
train_generator = HDF5DataGeneratorFast(
    x_nmf_file=r'D:\stage\data\x_nmf.h5',
    y_file=r'D:\stage\data\cleaned_y.h5',
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    indices_list=train_indices,
    shuffle=True
)

val_generator = HDF5DataGeneratorFast(
    x_nmf_file=r'D:\stage\data\x_nmf.h5',
    y_file=r'D:\stage\data\cleaned_y.h5',
    batch_size=batch_size,
    sequence_length=sequence_length,
    target_shift=target_shift,
    indices_list=val_indices,
    shuffle=False
)

# Build a smaller model for GTX 1050
model = Sequential([
    LSTM(32, input_shape=(sequence_length, 6), return_sequences=True,  # Reduced from 64 to 32
         kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.3),  # Reduced from 0.4
    BatchNormalization(),
    LSTM(16, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),  # Reduced from 32 to 16
    Dropout(0.3),  # Reduced from 0.4
    BatchNormalization(),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),  # Reduced from 32 to 16
    Dropout(0.2),  # Reduced from 0.3
    Dense(18, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train model
print("Training the optimized LSTM model for GTX 1050...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1,
    max_queue_size=10,  # Reduce queue size to save memory
    workers=4,  # Reduce number of workers
    use_multiprocessing=False  # Disable multiprocessing to save memory
)

# Save final model
model.save('prosthetic_hand_model_final.h5')
print("Model saved as prosthetic_hand_model_final.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('training_history_final.png')
plt.show()
```

---

## File: v3.0\train_nmf_model.py
**Path:** `C:\stage\stage\v3.0\train_nmf_model.py`

```python
import h5py
import numpy as np
import joblib
from sklearn.decomposition import MiniBatchNMF

# Load normalization parameters
emg_mean = np.load('emg_rest_mean.npy')
emg_std = np.load('emg_rest_std.npy')

# File paths
x_file_path = r'D:\stage\data\cleaned_x.h5'
restimulus_file_path = r'D:\stage\data\cleaned_restimulus.h5'

# Load training indices to ensure NMF is trained only on training data
train_indices = np.load('train_indices.npy')

# We'll use a subset of the training movement data to train the NMF model
n_samples_for_nmf = min(100000, len(train_indices))
indices = np.random.choice(train_indices, n_samples_for_nmf, replace=False)

# Sort indices to satisfy HDF5 requirements
indices.sort()

# Load the corresponding EMG data in chunks to avoid memory issues
chunk_size = 10000
emg_chunks = []

with h5py.File(x_file_path, 'r') as f:
    dataset = f['data']
    for i in range(0, len(indices), chunk_size):
        chunk_indices = indices[i:i+chunk_size]
        emg_chunk = dataset[chunk_indices, :]
        emg_chunks.append(emg_chunk)

# Combine all chunks
emg_data = np.vstack(emg_chunks)

# Normalize the EMG data using the rest period statistics
emg_normalized = (emg_data - emg_mean) / emg_std

# Take absolute value for NMF (NMF requires non-negative data)
emg_abs = np.abs(emg_normalized)

# Train NMF model
nmf_components = 6  # Number of muscle synergies
nmf = MiniBatchNMF(n_components=nmf_components, random_state=42, batch_size=1000, max_iter=100)
nmf.fit(emg_abs)

# Save the NMF model
joblib.dump(nmf, 'nmf_model.pkl')
print("NMF model trained and saved successfully!")
```

---

## File: v3.0\verify_splits.py
**Path:** `C:\stage\stage\v3.0\verify_splits.py`

```python
import numpy as np
import pandas as pd

# Load the indices
train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')
test_indices = np.load('test_indices.npy')

# Load the subject master list
master_list = pd.read_csv('subject_master_list.csv')

# Function to find which subject an index belongs to
def find_subject_for_index(index, master_list):
    for _, row in master_list.iterrows():
        if row['global_start_index'] <= index <= row['global_end_index']:
            return row['unique_id']
    return None

# Check for overlap between sets
print("Checking for data leakage...")

# Check train-val overlap
train_val_overlap = np.intersect1d(train_indices, val_indices)
print(f"Train-Val overlap: {len(train_val_overlap)} indices")

# Check train-test overlap
train_test_overlap = np.intersect1d(train_indices, test_indices)
print(f"Train-Test overlap: {len(train_test_overlap)} indices")

# Check val-test overlap
val_test_overlap = np.intersect1d(val_indices, test_indices)
print(f"Val-Test overlap: {len(val_test_overlap)} indices")

# Check subject distribution in each set
def get_subjects_in_set(indices, master_list):
    subjects = set()
    # Check a representative sample of indices
    sample_indices = np.random.choice(indices, min(10000, len(indices)), replace=False)
    for idx in sample_indices:
        subject = find_subject_for_index(idx, master_list)
        if subject:
            subjects.add(subject)
    return subjects

train_subjects = get_subjects_in_set(train_indices, master_list)
val_subjects = get_subjects_in_set(val_indices, master_list)
test_subjects = get_subjects_in_set(test_indices, master_list)

print(f"Unique subjects in training set: {len(train_subjects)}")
print(f"Unique subjects in validation set: {len(val_subjects)}")
print(f"Unique subjects in test set: {len(test_subjects)}")

# Check for subject overlap between sets
print(f"Train-Val subject overlap: {len(train_subjects.intersection(val_subjects))}")
print(f"Train-Test subject overlap: {len(train_subjects.intersection(test_subjects))}")
print(f"Val-Test subject overlap: {len(val_subjects.intersection(test_subjects))}")

# If we're only seeing 1 subject per set, let's debug further
if len(train_subjects) == 1:
    print("\nDebugging training set:")
    sample_idx = train_indices[0]
    subject = find_subject_for_index(sample_idx, master_list)
    print(f"First index {sample_idx} belongs to subject: {subject}")
    
    # Check the actual subject list from create_data_splits.py
    train_subjects_list = np.load('train_subjects.npy', allow_pickle=True) if os.path.exists('train_subjects.npy') else "Not available"
    print(f"Intended training subjects: {train_subjects_list}")
```

---

## File: v4.0\convert_to_tfrecord.py
**Path:** `C:\stage\stage\v4.0\convert_to_tfrecord.py`

```python
import tensorflow as tf
import numpy as np
import h5py
from tqdm import tqdm
import os

def convert_to_tfrecord(h5_nmf_file, h5_glove_file, indices_file, output_file, sequence_length, target_shift):
    """Convert HDF5 data to TFRecord format"""
    # Load indices
    indices = np.load(indices_file)
    
    # Load normalization parameters
    glove_min = np.load('glove_min.npy')
    glove_range = np.load('glove_range.npy')
    
    with h5py.File(h5_nmf_file, 'r') as nmf_f, h5py.File(h5_glove_file, 'r') as glove_f, \
         tf.io.TFRecordWriter(output_file) as writer:
        
        nmf_data = nmf_f['nmf_features']
        glove_data = glove_f['labels']
        
        # Filter indices to ensure they're valid
        valid_indices = []
        for idx in indices:
            if idx + sequence_length + target_shift < len(glove_data):
                valid_indices.append(idx)
        
        print(f"Converting {len(valid_indices)} sequences to TFRecord...")
        
        for idx in tqdm(valid_indices):
            # Get NMF sequence
            nmf_seq = nmf_data[idx:idx+sequence_length]
            
            # Get target glove data
            target_idx = idx + sequence_length + target_shift
            glove_target = glove_data[target_idx]
            
            # Normalize glove data
            glove_normalized = (glove_target - glove_min) / glove_range
            
            # Create TFExample
            feature = {
                'nmf_sequence': tf.train.Feature(float_list=tf.train.FloatList(value=nmf_seq.flatten())),
                'glove_target': tf.train.Feature(float_list=tf.train.FloatList(value=glove_normalized)),
                'sequence_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=nmf_seq.shape)),
            }
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# Convert all splits
sequence_length = 500
target_shift = 300

print("Converting training data...")
convert_to_tfrecord(
    r'D:\stage\data\x_nmf.h5',
    r'D:\stage\data\cleaned_y.h5',
    'train_indices.npy',
    'train_data.tfrecord',
    sequence_length,
    target_shift
)

print("Converting validation data...")
convert_to_tfrecord(
    r'D:\stage\data\x_nmf.h5',
    r'D:\stage\data\cleaned_y.h5',
    'val_indices.npy',
    'val_data.tfrecord',
    sequence_length,
    target_shift
)

print("Converting test data...")
convert_to_tfrecord(
    r'D:\stage\data\x_nmf.h5',
    r'D:\stage\data\cleaned_y.h5',
    'test_indices.npy',
    'test_data.tfrecord',
    sequence_length,
    target_shift
)

print("Conversion complete!")
```

---

## File: v4.0\tfrecord_data_loader.py
**Path:** `C:\stage\stage\v4.0\tfrecord_data_loader.py`

```python
import tensorflow as tf
import numpy as np

def parse_tfrecord_fn(example_proto):
    """Parse TFRecord examples"""
    feature_description = {
        'nmf_sequence': tf.io.FixedLenFeature([500*6], tf.float32),
        'glove_target': tf.io.FixedLenFeature([18], tf.float32),
        'sequence_shape': tf.io.FixedLenFeature([2], tf.int64),
    }
    
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Reshape sequence
    sequence = tf.reshape(example['nmf_sequence'], (500, 6))
    target = example['glove_target']
    
    return sequence, target

def create_dataset(tfrecord_file, batch_size=32, shuffle=True, buffer_size=10000):
    """Create TensorFlow Dataset from TFRecord file"""
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Utility function to get dataset sizes
def get_dataset_size(tfrecord_file):
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_file):
        count += 1
    return count
```

---

## File: v4.0\train_tfrecord.py
**Path:** `C:\stage\stage\v4.0\train_tfrecord.py`

```python
from tfrecord_data_loader import create_dataset, get_dataset_size
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Parameters
batch_size = 64  # You can increase this with TFRecord format
sequence_length = 500
target_shift = 300

# Create datasets
train_dataset = create_dataset('train_data.tfrecord', batch_size=batch_size, shuffle=True)
val_dataset = create_dataset('val_data.tfrecord', batch_size=batch_size, shuffle=False)

# Optional: Print dataset sizes for verification
print(f"Training samples: {get_dataset_size('train_data.tfrecord')}")
print(f"Validation samples: {get_dataset_size('val_data.tfrecord')}")

# Build model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 6), return_sequences=True,
         kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    Dropout(0.4),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(18, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model_tfrecord.h5', monitor='val_loss', save_best_only=True)

# Train model
print("Training with TFRecord data...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Save final model
model.save('prosthetic_hand_model_tfrecord.h5')
print("Model saved as prosthetic_hand_model_tfrecord.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('training_history_tfrecord.png')
plt.show()
```

---

## File: v5.0\analyze_emg_noise.py
**Path:** `C:\stage\stage\v5.0\analyze_emg_noise.py`

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns

def analyze_emg_noise(x_file, restimulus_file, sample_duration=10, channels_to_analyze=[0, 1, 2]):
    """
    Analyze EMG data to identify noise frequencies and determine optimal filtering parameters.
    
    Args:
        x_file: Path to EMG HDF5 file
        restimulus_file: Path to restimulus HDF5 file  
        sample_duration: Duration in seconds to analyze (default 10s)
        channels_to_analyze: List of channel indices to analyze
    """
    
    print("Loading EMG data for noise analysis...")
    
    with h5py.File(x_file, 'r') as f_x, h5py.File(restimulus_file, 'r') as f_r:
        emg_data = f_x['data']
        restimulus = f_r['restimulus'][:]
        
        # Get sampling rate from data shape (assuming 2000 Hz as in v1.0)
        n_samples = emg_data.shape[0]
        sampling_rate = 2000  # Hz - adjust if different
        
        # Take a sample of data for analysis
        sample_samples = int(sample_duration * sampling_rate)
        if sample_samples > n_samples:
            sample_samples = n_samples
            
        print(f"Analyzing {sample_duration}s of data ({sample_samples} samples)")
        
        # Get sample data
        emg_sample = emg_data[:sample_samples, :]
        restimulus_sample = restimulus[:sample_samples]
        
        # Separate rest and movement periods
        rest_indices = np.where(restimulus_sample == 0)[0]
        movement_indices = np.where(restimulus_sample != 0)[0]
        
        print(f"Rest samples: {len(rest_indices)} ({len(rest_indices)/len(restimulus_sample)*100:.1f}%)")
        print(f"Movement samples: {len(movement_indices)} ({len(movement_indices)/len(restimulus_sample)*100:.1f}%)")
        
        # Analyze each channel
        fig, axes = plt.subplots(len(channels_to_analyze), 3, figsize=(15, 4*len(channels_to_analyze)))
        if len(channels_to_analyze) == 1:
            axes = axes.reshape(1, -1)
        
        noise_analysis_results = {}
        
        for i, channel in enumerate(channels_to_analyze):
            print(f"\nAnalyzing channel {channel}...")
            
            # Get channel data
            channel_data = emg_sample[:, channel]
            
            # Time domain plot
            time_axis = np.arange(len(channel_data)) / sampling_rate
            axes[i, 0].plot(time_axis, channel_data)
            axes[i, 0].set_title(f'Channel {channel} - Time Domain')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True)
            
            # Frequency domain analysis
            # Use FFT to get frequency spectrum
            fft_data = fft(channel_data)
            freqs = fftfreq(len(channel_data), 1/sampling_rate)
            
            # Only plot positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft_data[:len(fft_data)//2])
            
            # Plot frequency spectrum
            axes[i, 1].plot(positive_freqs, positive_fft)
            axes[i, 1].set_title(f'Channel {channel} - Frequency Spectrum')
            axes[i, 1].set_xlabel('Frequency (Hz)')
            axes[i, 1].set_ylabel('Magnitude')
            axes[i, 1].set_xlim(0, 500)  # Focus on 0-500 Hz range
            axes[i, 1].grid(True)
            
            # Log scale for better visibility
            axes[i, 2].semilogy(positive_freqs, positive_fft)
            axes[i, 2].set_title(f'Channel {channel} - Log Scale Spectrum')
            axes[i, 2].set_xlabel('Frequency (Hz)')
            axes[i, 2].set_ylabel('Magnitude (log)')
            axes[i, 2].set_xlim(0, 500)
            axes[i, 2].grid(True)
            
            # Mark common noise frequencies
            for freq in [50, 60, 100, 120, 150, 180, 200, 250, 300, 400]:
                if freq < sampling_rate/2:
                    axes[i, 1].axvline(freq, color='red', linestyle='--', alpha=0.7, label=f'{freq}Hz' if freq == 50 else "")
                    axes[i, 2].axvline(freq, color='red', linestyle='--', alpha=0.7)
            
            # Analyze specific frequency bands
            print(f"Channel {channel} frequency analysis:")
            
            # Check for 50/60 Hz and harmonics
            for base_freq in [50, 60]:
                for harmonic in range(1, 6):
                    freq = base_freq * harmonic
                    if freq < sampling_rate/2:
                        # Find closest frequency bin
                        freq_idx = np.argmin(np.abs(positive_freqs - freq))
                        magnitude = positive_fft[freq_idx]
                        print(f"  {freq}Hz: magnitude = {magnitude:.2f}")
            
            # Find peaks in the spectrum
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(positive_fft, height=np.max(positive_fft)*0.1, distance=10)
            significant_peaks = []
            for peak_idx in peaks:
                freq = positive_freqs[peak_idx]
                magnitude = positive_fft[peak_idx]
                if freq > 5 and freq < 500:  # Ignore DC and very high frequencies
                    significant_peaks.append((freq, magnitude))
            
            # Sort by magnitude
            significant_peaks.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top frequency peaks:")
            for freq, mag in significant_peaks[:10]:
                print(f"    {freq:.1f}Hz: {mag:.2f}")
            
            # Store results
            noise_analysis_results[channel] = {
                'frequencies': positive_freqs,
                'magnitude': positive_fft,
                'significant_peaks': significant_peaks[:10]
            }
        
        plt.tight_layout()
        plt.savefig('emg_noise_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return noise_analysis_results, sampling_rate

def recommend_filters(noise_analysis_results, sampling_rate):
    """
    Recommend filter parameters based on noise analysis
    """
    print("\n" + "="*50)
    print("FILTER RECOMMENDATIONS")
    print("="*50)
    
    # Analyze all channels for common noise patterns
    all_peaks = []
    for channel, results in noise_analysis_results.items():
        for freq, mag in results['significant_peaks']:
            all_peaks.append((freq, mag, channel))
    
    # Group peaks by frequency (within 2Hz tolerance)
    frequency_groups = {}
    for freq, mag, channel in all_peaks:
        found_group = False
        for group_freq in frequency_groups:
            if abs(freq - group_freq) <= 2:
                frequency_groups[group_freq].append((freq, mag, channel))
                found_group = True
                break
        if not found_group:
            frequency_groups[freq] = [(freq, mag, channel)]
    
    # Find consistent noise frequencies across channels
    consistent_noise = []
    for freq, peaks in frequency_groups.items():
        if len(peaks) >= 2:  # Present in at least 2 channels
            avg_magnitude = np.mean([mag for _, mag, _ in peaks])
            consistent_noise.append((freq, avg_magnitude, len(peaks)))
    
    # Sort by magnitude
    consistent_noise.sort(key=lambda x: x[1], reverse=True)
    
    print("Consistent noise frequencies across channels:")
    for freq, mag, count in consistent_noise[:10]:
        print(f"  {freq:.1f}Hz: magnitude={mag:.2f}, present in {count} channels")
    
    # Determine mains frequency
    mains_candidates = []
    for freq, mag, count in consistent_noise:
        if 48 <= freq <= 52:  # 50Hz region
            mains_candidates.append((freq, mag, count, 50))
        elif 58 <= freq <= 62:  # 60Hz region
            mains_candidates.append((freq, mag, count, 60))
    
    if mains_candidates:
        # Pick the strongest mains frequency
        mains_candidates.sort(key=lambda x: x[1], reverse=True)
        mains_freq, mains_mag, mains_count, mains_type = mains_candidates[0]
        print(f"\nDetected mains frequency: {mains_freq:.1f}Hz ({mains_type}Hz system)")
    else:
        print("\nNo clear mains frequency detected. Defaulting to 60Hz.")
        mains_freq = 60
        mains_type = 60
    
    # Recommend notch filter
    print(f"\nNotch filter recommendation:")
    print(f"  Center frequency: {mains_freq:.1f}Hz")
    print(f"  Quality factor: 30 (bandwidth ≈ {mains_freq/30:.1f}Hz)")
    
    # Check for harmonics
    harmonics = []
    for freq, mag, count in consistent_noise:
        for harmonic in range(2, 6):
            expected_harmonic = mains_freq * harmonic
            if abs(freq - expected_harmonic) <= 2:
                harmonics.append((freq, mag, harmonic))
    
    if harmonics:
        print(f"  Harmonics detected: {[f'{freq:.1f}Hz (x{h})' for freq, mag, h in harmonics]}")
        print(f"  Consider multiple notch filters for harmonics")
    
    # Recommend bandpass filter
    print(f"\nBandpass filter recommendation:")
    print(f"  Low cutoff: 20Hz (removes motion artifacts, DC drift)")
    print(f"  High cutoff: 450Hz (preserves EMG signal, removes high-frequency noise)")
    print(f"  Filter type: 4th-order Butterworth")
    print(f"  Implementation: scipy.signal.butter(4, [20/({sampling_rate}/2), 450/({sampling_rate}/2)], btype='band')")
    
    # Check if current settings are appropriate
    nyquist = sampling_rate / 2
    if 450 >= nyquist * 0.9:
        print(f"  WARNING: High cutoff (450Hz) is close to Nyquist frequency ({nyquist}Hz)")
        print(f"  Consider reducing high cutoff to {int(nyquist * 0.8)}Hz")
    
    return {
        'mains_frequency': mains_freq,
        'mains_type': mains_type,
        'harmonics': harmonics,
        'notch_freq': mains_freq,
        'notch_Q': 30,
        'bandpass_low': 20,
        'bandpass_high': min(450, int(nyquist * 0.8)),
        'filter_order': 4
    }

def analyze_tfrecord_data(tfrecord_path, sample_size=1000):
    """
    Analyze a sample of TFRecord data to understand the actual schema
    """
    import tensorflow as tf
    
    def parse_tfrecord_fn(example_proto):
        # Based on creation_final.py, the TFRecords contain glove data with features glove_0 to glove_17
        feature_description = {
            f'glove_{i}': tf.io.FixedLenFeature([], tf.float32) for i in range(18)
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        
        # Extract glove values and create a single array
        glove_values = [example[f'glove_{i}'] for i in range(18)]
        return tf.stack(glove_values)
    
    print(f"Analyzing TFRecord data from {tfrecord_path}...")
    
    # Load a sample of TFRecord data
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn)
    dataset = dataset.take(sample_size)
    
    glove_data = []
    
    for glove_sample in dataset:
        glove_data.append(glove_sample.numpy())
    
    glove_data = np.array(glove_data)
    
    print(f"Loaded {len(glove_data)} glove samples")
    print(f"Glove data shape: {glove_data.shape}")
    
    # Analyze glove data
    print("\nGlove Data Analysis:")
    print(f"Mean glove values: {np.mean(glove_data, axis=0)}")
    print(f"Std glove values: {np.std(glove_data, axis=0)}")
    print(f"Min glove values: {np.min(glove_data, axis=0)}")
    print(f"Max glove values: {np.max(glove_data, axis=0)}")
    
    # Check for NaN/Inf values
    print(f"\nData Quality Check:")
    print(f"NaN values in glove: {np.isnan(glove_data).sum()}")
    print(f"Inf values in glove: {np.isinf(glove_data).sum()}")
    
    return glove_data

def analyze_all_tfrecords(tfrecords_dir, sample_files=10, sample_per_file=100):
    """
    Analyze multiple TFRecord files to understand data distribution
    """
    import tensorflow as tf
    import os
    import glob
    
    def parse_tfrecord_fn(example_proto):
        # Based on creation_final.py, the TFRecords contain glove data with features glove_0 to glove_17
        feature_description = {
            f'glove_{i}': tf.io.FixedLenFeature([], tf.float32) for i in range(18)
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        
        # Extract glove values and create a single array
        glove_values = [example[f'glove_{i}'] for i in range(18)]
        return tf.stack(glove_values)
    
    # Get list of TFRecord files
    tfrecord_files = glob.glob(os.path.join(tfrecords_dir, "data_*.tfrecord"))
    tfrecord_files.sort()
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    print(f"Analyzing {min(sample_files, len(tfrecord_files))} files with {sample_per_file} samples each")
    
    # Sample files for analysis
    sample_files_list = tfrecord_files[:sample_files]
    
    all_glove_data = []
    file_stats = []
    
    for i, tfrecord_file in enumerate(sample_files_list):
        print(f"\nAnalyzing file {i+1}/{len(sample_files_list)}: {os.path.basename(tfrecord_file)}")
        
        try:
            # Load sample from this file
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            dataset = dataset.map(parse_tfrecord_fn)
            dataset = dataset.take(sample_per_file)
            
            glove_data = []
            
            for glove_sample in dataset:
                glove_data.append(glove_sample.numpy())
            
            glove_data = np.array(glove_data)
            
            # Store for overall analysis
            all_glove_data.append(glove_data)
            
            # File-specific stats
            file_stat = {
                'file': os.path.basename(tfrecord_file),
                'samples': len(glove_data),
                'glove_mean': np.mean(glove_data, axis=0),
                'glove_std': np.std(glove_data, axis=0),
                'glove_min': np.min(glove_data, axis=0),
                'glove_max': np.max(glove_data, axis=0),
                'nan_count': np.isnan(glove_data).sum(),
                'inf_count': np.isinf(glove_data).sum()
            }
            file_stats.append(file_stat)
            
            print(f"  Samples: {len(glove_data)}")
            print(f"  Glove mean: {np.mean(glove_data, axis=0)}")
            print(f"  Glove std: {np.std(glove_data, axis=0)}")
            print(f"  NaN/Inf: {file_stat['nan_count']}/{file_stat['inf_count']}")
            
        except Exception as e:
            print(f"  Error processing {tfrecord_file}: {e}")
            continue
    
    # Overall analysis
    if all_glove_data:
        print(f"\n" + "="*60)
        print("OVERALL ANALYSIS ACROSS ALL FILES")
        print("="*60)
        
        # Combine all data
        combined_glove = np.vstack(all_glove_data)
        
        print(f"Total samples analyzed: {len(combined_glove)}")
        print(f"Total files processed: {len(all_glove_data)}")
        
        # Glove analysis
        print(f"\nGlove Data (across all files):")
        print(f"  Mean: {np.mean(combined_glove, axis=0)}")
        print(f"  Std: {np.std(combined_glove, axis=0)}")
        print(f"  Min: {np.min(combined_glove, axis=0)}")
        print(f"  Max: {np.max(combined_glove, axis=0)}")
        
        # Data quality
        print(f"\nData Quality (across all files):")
        print(f"  NaN in Glove: {np.isnan(combined_glove).sum()}")
        print(f"  Inf in Glove: {np.isinf(combined_glove).sum()}")
        
        # File-to-file consistency
        print(f"\nFile-to-File Consistency:")
        glove_means = [stat['glove_mean'] for stat in file_stats]
        
        if len(glove_means) > 1:
            glove_std_across_files = np.std(glove_means, axis=0)
            print(f"  Glove mean std across files: {glove_std_across_files}")
        
        return {
            'combined_glove': combined_glove,
            'file_stats': file_stats,
            'total_samples': len(combined_glove),
            'total_files': len(all_glove_data)
        }
    
    return None

if __name__ == "__main__":
    # Analyze multiple TFRecord files to understand data distribution and preprocessing
    tfrecords_dir = r'D:\stage\v5.0\db\tfrecords'
    
    try:
        print("="*60)
        print("COMPREHENSIVE TFRecord ANALYSIS")
        print("="*60)
        
        # Analyze multiple files to understand data distribution
        analysis_results = analyze_all_tfrecords(
            tfrecords_dir, 
            sample_files=20,  # Analyze 20 files
            sample_per_file=200  # 200 samples per file
        )
        
        if analysis_results:
            print(f"\n" + "="*60)
            print("DATA UNDERSTANDING")
            print("="*60)
            print("Based on the analysis, the TFRecords contain:")
            print("1. Raw glove data only (18 sensors: glove_0 to glove_17)")
            print("2. No EMG data, no NMF features, no sequences")
            print("3. This is just the glove targets from the original databases")
            print("4. No preprocessing has been applied yet")
            
            print(f"\nData Quality Assessment:")
            if analysis_results['combined_glove'].size > 0:
                nan_glove = np.isnan(analysis_results['combined_glove']).sum()
                inf_glove = np.isinf(analysis_results['combined_glove']).sum()
                
                print(f"✓ Glove data quality: {nan_glove} NaN, {inf_glove} Inf values")
                
                if nan_glove == 0 and inf_glove == 0:
                    print("✓ Excellent data quality - no NaN/Inf values detected")
                else:
                    print("⚠ Data quality issues detected - may need cleaning")
            
            print(f"\nDatabase Configuration:")
            print(f"✓ Sampling rate: 2000 Hz (from database guides)")
            print(f"✓ EMG channels: 12 (Delsys Trigno electrodes)")
            print(f"✓ Databases: DB2, DB3, DB7 (2 exercises each)")
            print(f"✓ Total files: {analysis_results['total_files']}")
            print(f"✓ Total samples analyzed: {analysis_results['total_samples']}")
            
            print(f"\nCurrent Status:")
            print(f"⚠ These TFRecords contain ONLY glove data")
            print(f"⚠ No EMG preprocessing has been done yet")
            print(f"⚠ No NMF features, no sequences, no filtering")
            
            print(f"\nWhat We Need to Do for v5.0:")
            print(f"1. Get the raw EMG data (not in these TFRecords)")
            print(f"2. Apply EMG filtering: 60 Hz notch + 20-450 Hz bandpass")
            print(f"3. Apply EMG normalization: z-score using rest periods")
            print(f"4. Apply EMG rectification: absolute value")
            print(f"5. Train NMF model: 6 components on movement data")
            print(f"6. Create sequences: 500 samples with 300 target shift")
            print(f"7. Generate new TFRecords with preprocessed EMG sequences + glove targets")
            
            print(f"\nNext Steps:")
            print(f"1. Find the raw EMG data source files")
            print(f"2. Implement the full preprocessing pipeline")
            print(f"3. Create new TFRecords with EMG sequences + glove targets")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

```

---

## File: v5.0\analyze_glove_data.py
**Path:** `C:\stage\stage\v5.0\analyze_glove_data.py`

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Union

TRAIN_MAP_CSV = r"D:\stage\v5.0\logs\map\train_map.csv"
SAMPLE_ROWS = 100000  # rows to read per file for analysis (adjust if needed)
OUTPUT_DIR = "glove"  # Folder in the code directory


def find_glove_columns(df: pd.DataFrame) -> list:
    """Detect glove columns by prefix 'glove_' and sort by numeric index."""
    glove_cols = [c for c in df.columns if c.startswith('glove_')]
    if not glove_cols:
        return []
    # Sort by numeric suffix if present
    try:
        glove_cols_sorted = sorted(glove_cols, key=lambda x: int(x.split('_')[1]))
        return glove_cols_sorted
    except Exception:
        return sorted(glove_cols)


def analyze_glove_file(csv_path: str, output_base_dir: str, sample_rows: int = SAMPLE_ROWS) -> dict:
    print(f"Analyzing glove data from {csv_path}...")
    
    # Create a safe filename for the output folder (replace problematic characters)
    safe_filename = os.path.basename(csv_path).replace('.', '_').replace('\\', '_').replace('/', '_')
    out_dir = os.path.join(output_base_dir, safe_filename)
    os.makedirs(out_dir, exist_ok=True)
    
    base = os.path.splitext(os.path.basename(csv_path))[0]

    # Read a sample of rows for quick analysis
    try:
        df = pd.read_csv(csv_path, nrows=sample_rows)
    except Exception as e:
        print(f"  Error reading {csv_path}: {e}")
        return {}

    glove_cols = find_glove_columns(df)
    if not glove_cols:
        print("  No glove columns found.")
        return {}

    restimulus_present = 'restimulus' in df.columns
    rest_mask = (df['restimulus'] == 0) if restimulus_present else pd.Series([False] * len(df))
    move_mask = (df['restimulus'] != 0) if restimulus_present else pd.Series([True] * len(df))

    stats_rows = []
    # Per-sensor plots and stats
    glitch_rows = []
    for g in glove_cols:
        series = pd.to_numeric(df[g], errors='coerce')
        overall = series.values
        rest_vals = series.values[rest_mask.values]
        move_vals = series.values[move_mask.values]

        # Compute robust quantiles
        def safe_stats(x: np.ndarray):
            x = x[~np.isnan(x) & ~np.isinf(x)]
            if x.size == 0:
                return {
                    'mean': np.nan, 'std': np.nan, 'min': np.nan, 'p1': np.nan,
                    'p50': np.nan, 'p99': np.nan, 'max': np.nan, 'nan': np.sum(np.isnan(x)), 'inf': 0
                }
            return {
                'mean': float(np.mean(x)),
                'std': float(np.std(x)),
                'min': float(np.min(x)),
                'p1': float(np.percentile(x, 1)),
                'p50': float(np.percentile(x, 50)),
                'p99': float(np.percentile(x, 99)),
                'max': float(np.max(x)),
                'nan': int(np.sum(np.isnan(series.values))),
                'inf': int(np.isinf(series.values).sum())
            }

        o_stats = safe_stats(overall)
        r_stats = safe_stats(rest_vals)
        m_stats = safe_stats(move_vals)

        # Glitch detection
        # Define robust in-range band using p1-p99 from overall
        band_low = o_stats['p1']
        band_high = o_stats['p99']
        # Absolute hard caps (catch absurd values)
        hard_low = np.nanmin([o_stats['min'], band_low])
        hard_high = np.nanmax([o_stats['max'], band_high])
        # Compute masks
        def mask_valid(x):
            return (~np.isnan(x)) & (~np.isinf(x))
        ov_mask = mask_valid(overall)
        re_mask = mask_valid(rest_vals)
        mv_mask = mask_valid(move_vals)
        out_of_band_overall = ((overall < band_low) | (overall > band_high)) & ov_mask
        out_of_band_rest = ((rest_vals < band_low) | (rest_vals > band_high)) & re_mask
        out_of_band_move = ((move_vals < band_low) | (move_vals > band_high)) & mv_mask

        # Count spikes far beyond band (e.g., > 10x IQR proxy)
        iqr = (o_stats['p99'] - o_stats['p1'])
        spike_thresh_high = o_stats['p99'] + 10 * iqr if np.isfinite(iqr) else o_stats['p99']
        spike_thresh_low = o_stats['p1'] - 10 * iqr if np.isfinite(iqr) else o_stats['p1']
        spikes_overall = ((overall > spike_thresh_high) | (overall < spike_thresh_low)) & ov_mask
        spikes_rest = ((rest_vals > spike_thresh_high) | (rest_vals < spike_thresh_low)) & re_mask
        spikes_move = ((move_vals > spike_thresh_high) | (move_vals < spike_thresh_low)) & mv_mask

        stats_rows.append({
            'sensor': g,
            'overall_mean': o_stats['mean'], 'overall_std': o_stats['std'], 'overall_min': o_stats['min'], 'overall_p1': o_stats['p1'], 'overall_p50': o_stats['p50'], 'overall_p99': o_stats['p99'], 'overall_max': o_stats['max'], 'overall_nan': o_stats['nan'], 'overall_inf': o_stats['inf'],
            'rest_mean': r_stats['mean'], 'rest_std': r_stats['std'], 'rest_min': r_stats['min'], 'rest_p1': r_stats['p1'], 'rest_p50': r_stats['p50'], 'rest_p99': r_stats['p99'], 'rest_max': r_stats['max'],
            'move_mean': m_stats['mean'], 'move_std': m_stats['std'], 'move_min': m_stats['min'], 'move_p1': m_stats['p1'], 'move_p50': m_stats['p50'], 'move_p99': m_stats['p99'], 'move_max': m_stats['max'],
            'band_low': band_low, 'band_high': band_high,
            'out_of_band_overall': int(out_of_band_overall.sum()),
            'out_of_band_rest': int(out_of_band_rest.sum()),
            'out_of_band_move': int(out_of_band_move.sum()),
            'spikes_overall': int(spikes_overall.sum()),
            'spikes_rest': int(spikes_rest.sum()),
            'spikes_move': int(spikes_move.sum())
        })

        # Record glitch positions summary (sampled)
        def first_n_indices(mask, n=10):
            idx = np.where(mask)[0]
            return idx[:n].tolist()
        glitch_rows.append({
            'sensor': g,
            'first_oob_idx_overall': first_n_indices(out_of_band_overall),
            'first_spike_idx_overall': first_n_indices(spikes_overall),
        })

        # Plot histograms (overall/rest/move) per sensor
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, data, title in zip(axes, [overall, rest_vals, move_vals], ['Overall', 'Rest', 'Movement']):
            clean = data[~np.isnan(data) & ~np.isinf(data)]
            if clean.size > 0:
                # Plot unclipped and clipped hist overlays
                ax.hist(clean, bins=50, color='#4C78A8', alpha=0.6, label='raw')
                clipped = clean[(clean >= band_low) & (clean <= band_high)]
                if clipped.size > 0:
                    ax.hist(clipped, bins=50, color='#F58518', alpha=0.6, label='within p1-p99')
                ax.legend()
            ax.set_title(title)
            ax.grid(True)
        fig.suptitle(f'{g} distribution')
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f'{base}.{g}.hist.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Save summary CSV
    summary_csv = os.path.join(out_dir, f'{base}.glove_summary.csv')
    pd.DataFrame(stats_rows).to_csv(summary_csv, index=False)
    print(f"  Saved: {summary_csv}")

    glitches_csv = os.path.join(out_dir, f'{base}.glove_glitches.csv')
    pd.DataFrame(glitch_rows).to_csv(glitches_csv, index=False)
    print(f"  Saved: {glitches_csv}")

    # Optional: overall heatmap of means/stds
    try:
        means = [row['overall_mean'] for row in stats_rows]
        stds = [row['overall_std'] for row in stats_rows]
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3))
        ax2[0].bar(range(len(glove_cols)), means)
        ax2[0].set_title('Overall mean per sensor')
        ax2[1].bar(range(len(glove_cols)), stds)
        ax2[1].set_title('Overall std per sensor')
        for a in ax2:
            a.grid(True)
        fig2.tight_layout()
        fig2_path = os.path.join(out_dir, f'{base}.glove_overview.png')
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
    except Exception:
        pass

    return {'file': csv_path, 'glove_cols': glove_cols, 'summary_csv': summary_csv}


def analyze_from_map(train_map_csv: str = TRAIN_MAP_CSV, limit: Optional[int] = None):
    # Create the main output directory in the same folder as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Output directory: {output_base_dir}")
    
    df_map = pd.read_csv(train_map_csv)
    n = len(df_map) if limit is None else min(limit, len(df_map))
    print(f"Found {len(df_map)} files in map; analyzing {n}...")
    results = []
    rows_iter = list(df_map.iterrows()) if limit is None else list(df_map.iterrows())[:limit]
    for i, row in tqdm(rows_iter, total=n):
        csv_path = row['filename']
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        res = analyze_glove_file(csv_path, output_base_dir)
        if res:
            results.append(res)
    return results


if __name__ == "__main__":
    print("Glove Data Analysis (rest vs movement)")
    print("=" * 60)
    # Set limit=None to process all files listed in train_map.csv
    analyze_from_map(TRAIN_MAP_CSV, limit=None)
```

---

## File: v5.0\analyze_raw_emg_data.py
**Path:** `C:\stage\stage\v5.0\analyze_raw_emg_data.py`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
from tqdm import tqdm

def analyze_emg_noise_from_csv(csv_file, sample_duration=10, channels_to_analyze=None, output_dir=None):
    """
    Analyze EMG data from CSV file to identify noise frequencies and determine optimal filtering parameters.
    
    Args:
        csv_file: Path to CSV file
        sample_duration: Duration in seconds to analyze (default 10s)
        channels_to_analyze: List of channel indices to analyze
    """
    
    print(f"Loading EMG data from {csv_file}...")
    
    try:
        # Read a sample of the CSV file
        chunk_size = int(sample_duration * 2000)  # 2000 Hz sampling rate
        df_sample = pd.read_csv(csv_file, nrows=chunk_size)
        
        # Get EMG columns (assuming they start with 'emg_')
        emg_columns = [col for col in df_sample.columns if col.startswith('emg_')]
        restimulus_col = 'restimulus' if 'restimulus' in df_sample.columns else None
        
        if not emg_columns:
            print(f"No EMG columns found in {csv_file}")
            return None, None
        
        print(f"Found {len(emg_columns)} EMG channels: {emg_columns}")
        print(f"Restimulus column: {restimulus_col}")
        
        # Get EMG data
        emg_data = df_sample[emg_columns].values.astype(np.float32)
        
        # Get restimulus data if available
        restimulus_data = None
        if restimulus_col:
            restimulus_data = df_sample[restimulus_col].values
            rest_indices = np.where(restimulus_data == 0)[0]
            movement_indices = np.where(restimulus_data != 0)[0]
            print(f"Rest samples: {len(rest_indices)} ({len(rest_indices)/len(restimulus_data)*100:.1f}%)")
            print(f"Movement samples: {len(movement_indices)} ({len(movement_indices)/len(restimulus_data)*100:.1f}%)")
        
        # Channels to analyze: default to all
        if channels_to_analyze is None:
            channels_to_analyze = list(range(len(emg_columns)))

        # Prepare output directory
        if output_dir is None:
            output_dir = os.path.dirname(csv_file)
        os.makedirs(output_dir, exist_ok=True)
        
        noise_analysis_results = {}
        sampling_rate = 2000  # Hz - from database guides
        
        # Collect per-channel spectra for summary grid
        channel_summaries = []

        for i, channel in enumerate(channels_to_analyze):
            if channel >= len(emg_columns):
                continue
                
            print(f"\nAnalyzing channel {channel} ({emg_columns[channel]})...")
            
            # Get channel data
            channel_data = emg_data[:, channel]
            
            # Frequency domain analysis
            # Use FFT to get frequency spectrum
            fft_data = fft(channel_data)
            freqs = fftfreq(len(channel_data), 1/sampling_rate)
            
            # Only plot positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft_data[:len(fft_data)//2])
            
            # Save per-channel figure (time, linear spectrum, log spectrum)
            time_axis = np.arange(len(channel_data)) / sampling_rate
            fig_ch, ax_ch = plt.subplots(3, 1, figsize=(12, 8))
            # Time
            ax_ch[0].plot(time_axis, channel_data)
            ax_ch[0].set_title(f'Channel {channel} ({emg_columns[channel]}) - Time Domain')
            ax_ch[0].set_xlabel('Time (s)')
            ax_ch[0].set_ylabel('Amplitude')
            ax_ch[0].grid(True)
            # Linear spectrum
            ax_ch[1].plot(positive_freqs, positive_fft)
            ax_ch[1].set_title('Frequency Spectrum')
            ax_ch[1].set_xlabel('Frequency (Hz)')
            ax_ch[1].set_ylabel('Magnitude')
            ax_ch[1].set_xlim(0, 500)
            ax_ch[1].grid(True)
            # Log spectrum
            ax_ch[2].semilogy(positive_freqs, positive_fft)
            ax_ch[2].set_title('Log Scale Spectrum')
            ax_ch[2].set_xlabel('Frequency (Hz)')
            ax_ch[2].set_ylabel('Magnitude (log)')
            ax_ch[2].set_xlim(0, 500)
            ax_ch[2].grid(True)
            
            # Mark common noise frequencies
            for freq in [50, 60, 100, 120, 150, 180, 200, 250, 300, 400]:
                if freq < sampling_rate/2:
                    ax_ch[1].axvline(freq, color='red', linestyle='--', alpha=0.7)
                    ax_ch[2].axvline(freq, color='red', linestyle='--', alpha=0.7)

            # Save per-channel plot
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            fig_path = os.path.join(output_dir, f'{base_name}.ch{channel:02d}.png')
            fig_ch.tight_layout()
            fig_ch.savefig(fig_path, dpi=200, bbox_inches='tight')
            plt.close(fig_ch)
            
            # Analyze specific frequency bands
            print(f"Channel {channel} frequency analysis:")
            
            # Check for 50/60 Hz and harmonics
            for base_freq in [50, 60]:
                for harmonic in range(1, 6):
                    freq = base_freq * harmonic
                    if freq < sampling_rate/2:
                        # Find closest frequency bin
                        freq_idx = np.argmin(np.abs(positive_freqs - freq))
                        magnitude = positive_fft[freq_idx]
                        print(f"  {freq}Hz: magnitude = {magnitude:.2f}")
            
            # Find peaks in the spectrum
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(positive_fft, height=np.max(positive_fft)*0.1, distance=10)
            significant_peaks = []
            for peak_idx in peaks:
                freq = positive_freqs[peak_idx]
                magnitude = positive_fft[peak_idx]
                if freq > 5 and freq < 500:  # Ignore DC and very high frequencies
                    significant_peaks.append((freq, magnitude))
            
            # Sort by magnitude
            significant_peaks.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top frequency peaks:")
            for freq, mag in significant_peaks[:10]:
                print(f"    {freq:.1f}Hz: {mag:.2f}")
            
            # Store results
            noise_analysis_results[channel] = {
                'frequencies': positive_freqs,
                'magnitude': positive_fft,
                'significant_peaks': significant_peaks[:10]
            }

            # Add summary entry
            channel_summaries.append((channel, positive_freqs, positive_fft))

        # Save summary grid (12 channels if available)
        cols = 4
        rows = int(np.ceil(len(channel_summaries) / cols)) if channel_summaries else 1
        fig_sum, axes_sum = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
        axes_sum = np.atleast_2d(axes_sum)
        for idx, (ch, faxis, mag) in enumerate(channel_summaries):
            r = idx // cols
            c = idx % cols
            ax = axes_sum[r, c]
            ax.semilogy(faxis, mag)
            ax.set_title(f'ch{ch:02d}')
            ax.set_xlim(0, 500)
            ax.grid(True)
        # Hide any unused subplots
        for idx in range(len(channel_summaries), rows*cols):
            r = idx // cols
            c = idx % cols
            axes_sum[r, c].axis('off')
        fig_sum.tight_layout()
        sum_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(csv_file))[0]}.summary.png')
        fig_sum.savefig(sum_path, dpi=200, bbox_inches='tight')
        plt.close(fig_sum)
        
        return noise_analysis_results, sampling_rate, restimulus_data
        
    except Exception as e:
        print(f"Error analyzing {csv_file}: {e}")
        return None, None, None

def recommend_filters(noise_analysis_results, sampling_rate):
    """
    Recommend filter parameters based on noise analysis
    """
    print("\n" + "="*50)
    print("FILTER RECOMMENDATIONS")
    print("="*50)
    
    # Analyze all channels for common noise patterns
    all_peaks = []
    for channel, results in noise_analysis_results.items():
        for freq, mag in results['significant_peaks']:
            all_peaks.append((freq, mag, channel))
    
    # Group peaks by frequency (within 2Hz tolerance)
    frequency_groups = {}
    for freq, mag, channel in all_peaks:
        found_group = False
        for group_freq in frequency_groups:
            if abs(freq - group_freq) <= 2:
                frequency_groups[group_freq].append((freq, mag, channel))
                found_group = True
                break
        if not found_group:
            frequency_groups[freq] = [(freq, mag, channel)]
    
    # Find consistent noise frequencies across channels
    consistent_noise = []
    for freq, peaks in frequency_groups.items():
        if len(peaks) >= 2:  # Present in at least 2 channels
            avg_magnitude = np.mean([mag for _, mag, _ in peaks])
            consistent_noise.append((freq, avg_magnitude, len(peaks)))
    
    # Sort by magnitude
    consistent_noise.sort(key=lambda x: x[1], reverse=True)
    
    print("Consistent noise frequencies across channels:")
    for freq, mag, count in consistent_noise[:10]:
        print(f"  {freq:.1f}Hz: magnitude={mag:.2f}, present in {count} channels")
    
    # Determine mains frequency
    mains_candidates = []
    for freq, mag, count in consistent_noise:
        if 48 <= freq <= 52:  # 50Hz region
            mains_candidates.append((freq, mag, count, 50))
        elif 58 <= freq <= 62:  # 60Hz region
            mains_candidates.append((freq, mag, count, 60))
    
    if mains_candidates:
        # Pick the strongest mains frequency
        mains_candidates.sort(key=lambda x: x[1], reverse=True)
        mains_freq, mains_mag, mains_count, mains_type = mains_candidates[0]
        print(f"\nDetected mains frequency: {mains_freq:.1f}Hz ({mains_type}Hz system)")
    else:
        print("\nNo clear mains frequency detected. Defaulting to 60Hz.")
        mains_freq = 60
        mains_type = 60
    
    # Recommend notch filter
    print(f"\nNotch filter recommendation:")
    print(f"  Center frequency: {mains_freq:.1f}Hz")
    print(f"  Quality factor: 30 (bandwidth ≈ {mains_freq/30:.1f}Hz)")
    
    # Check for harmonics
    harmonics = []
    for freq, mag, count in consistent_noise:
        for harmonic in range(2, 6):
            expected_harmonic = mains_freq * harmonic
            if abs(freq - expected_harmonic) <= 2:
                harmonics.append((freq, mag, harmonic))
    
    if harmonics:
        print(f"  Harmonics detected: {[f'{freq:.1f}Hz (x{h})' for freq, mag, h in harmonics]}")
        print(f"  Consider multiple notch filters for harmonics")
    
    # Recommend bandpass filter
    print(f"\nBandpass filter recommendation:")
    print(f"  Low cutoff: 20Hz (removes motion artifacts, DC drift)")
    print(f"  High cutoff: 450Hz (preserves EMG signal, removes high-frequency noise)")
    print(f"  Filter type: 4th-order Butterworth")
    print(f"  Implementation: scipy.signal.butter(4, [20/({sampling_rate}/2), 450/({sampling_rate}/2)], btype='band')")
    
    # Check if current settings are appropriate
    nyquist = sampling_rate / 2
    if 450 >= nyquist * 0.9:
        print(f"  WARNING: High cutoff (450Hz) is close to Nyquist frequency ({nyquist}Hz)")
        print(f"  Consider reducing high cutoff to {int(nyquist * 0.8)}Hz")
    
    return {
        'mains_frequency': mains_freq,
        'mains_type': mains_type,
        'harmonics': harmonics,
        'notch_freq': mains_freq,
        'notch_Q': 30,
        'bandpass_low': 20,
        'bandpass_high': min(450, int(nyquist * 0.8)),
        'filter_order': 4
    }

def per_channel_recommendations(noise_analysis_results, sampling_rate):
    """Compute per-channel mains and harmonics recommendations."""
    nyquist = sampling_rate / 2
    recs = {}
    for ch, res in noise_analysis_results.items():
        peaks = res['significant_peaks']
        mains_candidate = None
        # Search around 50 and 60 Hz windows
        best_mag = -np.inf
        for base in [50, 60]:
            for freq, mag in peaks:
                if abs(freq - base) <= 2:
                    if mag > best_mag:
                        best_mag = mag
                        mains_candidate = round(freq, 1)
        # fallback: pick strongest peak in 40-80 Hz if no candidate
        if mains_candidate is None:
            for freq, mag in peaks:
                if 40 <= freq <= 80 and mag > best_mag:
                    best_mag = mag
                    mains_candidate = round(freq, 1)
        if mains_candidate is None:
            mains_candidate = 50.0
        # Detect harmonics (up to 5th)
        harmonics = []
        for h in range(2, 6):
            target = mains_candidate * h
            if target >= nyquist:
                break
            for freq, mag in peaks:
                if abs(freq - target) <= 2:
                    harmonics.append(round(freq, 1))
                    break
        recs[ch] = {
            'notch_freq': mains_candidate,
            'notch_Q': 30,
            'harmonics': harmonics,
            'bandpass_low': 20,
            'bandpass_high': min(450, int(nyquist * 0.8)),
            'filter_order': 4
        }
    return recs

def analyze_multiple_files(train_map_csv, sample_files=3):
    """
    Analyze multiple CSV files from train_map.csv
    """
    print("Loading file list from train_map.csv...")
    df_map = pd.read_csv(train_map_csv)
    
    print(f"Found {len(df_map)} files in train_map.csv")
    print(f"Analyzing {min(sample_files, len(df_map))} files")
    
    all_results = []
    
    for i, (_, row) in enumerate(df_map.iterrows()):
        if i >= sample_files:
            break
            
        csv_file = row['filename']
        print(f"\n{'='*60}")
        print(f"File {i+1}: {os.path.basename(csv_file)}")
        print(f"{'='*60}")
        
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue
        
        try:
            out_dir = os.path.join(os.path.dirname(csv_file), 'emg_noise_reports')
            noise_results, sampling_rate, restimulus_data = analyze_emg_noise_from_csv(
                csv_file,
                sample_duration=10,
                channels_to_analyze=None,  # analyze all channels
                output_dir=out_dir
            )
            
            if noise_results:
                # Global summary (optional)
                filter_recommendations = recommend_filters(noise_results, sampling_rate)
                # Per-channel
                channel_recs = per_channel_recommendations(noise_results, sampling_rate)

                # Write per-file CSV and JSON
                import json
                import csv
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                os.makedirs(out_dir, exist_ok=True)
                csv_path = os.path.join(out_dir, f'{base_name}.channel_recommendations.csv')
                with open(csv_path, 'w', newline='') as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(['channel', 'notch_freq', 'notch_Q', 'harmonics', 'bandpass_low', 'bandpass_high', 'filter_order'])
                    for ch, rec in sorted(channel_recs.items()):
                        writer.writerow([ch, rec['notch_freq'], rec['notch_Q'], ';'.join(map(str, rec['harmonics'])), rec['bandpass_low'], rec['bandpass_high'], rec['filter_order']])

                json_path = os.path.join(out_dir, f'{base_name}.filter_params.json')
                with open(json_path, 'w') as jf:
                    json.dump({str(ch): rec for ch, rec in channel_recs.items()}, jf, indent=2)
                all_results.append({
                    'file': csv_file,
                    'noise_results': noise_results,
                    'filter_recommendations': filter_recommendations,
                    'channel_recommendations': channel_recs,
                    'restimulus_data': restimulus_data
                })
        
        except Exception as e:
            print(f"Error analyzing {csv_file}: {e}")
            continue
    
    return all_results

if __name__ == "__main__":
    train_map_csv = r'D:\stage\v5.0\logs\map\train_map.csv'
    
    print("Raw EMG Data Analysis")
    print("=" * 60)
    
    # Analyze multiple files
    results = analyze_multiple_files(train_map_csv, sample_files=3)
    
    if results:
        print(f"\n" + "="*60)
        print("SUMMARY ACROSS ALL FILES")
        print("="*60)
        
        # Compare filter recommendations across files
        mains_freqs = [r['filter_recommendations']['mains_frequency'] for r in results]
        mains_types = [r['filter_recommendations']['mains_type'] for r in results]
        
        print(f"Detected mains frequencies: {mains_freqs}")
        print(f"Detected mains types: {mains_types}")
        
        # Use the most common recommendation
        most_common_mains = max(set(mains_freqs), key=mains_freqs.count)
        most_common_type = max(set(mains_types), key=mains_types.count)
        
        print(f"\nRecommended parameters for v5.0:")
        print(f"✓ Mains frequency: {most_common_mains:.1f}Hz ({most_common_type}Hz system)")
        print(f"✓ Notch filter: {most_common_mains:.1f}Hz, Q=30")
        print(f"✓ Bandpass filter: 20-450Hz (4th order Butterworth)")
        print(f"✓ Zero-phase filtering (filtfilt)")
        print(f"✓ Sampling rate: 2000 Hz")
        
        print(f"\nNext steps:")
        print(f"1. Use these parameters for EMG preprocessing")
        print(f"2. Implement the full preprocessing pipeline")
        print(f"3. Create new TFRecords with preprocessed EMG sequences + glove targets")


```

---

## File: v5.0\convert_cleaned_to_tfrecord.py
**Path:** `C:\stage\stage\v5.0\convert_cleaned_to_tfrecord.py`

```python
import os
import json
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Inputs
READY_CSV = r"D:\stage\v5.0\logs\map\ready.csv"
OUTPUT_DIR = r"F:\stage\v5.0\db\tfrecords_cleaned"

# Sequencing params
SEQUENCE_LENGTH = 500
TARGET_SHIFT = 300
EXAMPLES_PER_SHARD = 50000


def list_files(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    return df['filename'].tolist() if 'filename' in df.columns else []


def find_nmf_columns(cols: List[str]) -> List[str]:
    expected = [f'emg_nmf_{i}' for i in range(1, 7)]
    present = [c for c in expected if c in cols]
    if present:
        return present
    # fallback: any emg_nmf_*
    any_nmf = [c for c in cols if c.startswith('emg_nmf_')]
    try:
        any_nmf.sort(key=lambda x: int(x.split('_')[-1]))
    except Exception:
        any_nmf.sort()
    return any_nmf


def find_glove_columns(cols: List[str]) -> List[str]:
    prefer = [f'glove_{i}' for i in range(1, 23)]
    present = [c for c in prefer if c in cols]
    if present:
        return present
    any_glove = [c for c in cols if c.startswith('glove_')]
    try:
        any_glove.sort(key=lambda x: int(x.split('_')[1]))
    except Exception:
        any_glove.sort()
    return any_glove


def make_example(nmf_seq: np.ndarray, glove_vec: np.ndarray) -> tf.train.Example:
    feature = {
        'nmf_sequence': tf.train.Feature(float_list=tf.train.FloatList(value=nmf_seq.astype(np.float32).ravel())),
        'glove_target': tf.train.Feature(float_list=tf.train.FloatList(value=glove_vec.astype(np.float32).ravel())),
        'sequence_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[nmf_seq.shape[0], nmf_seq.shape[1]])),
        'glove_dim': tf.train.Feature(int64_list=tf.train.Int64List(value=[glove_vec.shape[0]])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def process_file(csv_path: str, writer: tf.io.TFRecordWriter, meta: dict) -> int:
    df = pd.read_csv(csv_path)
    nmf_cols = find_nmf_columns(df.columns.tolist())
    glove_cols = find_glove_columns(df.columns.tolist())
    if not nmf_cols or not glove_cols:
        return 0

    X = df[nmf_cols].values.astype(np.float32)
    Y = df[glove_cols].values.astype(np.float32)

    n = X.shape[0]
    written = 0
    last_start = n - SEQUENCE_LENGTH - TARGET_SHIFT
    if last_start <= 0:
        return 0

    for start in range(0, last_start):
        seq = X[start:start + SEQUENCE_LENGTH, :]
        target_idx = start + SEQUENCE_LENGTH + TARGET_SHIFT
        y = Y[target_idx, :]
        example = make_example(seq, y)
        writer.write(example.SerializeToString())
        written += 1

    # Store per-file meta
    meta[csv_path] = {
        'nmf_cols': nmf_cols,
        'glove_cols': glove_cols,
        'rows': int(n),
        'examples': int(written)
    }
    return written


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = list_files(READY_CSV)
    if not files:
        print('No files to process in ready.csv')
        return

    shard_idx = 0
    examples_in_shard = 0
    total_written = 0
    writer = None
    metadata = {}

    def open_new_writer(idx: int):
        path = os.path.join(OUTPUT_DIR, f'data_{idx:04d}.tfrecord')
        return tf.io.TFRecordWriter(path), path

    writer, current_path = open_new_writer(shard_idx)
    print(f'Writing shard: {current_path}')

    for fp in tqdm(files, desc='Converting to TFRecord'):
        written = process_file(fp, writer, metadata)
        total_written += written
        examples_in_shard += written
        if examples_in_shard >= EXAMPLES_PER_SHARD:
            writer.close()
            shard_idx += 1
            examples_in_shard = 0
            writer, current_path = open_new_writer(shard_idx)
            print(f'Writing shard: {current_path}')

    if writer is not None:
        writer.close()

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as jf:
        json.dump({
            'sequence_length': SEQUENCE_LENGTH,
            'target_shift': TARGET_SHIFT,
            'examples_per_shard': EXAMPLES_PER_SHARD,
            'total_examples': int(total_written),
            'files': metadata
        }, jf, indent=2)
    print(f'Total examples written: {total_written}')
    print(f'Metadata saved to: {meta_path}')


if __name__ == '__main__':
    main()
```

---

## File: v5.0\glove_cleaner.py
**Path:** `C:\stage\stage\v5.0\glove_cleaner.py`

```python
import os
import json
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

TRAIN_MAP_CSV = r"D:\stage\v5.0\logs\map\train_map.csv"
OUTPUT_ROOT = r"F:\data"

# Read full file by default. Set to an int to limit rows for performance testing.
SAMPLE_ROWS: Optional[int] = None

# Thresholds
OOB_COL_THRESHOLD = 0.05  # 5% per column
OOB_FILE_THRESHOLD = 0.10  # 10% per file


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def map_output_path(src_path: str) -> str:
    # Mirror original path under OUTPUT_ROOT by replacing drive root up to first 'DB'
    # e.g., E:\DB\DB2\E1\S20_E1_A1.csv -> F:\data\DB\DB2\E1\S20_E1_A1.csv
    parts = src_path.replace('\\', '/').split('/DB/', 1)
    rel = parts[1] if len(parts) > 1 else os.path.basename(src_path)
    return os.path.join(OUTPUT_ROOT, "DB", rel)


def find_glove_columns(df: pd.DataFrame) -> List[str]:
    preferred = [f"glove_{i}" for i in range(1, 23)]
    cols = [c for c in preferred if c in df.columns]
    if cols:
        return cols
    any_gloves = [c for c in df.columns if c.startswith('glove_')]
    try:
        any_gloves.sort(key=lambda x: int(x.split('_')[1]))
    except Exception:
        any_gloves.sort()
    return any_gloves


def analyze_and_validate(df: pd.DataFrame, glove_cols: List[str]) -> Dict[str, object]:
    per_col = {}
    total_rows = len(df)
    total_oob = 0
    for col in glove_cols:
        s = pd.to_numeric(df[col], errors='coerce')
        x = s.to_numpy()
        mask = (~np.isnan(x)) & (~np.isinf(x))
        xv = x[mask]
        if xv.size == 0:
            per_col[col] = {'p1': np.nan, 'p99': np.nan, 'oob_count': int(mask.sum()), 'oob_ratio': 1.0}
            total_oob += int(mask.sum())
            continue
        p1 = float(np.percentile(xv, 1))
        p99 = float(np.percentile(xv, 99))
        oob_mask = ((xv < p1) | (xv > p99))
        oob_count = int(np.sum(oob_mask))
        oob_ratio = float(oob_count / xv.size)
        per_col[col] = {'p1': p1, 'p99': p99, 'oob_count': oob_count, 'oob_ratio': oob_ratio, 'valid_count': int(xv.size)}
        total_oob += oob_count

    file_valid_count = int(np.sum([per_col[c]['valid_count'] for c in per_col if 'valid_count' in per_col[c]]))
    file_oob_ratio = float(total_oob / file_valid_count) if file_valid_count > 0 else 1.0

    any_col_exceeds = any((per_col[c]['oob_ratio'] is not None and per_col[c]['oob_ratio'] > OOB_COL_THRESHOLD) for c in per_col if 'oob_ratio' in per_col[c])
    file_exceeds = file_oob_ratio > OOB_FILE_THRESHOLD

    return {
        'per_col': per_col,
        'file_valid_count': file_valid_count,
        'file_oob_total': total_oob,
        'file_oob_ratio': file_oob_ratio,
        'any_col_exceeds': any_col_exceeds,
        'file_exceeds': file_exceeds
    }


def clamp_and_write(src_path: str, df: pd.DataFrame, glove_cols: List[str], per_col: Dict[str, dict]) -> str:
    # Clamp per column to [p1, p99]
    df_out = df.copy()
    for col in glove_cols:
        p1 = per_col[col]['p1']
        p99 = per_col[col]['p99']
        if not (np.isfinite(p1) and np.isfinite(p99)):
            continue
        s = pd.to_numeric(df_out[col], errors='coerce')
        s = s.clip(lower=p1, upper=p99)
        df_out[col] = s

    out_path = map_output_path(src_path)
    ensure_dir(os.path.dirname(out_path))
    df_out.to_csv(out_path, index=False)
    return out_path


def process_file(src_path: str) -> Dict[str, object]:
    print(f"Processing: {src_path}")
    try:
        df = pd.read_csv(src_path, nrows=SAMPLE_ROWS)
    except Exception as e:
        return {'file': src_path, 'status': 'error', 'error': str(e)}

    glove_cols = find_glove_columns(df)
    if not glove_cols:
        return {'file': src_path, 'status': 'skipped', 'reason': 'no_glove_columns'}

    analysis = analyze_and_validate(df, glove_cols)
    report = {
        'file': src_path,
        'status': 'validated',
        'file_oob_ratio': analysis['file_oob_ratio'],
        'file_oob_total': analysis['file_oob_total'],
        'per_col': analysis['per_col']
    }

    # Save analysis report next to destination path (under OUTPUT_ROOT)
    rep_dir = os.path.join(OUTPUT_ROOT, '_reports')
    ensure_dir(rep_dir)
    rep_path = os.path.join(rep_dir, os.path.basename(src_path) + '.glove_clean_report.json')
    with open(rep_path, 'w') as jf:
        json.dump(report, jf, indent=2)

    if analysis['any_col_exceeds'] or analysis['file_exceeds']:
        # Do not write cleaned file; require user decision
        print(f"  Threshold exceeded. Per-col>5% or file>10% OOB. See report: {rep_path}")
        return {
            'file': src_path,
            'status': 'threshold_exceeded',
            'report': rep_path,
            'file_oob_ratio': analysis['file_oob_ratio']
        }

    # Clamp and write cleaned copy
    out_path = clamp_and_write(src_path, df, glove_cols, analysis['per_col'])
    print(f"  Cleaned written: {out_path}")
    return {'file': src_path, 'status': 'cleaned', 'output': out_path, 'report': rep_path}


def main():
    df_map = pd.read_csv(TRAIN_MAP_CSV)
    results = []
    for _, row in tqdm(df_map.iterrows(), total=len(df_map)):
        src = row['filename']
        if not os.path.exists(src):
            results.append({'file': src, 'status': 'missing'})
            print(f"Missing: {src}")
            continue
        res = process_file(src)
        results.append(res)

    # Summary report
    summary_path = os.path.join(OUTPUT_ROOT, '_reports', 'glove_clean_summary.json')
    ensure_dir(os.path.dirname(summary_path))
    with open(summary_path, 'w') as jf:
        json.dump(results, jf, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()





```

---

## File: v5.0\inspect.py
**Path:** `C:\stage\stage\v5.0\inspect.py`

```python
import tensorflow as tf

def inspect_tfrecord(filename, num_examples=5):
    """Prints the structure and data types of features in a TFRecord file."""
    dataset = tf.data.TFRecordDataset(filename)
    for i, serialized_example in enumerate(dataset.take(num_examples)):
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        print(f"--- Example {i} ---")
        for feature_name, feature in example.features.feature.items():
            kind = feature.WhichOneof('kind')
            value = getattr(feature, kind).value
            # Show the data type and a sample of the value
            if kind == 'int64_list':
                print(f"  Feature '{feature_name}': int64_list with value(s) {list(value[:5])}")  # Show first 5 values
            elif kind == 'float_list':
                print(f"  Feature '{feature_name}': float_list with value(s) {list(value[:5])}")
            elif kind == 'bytes_list':
                # Be cautious with bytes, decode if it's text
                print(f"  Feature '{feature_name}': bytes_list with {len(value)} value(s)")
            else:
                print(f"  Feature '{feature_name}': unknown type")
inspect_tfrecord('F:\stage\v5.0\db\tfrecords_cleaned')
```

---

## File: v5.0\inspect_tfrecord_schema.py
**Path:** `C:\stage\stage\v5.0\inspect_tfrecord_schema.py`

```python
import tensorflow as tf
import os

def inspect_tfrecord_schema(tfrecord_path):
    """
    Inspect the actual schema of a TFRecord file
    """
    print(f"Inspecting TFRecord schema: {tfrecord_path}")
    
    try:
        # Read the first example to understand the schema
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        
        # Get the first example
        for raw_record in dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            print("\nActual TFRecord schema:")
            print("=" * 50)
            
            # Print all features
            for feature_name, feature in example.features.feature.items():
                print(f"Feature: {feature_name}")
                
                # Determine feature type and shape
                if feature.HasField('float_list'):
                    values = list(feature.float_list.value)
                    print(f"  Type: float_list")
                    print(f"  Length: {len(values)}")
                    print(f"  Sample values: {values[:5]}...")
                elif feature.HasField('int64_list'):
                    values = list(feature.int64_list.value)
                    print(f"  Type: int64_list")
                    print(f"  Length: {len(values)}")
                    print(f"  Sample values: {values[:5]}...")
                elif feature.HasField('bytes_list'):
                    values = list(feature.bytes_list.value)
                    print(f"  Type: bytes_list")
                    print(f"  Length: {len(values)}")
                    print(f"  Sample values: {values[:2]}...")
                else:
                    print(f"  Type: unknown")
                
                print()
            
            return example.features.feature.keys()
    
    except Exception as e:
        print(f"Error reading TFRecord: {e}")
        return None

def inspect_multiple_tfrecords(tfrecords_dir, num_files=5):
    """
    Inspect multiple TFRecord files to understand the schema
    """
    import glob
    
    tfrecord_files = glob.glob(os.path.join(tfrecords_dir, "data_*.tfrecord"))
    tfrecord_files.sort()
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    print(f"Inspecting first {min(num_files, len(tfrecord_files))} files")
    
    all_schemas = []
    
    for i, tfrecord_file in enumerate(tfrecord_files[:num_files]):
        print(f"\n{'='*60}")
        print(f"File {i+1}: {os.path.basename(tfrecord_file)}")
        print(f"{'='*60}")
        
        schema = inspect_tfrecord_schema(tfrecord_file)
        if schema:
            all_schemas.append(schema)
    
    # Compare schemas across files
    if all_schemas:
        print(f"\n{'='*60}")
        print("SCHEMA COMPARISON")
        print(f"{'='*60}")
        
        # Check if all files have the same schema
        first_schema = set(all_schemas[0])
        all_consistent = all(set(schema) == first_schema for schema in all_schemas)
        
        if all_consistent:
            print("✓ All files have consistent schema")
            print(f"Features: {sorted(first_schema)}")
        else:
            print("⚠ Files have different schemas")
            for i, schema in enumerate(all_schemas):
                print(f"File {i+1}: {sorted(schema)}")
    
    return all_schemas

if __name__ == "__main__":
    tfrecords_dir = r'D:\stage\v5.0\db\tfrecords'
    
    print("TFRecord Schema Inspection")
    print("=" * 60)
    
    # Inspect multiple files to understand the schema
    schemas = inspect_multiple_tfrecords(tfrecords_dir, num_files=3)
    
    if schemas:
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        print("Based on the schema inspection, update the parse function with the correct feature names.")
        print("The current analysis script assumes features named 'nmf_sequence' and 'glove_target',")
        print("but the actual TFRecord files may use different feature names.")


```

---

## File: v5.0\process_to_cleaned_csv.py
**Path:** `C:\stage\stage\v5.0\process_to_cleaned_csv.py`

```python
import os
import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
from sklearn.decomposition import MiniBatchNMF

MAP_CSV = r"D:\stage\v5.0\logs\map\cleaned_labels.csv"
OUTPUT_ROOT = r"F:\cleaned"

# EMG filter params
FS = 2000
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
BAND_LOW = 20.0
BAND_HIGH = 450.0
BUTTER_ORDER = 4

# NMF params
NMF_COMPONENTS = 6
NMF_MAX_SAMPLES = 200000  # cap total samples collected for NMF training across files

# Glove stats sampling
GLOVE_MAX_SAMPLES = 500000  # for robust p1/p99 estimation


def read_file_list(map_csv: str) -> List[str]:
    df = pd.read_csv(map_csv)
    return df['filename'].tolist() if 'filename' in df.columns else []


def list_emg_cols(df_columns: List[str]) -> List[str]:
    prefer = [f"emg_{i}" for i in range(1, 13)]
    cols = [c for c in prefer if c in df_columns]
    return cols


def list_glove_cols(df_columns: List[str]) -> List[str]:
    prefer = [f"glove_{i}" for i in range(1, 23)]
    cols = [c for c in prefer if c in df_columns]
    if not cols:
        cols = [c for c in df_columns if c.startswith('glove_')]
        try:
            cols.sort(key=lambda x: int(x.split('_')[1]))
        except Exception:
            cols.sort()
    return cols


def design_filters(fs: int):
    # Notch
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs)
    # Bandpass
    nyq = 0.5 * fs
    low = BAND_LOW / nyq
    high = BAND_HIGH / nyq
    b_band, a_band = signal.butter(BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)


def compute_emg_rest_stats(files: List[str]) -> Dict[str, np.ndarray]:
    emg_sum = None
    emg_sumsq = None
    count = 0
    for fp in tqdm(files, desc='EMG rest stats'):
        try:
            df = pd.read_csv(fp, usecols=lambda c: c.startswith('emg_') or c == 'restimulus')
        except Exception:
            continue
        emg_cols = list_emg_cols(df.columns.tolist())
        if not emg_cols or 'restimulus' not in df.columns:
            continue
        rest = df['restimulus'].values == 0
        if not np.any(rest):
            continue
        x = df[emg_cols].values[rest, :].astype(np.float64)
        if x.size == 0:
            continue
        if emg_sum is None:
            emg_sum = np.zeros(x.shape[1], dtype=np.float64)
            emg_sumsq = np.zeros(x.shape[1], dtype=np.float64)
        emg_sum += np.sum(x, axis=0)
        emg_sumsq += np.sum(x * x, axis=0)
        count += x.shape[0]
    if count == 0:
        raise RuntimeError('No rest samples found to compute EMG stats.')
    mean = emg_sum / count
    var = emg_sumsq / count - mean * mean
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var)
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}


def collect_nmf_samples(files: List[str], mean: np.ndarray, std: np.ndarray, max_samples: int) -> np.ndarray:
    samples = []
    remaining = max_samples
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    for fp in tqdm(files, desc='Collect NMF samples'):
        if remaining <= 0:
            break
        try:
            df = pd.read_csv(fp, usecols=lambda c: c.startswith('emg_') or c == 'restimulus')
        except Exception:
            continue
        emg_cols = list_emg_cols(df.columns.tolist())
        if not emg_cols or 'restimulus' not in df.columns:
            continue
        move = df['restimulus'].values != 0
        if not np.any(move):
            continue
        x = df[emg_cols].values.astype(np.float64)
        # Filter per channel
        for ch in range(x.shape[1]):
            sig = x[:, ch]
            sig = signal.filtfilt(b_notch, a_notch, sig)
            sig = signal.filtfilt(b_band, a_band, sig)
            x[:, ch] = sig
        # Z-score
        x = (x - mean) / std
        x = np.abs(x)
        xm = x[move, :]
        if xm.shape[0] == 0:
            continue
        take = min(remaining, xm.shape[0])
        # Uniformly sample without replacement
        idx = np.linspace(0, xm.shape[0] - 1, num=take, dtype=int)
        samples.append(xm[idx, :].astype(np.float32))
        remaining -= take
    if not samples:
        raise RuntimeError('No movement samples collected for NMF.')
    X = np.vstack(samples)
    return X


def train_nmf(features: np.ndarray, n_components: int) -> MiniBatchNMF:
    nmf = MiniBatchNMF(n_components=n_components, random_state=42, batch_size=512, max_iter=200)
    nmf.fit(features)
    return nmf


def compute_glove_stats_per_file(df: pd.DataFrame, glove_cols: List[str]) -> Dict[str, np.ndarray]:
    mins = []
    maxs = []
    for c in glove_cols:
        vals = pd.to_numeric(df[c], errors='coerce').to_numpy()
        vals = vals[~np.isnan(vals) & ~np.isinf(vals)]
        if vals.size == 0:
            mins.append(0.0)
            maxs.append(1.0)
        else:
            p1 = float(np.percentile(vals, 1))
            p99 = float(np.percentile(vals, 99))
            mins.append(p1)
            maxs.append(max(p99, p1 + 1e-6))
    return {'min': np.array(mins, dtype=np.float32), 'max': np.array(maxs, dtype=np.float32)}


def map_output_path(src_path: str) -> str:
    # Mirror under OUTPUT_ROOT by replacing up to '/DB/' anchor
    norm = src_path.replace('\\', '/')
    parts = norm.split('/DB/', 1)
    rel = parts[1] if len(parts) > 1 else os.path.basename(src_path)
    out_path = os.path.join(OUTPUT_ROOT, 'DB', rel)
    return out_path


def process_file(fp: str, emg_cols: List[str], glove_cols: List[str],
                 emg_mean: np.ndarray, emg_std: np.ndarray,
                 nmf: MiniBatchNMF) -> Optional[str]:
    try:
        usecols = emg_cols + ['restimulus'] + glove_cols
        df = pd.read_csv(fp, usecols=lambda c: c in usecols)
    except Exception:
        return None
    if not all(c in df.columns for c in emg_cols) or not all(c in df.columns for c in glove_cols) or 'restimulus' not in df.columns:
        return None
    x = df[emg_cols].values.astype(np.float64)
    (b_notch, a_notch), (b_band, a_band) = design_filters(FS)
    # Filter per channel
    for ch in range(x.shape[1]):
        sig = x[:, ch]
        sig = signal.filtfilt(b_notch, a_notch, sig)
        sig = signal.filtfilt(b_band, a_band, sig)
        x[:, ch] = sig
    # Z-score and rect
    x = (x - emg_mean) / emg_std
    x = np.abs(x).astype(np.float32)
    # NMF transform
    Xn = nmf.transform(x).astype(np.float32)
    nmf_cols = [f'emg_nmf_{i+1}' for i in range(Xn.shape[1])]

    # Normalize glove per-file using p1/p99 per column
    stats = compute_glove_stats_per_file(df, glove_cols)
    g = df[glove_cols].values.astype(np.float32)
    min_map = stats['min']
    range_map = np.maximum(stats['max'] - stats['min'], 1e-6)
    g_norm = np.clip((g - min_map) / range_map, 0.0, 1.0)

    out_df = pd.DataFrame(Xn, columns=nmf_cols)
    out_df['restimulus'] = df['restimulus'].values
    for i, c in enumerate(glove_cols):
        out_df[c] = g_norm[:, i]

    out_path = map_output_path(fp)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    # Save per-file glove params alongside
    params_path = os.path.join(out_dir, os.path.splitext(os.path.basename(fp))[0] + '.glove_params.json')
    with open(params_path, 'w') as jf:
        json.dump({'glove_cols': glove_cols,
                   'min': [float(x) for x in min_map],
                   'max': [float(x) for x in (stats['max'])]}, jf, indent=2)
    return out_path


def main():
    files = read_file_list(MAP_CSV)
    if not files:
        raise RuntimeError('No files found in mapping CSV.')

    # Determine columns from first valid file
    emg_cols = []
    glove_cols = []
    for fp in files:
        try:
            df0 = pd.read_csv(fp, nrows=5)
            emg_cols = list_emg_cols(df0.columns.tolist())
            glove_cols = list_glove_cols(df0.columns.tolist())
            if emg_cols and glove_cols and 'restimulus' in df0.columns:
                break
        except Exception:
            continue
    if not emg_cols or not glove_cols:
        raise RuntimeError('Could not detect EMG/Glove columns.')

    # Phase 1: EMG rest mean/std
    stats = compute_emg_rest_stats(files)
    np.save('emg_rest_mean.npy', stats['mean'])
    np.save('emg_rest_std.npy', stats['std'])

    # Phase 2: NMF training
    nmf_samples = collect_nmf_samples(files, stats['mean'], stats['std'], NMF_MAX_SAMPLES)
    nmf = train_nmf(nmf_samples, NMF_COMPONENTS)
    import joblib
    joblib.dump(nmf, 'nmf_model.pkl')

    # Phase 4: Per-file processing and save
    outputs = []
    for fp in tqdm(files, desc='Process & save'):
        out = process_file(fp, emg_cols, glove_cols, stats['mean'], stats['std'], nmf)
        if out:
            outputs.append(out)

    with open(os.path.join(OUTPUT_ROOT, 'processing_outputs.json'), 'w') as jf:
        json.dump({'outputs': outputs}, jf, indent=2)
    print(f'Done. Saved {len(outputs)} files to {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()



```

---

## File: v5.0\resume_train.py
**Path:** `C:\stage\stage\v5.0\resume_train.py`

```python
import os
import json
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Inputs
READY_CSV = r"D:\stage\v5.0\logs\map\ready.csv"
OUTPUT_DIR = r"F:\stage\v5.0\db\tfrecords_cleaned"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

# Sequencing params
SEQUENCE_LENGTH = 500
TARGET_SHIFT = 300
EXAMPLES_PER_SHARD = 50000

def load_checkpoint():
    """Load progress checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        'processed_files': [],
        'current_shard': 0,
        'examples_in_current_shard': 0,
        'total_written': 0,
        'metadata': {}
    }

def save_checkpoint(checkpoint):
    """Save progress checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def list_files(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    return df['filename'].tolist() if 'filename' in df.columns else []

def find_nmf_columns(cols: List[str]) -> List[str]:
    expected = [f'emg_nmf_{i}' for i in range(1, 7)]
    present = [c for c in expected if c in cols]
    if present:
        return present
    any_nmf = [c for c in cols if c.startswith('emg_nmf_')]
    try:
        any_nmf.sort(key=lambda x: int(x.split('_')[-1]))
    except Exception:
        any_nmf.sort()
    return any_nmf

def find_glove_columns(cols: List[str]) -> List[str]:
    prefer = [f'glove_{i}' for i in range(1, 23)]
    present = [c for c in prefer if c in cols]
    if present:
        return present
    any_glove = [c for c in cols if c.startswith('glove_')]
    try:
        any_glove.sort(key=lambda x: int(x.split('_')[1]))
    except Exception:
        any_glove.sort()
    return any_glove

def make_example(nmf_seq: np.ndarray, glove_vec: np.ndarray) -> tf.train.Example:
    feature = {
        'nmf_sequence': tf.train.Feature(float_list=tf.train.FloatList(value=nmf_seq.astype(np.float32).ravel())),
        'glove_target': tf.train.Feature(float_list=tf.train.FloatList(value=glove_vec.astype(np.float32).ravel())),
        'sequence_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[nmf_seq.shape[0], nmf_seq.shape[1]])),
        'glove_dim': tf.train.Feature(int64_list=tf.train.Int64List(value=[glove_vec.shape[0]])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def process_file(csv_path: str, writer: tf.io.TFRecordWriter, meta: dict) -> int:
    df = pd.read_csv(csv_path)
    nmf_cols = find_nmf_columns(df.columns.tolist())
    glove_cols = find_glove_columns(df.columns.tolist())
    if not nmf_cols or not glove_cols:
        return 0

    X = df[nmf_cols].values.astype(np.float32)
    Y = df[glove_cols].values.astype(np.float32)

    n = X.shape[0]
    written = 0
    last_start = n - SEQUENCE_LENGTH - TARGET_SHIFT
    if last_start <= 0:
        return 0

    for start in range(0, last_start):
        seq = X[start:start + SEQUENCE_LENGTH, :]
        target_idx = start + SEQUENCE_LENGTH + TARGET_SHIFT
        y = Y[target_idx, :]
        example = make_example(seq, y)
        writer.write(example.SerializeToString())
        written += 1

    # Store per-file meta
    meta[csv_path] = {
        'nmf_cols': nmf_cols,
        'glove_cols': glove_cols,
        'rows': int(n),
        'examples': int(written)
    }
    return written

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_files = set(checkpoint['processed_files'])
    
    files = list_files(READY_CSV)
    if not files:
        print('No files to process in ready.csv')
        return

    # Filter out already processed files
    files_to_process = [f for f in files if f not in processed_files]
    
    if not files_to_process:
        print('All files already processed!')
        return
        
    print(f'Resuming from checkpoint: {len(processed_files)} files already processed')
    print(f'Files remaining: {len(files_to_process)}')

    shard_idx = checkpoint['current_shard']
    examples_in_shard = checkpoint['examples_in_current_shard']
    total_written = checkpoint['total_written']
    metadata = checkpoint['metadata']
    
    writer = None

    def open_new_writer(idx: int):
        path = os.path.join(OUTPUT_DIR, f'data_{idx:04d}.tfrecord')
        return tf.io.TFRecordWriter(path), path

    # Open current writer if we have examples in current shard
    if examples_in_shard > 0:
        writer, current_path = open_new_writer(shard_idx)
        print(f'Resuming shard: {current_path} (contains {examples_in_shard} examples)')
    else:
        writer, current_path = open_new_writer(shard_idx)
        print(f'Starting new shard: {current_path}')

    try:
        for fp in tqdm(files_to_process, desc='Converting to TFRecord'):
            written = process_file(fp, writer, metadata)
            total_written += written
            examples_in_shard += written
            processed_files.add(fp)
            
            # Update checkpoint after each file
            checkpoint.update({
                'processed_files': list(processed_files),
                'current_shard': shard_idx,
                'examples_in_current_shard': examples_in_shard,
                'total_written': total_written,
                'metadata': metadata
            })
            save_checkpoint(checkpoint)
            
            if examples_in_shard >= EXAMPLES_PER_SHARD:
                writer.close()
                shard_idx += 1
                examples_in_shard = 0
                writer, current_path = open_new_writer(shard_idx)
                print(f'Starting new shard: {current_path}')
                
                # Update checkpoint for new shard
                checkpoint.update({
                    'current_shard': shard_idx,
                    'examples_in_current_shard': examples_in_shard
                })
                save_checkpoint(checkpoint)

        # Final cleanup
        if writer is not None:
            writer.close()

        # Save final metadata and remove checkpoint
        meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
        with open(meta_path, 'w') as jf:
            json.dump({
                'sequence_length': SEQUENCE_LENGTH,
                'target_shift': TARGET_SHIFT,
                'examples_per_shard': EXAMPLES_PER_SHARD,
                'total_examples': int(total_written),
                'files': metadata
            }, jf, indent=2)
        
        # Remove checkpoint file since we're done
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            
        print(f'Total examples written: {total_written}')
        print(f'Metadata saved to: {meta_path}')

    except Exception as e:
        print(f'Process interrupted. Checkpoint saved. Resume later to continue.')
        print(f'Error: {e}')
        if writer is not None:
            writer.close()
        raise

if __name__ == '__main__':
    main()
```

---

## File: v5.0\train_lstm.py
**Path:** `C:\stage\stage\v5.0\train_lstm.py`

```python
import os
from typing import Tuple, List
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# -----------------------
# Config
# -----------------------
TFRECORD_DIR = r"D:\stage\v5.0\db\tfrecords_cleaned"
MODEL_DIR = r"C:\models\emg_lstm"
SEQUENCE_LENGTH = 500
NMF_DIM = 6
OUTPUT_DIM = 22  # glove dimension (auto-checked at runtime)

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.2


def list_tfrecords(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.tfrecord')
    ]


def _parse_example(serialized_example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    feature_description = {
        'nmf_sequence': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'glove_target': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sequence_shape': tf.io.FixedLenFeature([2], tf.int64),
        'glove_dim': tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_description)

    seq_shape = tf.cast(parsed['sequence_shape'], tf.int32)
    glove_dim = tf.cast(parsed['glove_dim'][0], tf.int32)

    seq = parsed['nmf_sequence']
    seq = tf.reshape(seq, (seq_shape[0], seq_shape[1]))  # (time, features)

    y = parsed['glove_target']
    y = tf.reshape(y, (glove_dim,))

    return seq, y


def build_dataset(paths: List[str], batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units: int):
        super(TemporalAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (batch, time, features)
        # score: (batch, time, 1)
        score = self.V(tf.nn.tanh(self.W1(inputs) + self.W2(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, features)
        return context_vector


def build_model(sequence_length: int, nmf_dim: int, output_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length, nmf_dim), name='nmf_seq')

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
    )(inputs)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
    )(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    x = TemporalAttention(units=128)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='linear', name='glove')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=2000,
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-4,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    tfrecords = list_tfrecords(TFRECORD_DIR)
    if not tfrecords:
        print('No TFRecord files found.')
        return

    # Split train/val by shards (90/10 split)
    tfrecords.sort()
    split_idx = max(1, int(len(tfrecords) * 0.9))
    train_paths = tfrecords[:split_idx]
    val_paths = tfrecords[split_idx:]
    if not val_paths:
        val_paths = train_paths[-1:]

    # Build datasets
    train_ds = build_dataset(train_paths, BATCH_SIZE, shuffle=True)
    val_ds = build_dataset(val_paths, BATCH_SIZE, shuffle=False)

    # Infer output dimension from one element
    for _, y_sample in train_ds.take(1):
        out_dim = int(y_sample.shape[-1])
    else:
        out_dim = OUTPUT_DIM

    model = build_model(SEQUENCE_LENGTH, NMF_DIM, out_dim)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'bilstm_attn_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, 'training_log.csv')),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(os.path.join(MODEL_DIR, 'bilstm_attn_final'))
    print('Training complete. Model saved to:', MODEL_DIR)


if __name__ == '__main__':
    main()



```

---

## File: v5.0\train_lstm1.py
**Path:** `C:\stage\stage\v5.0\train_lstm1.py`

```python
import os
from typing import Tuple, List
import tensorflow as tf

# -----------------------
# Config
# -----------------------
TFRECORD_DIR = r"F:\stage\v5.0\db\tfrecords_cleaned"
MODEL_DIR = r"C:\models\2emg_lstm"
SEQUENCE_LENGTH = 500
NMF_DIM = 6
OUTPUT_DIM = 22

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.2


def list_tfrecords(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.tfrecord')
    ]


def _parse_example(serialized_example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    feature_description = {
        'nmf_sequence': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'glove_target': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sequence_shape': tf.io.FixedLenFeature([2], tf.int64),
        'glove_dim': tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_description)

    seq_shape = tf.cast(parsed['sequence_shape'], tf.int32)
    glove_dim = tf.cast(parsed['glove_dim'][0], tf.int32)

    seq = parsed['nmf_sequence']
    seq = tf.reshape(seq, (seq_shape[0], seq_shape[1]))  # (time, features)

    y = parsed['glove_target']
    y = tf.reshape(y, (glove_dim,))

    return seq, y


def build_dataset(paths: List[str], batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    # Remove shuffling entirely for max throughput
    ds = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    # 🔥 SHUFFLING REMOVED 🔥
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units: int):
        super(TemporalAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        score = self.V(tf.nn.tanh(self.W1(inputs) + self.W2(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


def build_model(sequence_length: int, nmf_dim: int, output_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length, nmf_dim), name='nmf_seq')

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
    )(inputs)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
    )(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    x = TemporalAttention(units=128)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='linear', name='glove')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=2000,
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-4,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    tfrecords = list_tfrecords(TFRECORD_DIR)
    if not tfrecords:
        print('No TFRecord files found.')
        return

    # Split train/val by shards (90/10 split)
    tfrecords.sort()
    split_idx = max(1, int(len(tfrecords) * 0.9))
    train_paths = tfrecords[:split_idx]
    val_paths = tfrecords[split_idx:]
    if not val_paths:
        val_paths = train_paths[-1:]

    # Build datasets — NO SHUFFLING
    train_ds = build_dataset(train_paths, BATCH_SIZE, shuffle=False)  # ← changed to False
    val_ds = build_dataset(val_paths, BATCH_SIZE, shuffle=False)

    # Infer output dimension
    for _, y_sample in train_ds.take(1):
        out_dim = int(y_sample.shape[-1])
        break
    else:
        out_dim = OUTPUT_DIM

    model = build_model(SEQUENCE_LENGTH, NMF_DIM, out_dim)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'bilstm_attn_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, 'training_log.csv')),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(os.path.join(MODEL_DIR, 'bilstm_attn_final'))
    print('Training complete. Model saved to:', MODEL_DIR)


if __name__ == '__main__':
    main()
```

---

## File: v6.0\checkkk.py
**Path:** `C:\stage\stage\v6.0\checkkk.py`

```python
# Check your TFRecord files
import tensorflow as tf

def inspect_tfrecord(filepath):
    try:
        dataset = tf.data.TFRecordDataset(filepath)
        count = 0
        for record in dataset:
            count += 1
        print(f"{filepath}: {count} records")
        return count
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

# Check all your files
for i in range(9):  # assuming 000 to 008
    filename = f"data_{i:04d}.tfrecord"
    inspect_tfrecord(filename)
```

---

## File: v6.0\convert_tf.py
**Path:** `C:\stage\stage\v6.0\convert_tf.py`

```python
import os
import json
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Inputs
READY_CSV = r"F:\stage\v6.0\logs\map\ready.csv"
OUTPUT_DIR = r"F:\stage\v6.0\db\tfrecords_cleaned"

# Sequencing params
SEQUENCE_LENGTH = 500
TARGET_SHIFT = 300
EXAMPLES_PER_SHARD = 50000


def list_files(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    return df['filename'].tolist() if 'filename' in df.columns else []


def find_nmf_columns(cols: List[str]) -> List[str]:
    expected = [f'emg_nmf_{i}' for i in range(1, 7)]
    present = [c for c in expected if c in cols]
    if present:
        return present
    # fallback: any emg_nmf_*
    any_nmf = [c for c in cols if c.startswith('emg_nmf_')]
    try:
        any_nmf.sort(key=lambda x: int(x.split('_')[-1]))
    except Exception:
        any_nmf.sort()
    return any_nmf


def find_glove_columns(cols: List[str]) -> List[str]:
    prefer = [f'glove_{i}' for i in range(1, 23)]
    present = [c for c in prefer if c in cols]
    if present:
        return present
    any_glove = [c for c in cols if c.startswith('glove_')]
    try:
        any_glove.sort(key=lambda x: int(x.split('_')[1]))
    except Exception:
        any_glove.sort()
    return any_glove


def make_example(nmf_seq: np.ndarray, glove_vec: np.ndarray) -> tf.train.Example:
    # Serialize entire arrays to bytes for efficient storage :cite[4]:cite[8]
    nmf_seq_bytes = tf.io.serialize_tensor(nmf_seq.astype(np.float32)).numpy()
    glove_vec_bytes = tf.io.serialize_tensor(glove_vec.astype(np.float32)).numpy()

    feature = {
        'nmf_sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[nmf_seq_bytes])),
        'glove_target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[glove_vec_bytes])),
        # Removed redundant shape info as it can be inferred during parsing
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def process_file(csv_path: str, writer: tf.io.TFRecordWriter, meta: dict) -> int:
    df = pd.read_csv(csv_path)
    nmf_cols = find_nmf_columns(df.columns.tolist())
    glove_cols = find_glove_columns(df.columns.tolist())
    if not nmf_cols or not glove_cols:
        return 0

    X = df[nmf_cols].values.astype(np.float32)
    Y = df[glove_cols].values.astype(np.float32)

    n = X.shape[0]
    written = 0
    last_start = n - SEQUENCE_LENGTH - TARGET_SHIFT
    if last_start <= 0:
        return 0

    for start in range(0, last_start):
        seq = X[start:start + SEQUENCE_LENGTH, :]
        target_idx = start + SEQUENCE_LENGTH + TARGET_SHIFT
        y = Y[target_idx, :]
        example = make_example(seq, y)
        writer.write(example.SerializeToString())
        written += 1

    # Store per-file meta
    meta[csv_path] = {
        'nmf_cols': nmf_cols,
        'glove_cols': glove_cols,
        'rows': int(n),
        'examples': int(written)
    }
    return written


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = list_files(READY_CSV)
    if not files:
        print('No files to process in ready.csv')
        return

    shard_idx = 0
    examples_in_shard = 0
    total_written = 0
    writer = None
    metadata = {}

    def open_new_writer(idx: int):
        path = os.path.join(OUTPUT_DIR, f'data_{idx:04d}.tfrecord')
        # Add GZIP compression to reduce file size :cite[4]
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        return tf.io.TFRecordWriter(path, options=options), path

    writer, current_path = open_new_writer(shard_idx)
    print(f'Writing shard: {current_path}')

    for fp in tqdm(files, desc='Converting to TFRecord'):
        written = process_file(fp, writer, metadata)
        total_written += written
        examples_in_shard += written
        if examples_in_shard >= EXAMPLES_PER_SHARD:
            writer.close()
            shard_idx += 1
            examples_in_shard = 0
            writer, current_path = open_new_writer(shard_idx)
            print(f'Writing shard: {current_path}')

    if writer is not None:
        writer.close()

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as jf:
        json.dump({
            'sequence_length': SEQUENCE_LENGTH,
            'target_shift': TARGET_SHIFT,
            'examples_per_shard': EXAMPLES_PER_SHARD,
            'total_examples': int(total_written),
            'files': metadata
        }, jf, indent=2)
    print(f'Total examples written: {total_written}')
    print(f'Metadata saved to: {meta_path}')


if __name__ == '__main__':
    main()
```

---

## File: v6.0\convert_tf1.py
**Path:** `C:\stage\stage\v6.0\convert_tf1.py`

```python
import os
import json
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Inputs
READY_CSV = r"F:\stage\v6.0\logs\map\ready.csv"
OUTPUT_DIR = r"F:\stage\v6.0\db\tfrecords_cleaned_fixed"

# Sequencing params
SEQUENCE_LENGTH = 500
TARGET_SHIFT = 300
EXAMPLES_PER_SHARD = 50000


def list_files(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    return df['filename'].tolist() if 'filename' in df.columns else []


def find_nmf_columns(cols: List[str]) -> List[str]:
    expected = [f'emg_nmf_{i}' for i in range(1, 7)]
    present = [c for c in expected if c in cols]
    if present:
        return present
    any_nmf = [c for c in cols if c.startswith('emg_nmf_')]
    try:
        any_nmf.sort(key=lambda x: int(x.split('_')[-1]))
    except Exception:
        any_nmf.sort()
    return any_nmf


def find_glove_columns(cols: List[str]) -> List[str]:
    prefer = [f'glove_{i}' for i in range(1, 23)]
    present = [c for c in prefer if c in cols]
    if present:
        return present
    any_glove = [c for c in cols if c.startswith('glove_')]
    try:
        any_glove.sort(key=lambda x: int(x.split('_')[1]))
    except Exception:
        any_glove.sort()
    return any_glove


def make_example(nmf_seq: np.ndarray, glove_vec: np.ndarray) -> tf.train.Example:
    """FIXED VERSION: Use simple byte serialization without FeatureList complexity"""
    # Serialize the tensors to bytes - this is the most reliable method
    nmf_seq_bytes = tf.io.serialize_tensor(nmf_seq.astype(np.float32)).numpy()
    glove_vec_bytes = tf.io.serialize_tensor(glove_vec.astype(np.float32)).numpy()

    # Create features using the simple byte list approach
    feature_dict = {
        'nmf_sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[nmf_seq_bytes])),
        'glove_target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[glove_vec_bytes])),
        # Add shape information for validation
        'nmf_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=nmf_seq.shape)),
        'glove_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=glove_vec.shape)),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def process_file(csv_path: str, writer: tf.io.TFRecordWriter, meta: dict) -> int:
    df = pd.read_csv(csv_path)
    nmf_cols = find_nmf_columns(df.columns.tolist())
    glove_cols = find_glove_columns(df.columns.tolist())
    if not nmf_cols or not glove_cols:
        print(f"Warning: Missing columns in {csv_path}")
        return 0

    X = df[nmf_cols].values.astype(np.float32)
    Y = df[glove_cols].values.astype(np.float32)

    n = X.shape[0]
    written = 0
    
    # FIXED: Correct bounds calculation
    last_start = n - SEQUENCE_LENGTH - TARGET_SHIFT - 1
    if last_start <= 0:
        print(f"Warning: File {csv_path} too short. Has {n} rows, needs at least {SEQUENCE_LENGTH + TARGET_SHIFT + 1}")
        return 0

    for start in range(0, last_start):
        seq = X[start:start + SEQUENCE_LENGTH, :]
        target_idx = start + SEQUENCE_LENGTH + TARGET_SHIFT
        
        # Additional bounds safety check
        if target_idx >= n:
            break
            
        y = Y[target_idx, :]
        example = make_example(seq, y)
        writer.write(example.SerializeToString())
        written += 1

    # Store per-file meta
    meta[csv_path] = {
        'nmf_cols': nmf_cols,
        'glove_cols': glove_cols,
        'rows': int(n),
        'examples': int(written)
    }
    
    print(f"Processed {csv_path}: {written} examples")
    return written


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = list_files(READY_CSV)
    if not files:
        print('No files to process in ready.csv')
        return

    shard_idx = 0
    examples_in_shard = 0
    total_written = 0
    writer = None
    metadata = {}

    files = files[:1]  # Just process first file for testing
    print("TEST MODE: Processing only 1 file")

    def open_new_writer(idx: int):
        path = os.path.join(OUTPUT_DIR, f'data_{idx:04d}.tfrecord')
        # FIXED: No GZIP compression to avoid corruption
        return tf.io.TFRecordWriter(path), path

    writer, current_path = open_new_writer(shard_idx)
    print(f'Writing shard: {current_path}')

    for fp in tqdm(files, desc='Converting to TFRecord'):
        written = process_file(fp, writer, metadata)
        total_written += written
        examples_in_shard += written
        
        # Only create new writer if we have examples and reached threshold
        if examples_in_shard >= EXAMPLES_PER_SHARD and written > 0:
            writer.close()
            shard_idx += 1
            examples_in_shard = 0
            writer, current_path = open_new_writer(shard_idx)
            print(f'Writing shard: {current_path}')

    # Close writer if it exists
    if writer is not None:
        writer.close()
        
        # Remove empty final file if it exists
        final_path = os.path.join(OUTPUT_DIR, f'data_{shard_idx:04d}.tfrecord')
        if os.path.exists(final_path) and os.path.getsize(final_path) == 0:
            os.remove(final_path)
            print(f"Removed empty file: {final_path}")

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as jf:
        json.dump({
            'sequence_length': SEQUENCE_LENGTH,
            'target_shift': TARGET_SHIFT,
            'examples_per_shard': EXAMPLES_PER_SHARD,
            'total_examples': int(total_written),
            'files': metadata
        }, jf, indent=2)
    print(f'Total examples written: {total_written}')
    print(f'Metadata saved to: {meta_path}')


if __name__ == '__main__':
    main()
    
```

---

## File: v6.0\train.py
**Path:** `C:\stage\stage\v6.0\train.py`

```python
import os
from typing import Tuple, List
import tensorflow as tf

# -----------------------
# Config
# -----------------------
TFRECORD_DIR = r"F:\stage\v6.0\db\tfrecords_cleaned"
MODEL_DIR = r"F:\stage\models\2emg_lstm"
SEQUENCE_LENGTH = 500
NMF_DIM = 6
OUTPUT_DIM = 22

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.2


def list_tfrecords(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.tfrecord')
    ]


def _parse_example(serialized_example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parse serialized TFRecord examples with the new binary format"""
    feature_description = {
        'nmf_sequence': tf.io.FixedLenFeature([], tf.string),  # Changed to string for serialized tensor
        'glove_target': tf.io.FixedLenFeature([], tf.string),  # Changed to string for serialized tensor
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    
    # Deserialize the tensors from binary format
    nmf_sequence = tf.io.parse_tensor(parsed['nmf_sequence'], out_type=tf.float32)
    glove_target = tf.io.parse_tensor(parsed['glove_target'], out_type=tf.float32)
    
    # Reshape to expected dimensions
    nmf_sequence = tf.reshape(nmf_sequence, (SEQUENCE_LENGTH, NMF_DIM))  # (time, features)
    glove_target = tf.reshape(glove_target, (OUTPUT_DIM,))  # Single target vector
    
    return nmf_sequence, glove_target


def build_dataset(paths: List[str], batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    # Important: Add compression_type='GZIP' to match how files were written
    ds = tf.data.TFRecordDataset(paths, compression_type='GZIP', num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)  # Consider adding shuffle back for better training
    
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units: int):
        super(TemporalAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        score = self.V(tf.nn.tanh(self.W1(inputs) + self.W2(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


def build_model(sequence_length: int, nmf_dim: int, output_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length, nmf_dim), name='nmf_seq')

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
    )(inputs)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
    )(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    x = TemporalAttention(units=128)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='linear', name='glove')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=2000,
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-4,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    tfrecords = list_tfrecords(TFRECORD_DIR)
    if not tfrecords:
        print('No TFRecord files found.')
        return

    # Split train/val by shards (90/10 split)
    tfrecords.sort()
    split_idx = max(1, int(len(tfrecords) * 0.9))
    train_paths = tfrecords[:split_idx]
    val_paths = tfrecords[split_idx:]
    if not val_paths:
        val_paths = train_paths[-1:]

    # Build datasets
    train_ds = build_dataset(train_paths, BATCH_SIZE, shuffle=False)
    val_ds = build_dataset(val_paths, BATCH_SIZE, shuffle=False)

    # Infer output dimension (this should now work correctly)
    for _, y_sample in train_ds.take(1):
        out_dim = int(y_sample.shape[-1])
        break
    else:
        out_dim = OUTPUT_DIM

    model = build_model(SEQUENCE_LENGTH, NMF_DIM, out_dim)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'bilstm_attn_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, 'training_log.csv')),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(os.path.join(MODEL_DIR, 'bilstm_attn_final'))
    print('Training complete. Model saved to:', MODEL_DIR)


if __name__ == '__main__':
    main()
```

---

## File: v6.0\verif.py
**Path:** `C:\stage\stage\v6.0\verif.py`

```python
import tensorflow as tf
import os

def check_tfrecord(file_path):
    print(f"Checking: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        count = 0
        for record in dataset.take(3):  # Check first 3 examples
            count += 1
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            print(f"Example {count}: OK")
        
        print(f"✅ SUCCESS: File contains {count} valid examples")
        return True
    except Exception as e:
        print(f"❌ CORRUPTED: {e}")
        return False

check_tfrecord(r"F:\stage\v6.0\db\tfrecords_cleaned_fixed\data_0000.tfrecord")
```

---

## File: v6.0\viz.py
**Path:** `C:\stage\stage\v6.0\viz.py`

```python
import tensorflow as tf
import numpy as np

def inspect_tfrecord_content(file_path, num_examples=2):
    """Inspect the actual data content of TFRecord examples"""
    print(f"🔍 Inspecting content of: {file_path}")
    
    # Define the feature description based on how you wrote the data
    feature_description = {
        'nmf_sequence': tf.io.FixedLenFeature([], tf.string),
        'glove_target': tf.io.FixedLenFeature([], tf.string),
        'nmf_shape': tf.io.FixedLenFeature([2], tf.int64),
        'glove_shape': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    dataset = tf.data.TFRecordDataset(file_path)
    
    for i, serialized_example in enumerate(dataset.take(num_examples)):
        print(f"\n📊 Example {i+1}:")
        
        # Parse the example
        parsed = tf.io.parse_single_example(serialized_example, feature_description)
        
        # Parse the tensors from bytes
        nmf_sequence = tf.io.parse_tensor(parsed['nmf_sequence'], out_type=tf.float32)
        glove_target = tf.io.parse_tensor(parsed['glove_target'], out_type=tf.float32)
        
        # Reshape to expected dimensions
        nmf_sequence = tf.reshape(nmf_sequence, (500, 6))  # SEQUENCE_LENGTH × NMF_DIM
        glove_target = tf.reshape(glove_target, (22,))     # OUTPUT_DIM
        
        print(f"   NMF Sequence shape: {nmf_sequence.shape}")
        print(f"   Glove Target shape: {glove_target.shape}")
        
        # Print first few values
        print(f"   First 3 NMF values (first timestep): {nmf_sequence[0, :3].numpy()}")
        print(f"   First 3 Glove values: {glove_target[:3].numpy()}")
        
        # Print statistics
        print(f"   NMF Stats - Min: {tf.reduce_min(nmf_sequence):.4f}, "
              f"Max: {tf.reduce_max(nmf_sequence):.4f}, "
              f"Mean: {tf.reduce_mean(nmf_sequence):.4f}")
        print(f"   Glove Stats - Min: {tf.reduce_min(glove_target):.4f}, "
              f"Max: {tf.reduce_max(glove_target):.4f}, "
              f"Mean: {tf.reduce_mean(glove_target):.4f}")

# Run the inspection
inspect_tfrecord_content(r"F:\stage\v6.0\db\tfrecords_cleaned_fixed\data_0000.tfrecord")
```

---

## File: v6.0\logs\map\ana.py
**Path:** `C:\stage\stage\v6.0\logs\map\ana.py`

```python
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import time

def analyze_csv_files(file_list_path):
    """
    Comprehensive analysis of CSV files before TFRecord conversion.
    Returns everything we need to know for a successful conversion.
    """
    
    # Read the file list
    file_df = pd.read_csv(file_list_path)
    
    analysis_results = {
        'file_info': [],
        'column_analysis': defaultdict(list),
        'data_types': defaultdict(set),
        'missing_values': defaultdict(int),
        'global_columns': set(),
        'total_rows': 0,
        'schema_consistency': True,
        'warnings': []
    }
    
    print("🔍 CSVs DIAGNOSTIC TOOL - TFRecord Preparation Analysis")
    print("=" * 70)
    
    for idx, row in file_df.iterrows():
        file_path = row['filename']
        start_line = row['start_line']
        end_line = row['end_line']
        expected_rows = row['rows_extracted']
        
        print(f"\n📁 Analyzing: {os.path.basename(file_path)}")
        print(f"   Expected rows: {expected_rows:,}")
        
        try:
            # Read the file with optimized memory usage
            start_time = time.time()
            
            # First, let's check the header and first few rows
            sample_df = pd.read_csv(file_path, nrows=5)
            actual_columns = sample_df.columns.tolist()
            
            # Now read the full segment (adjust chunksize based on memory)
            full_df = pd.read_csv(file_path)
            
            read_time = time.time() - start_time
            
            # Basic file information
            actual_rows = len(full_df)
            file_info = {
                'filename': file_path,
                'expected_rows': expected_rows,
                'actual_rows': actual_rows,
                'columns': actual_columns,
                'memory_usage_mb': full_df.memory_usage(deep=True).sum() / 1024**2,
                'read_time_seconds': read_time
            }
            analysis_results['file_info'].append(file_info)
            
            # Update global columns
            analysis_results['global_columns'].update(actual_columns)
            
            # Column-level analysis
            for column in actual_columns:
                col_data = full_df[column]
                
                # Data type analysis
                dtype = str(col_data.dtype)
                analysis_results['data_types'][column].add(dtype)
                
                # Missing values
                missing_count = col_data.isnull().sum()
                analysis_results['missing_values'][column] += missing_count
                
                # Statistical analysis (for numeric columns)
                if np.issubdtype(col_data.dtype, np.number):
                    analysis_results['column_analysis'][column].append({
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'file': os.path.basename(file_path)
                    })
                else:
                    # For categorical/string columns
                    unique_count = col_data.nunique()
                    sample_values = col_data.dropna().head(3).tolist()
                    analysis_results['column_analysis'][column].append({
                        'unique_count': unique_count,
                        'sample_values': sample_values,
                        'file': os.path.basename(file_path)
                    })
            
            # Verify row count matches expectation
            if actual_rows != expected_rows:
                analysis_results['warnings'].append(
                    f"⚠️ Row count mismatch in {os.path.basename(file_path)}: "
                    f"expected {expected_rows:,}, got {actual_rows:,}"
                )
            
            print(f"   ✅ Actual rows: {actual_rows:,}")
            print(f"   📊 Columns: {len(actual_columns)}")
            print(f"   💾 Memory: {file_info['memory_usage_mb']:.2f} MB")
            print(f"   ⏱️ Read time: {read_time:.2f}s")
            
            # Clean up to save memory
            del full_df
            
        except Exception as e:
            error_msg = f"❌ Error reading {file_path}: {str(e)}"
            print(error_msg)
            analysis_results['warnings'].append(error_msg)
            continue
    
    return analysis_results

def generate_conversion_report(analysis_results):
    """Generate a comprehensive report for TFRecord conversion."""
    
    print("\n" + "=" * 70)
    print("📊 TFRecord Conversion Readiness Report")
    print("=" * 70)
    
    # 1. Overall Summary
    total_expected = sum(info['expected_rows'] for info in analysis_results['file_info'])
    total_actual = sum(info['actual_rows'] for info in analysis_results['file_info'])
    
    print(f"\n📈 OVERALL SUMMARY:")
    print(f"   Total files analyzed: {len(analysis_results['file_info'])}")
    print(f"   Total expected rows: {total_expected:,}")
    print(f"   Total actual rows: {total_actual:,}")
    print(f"   Unique columns across all files: {len(analysis_results['global_columns'])}")
    
    # 2. Schema Consistency Check
    print(f"\n🔍 SCHEMA CONSISTENCY ANALYSIS:")
    
    # Check if all files have same columns
    column_sets = [set(info['columns']) for info in analysis_results['file_info']]
    common_columns = set.intersection(*column_sets) if column_sets else set()
    
    if len(common_columns) == len(analysis_results['global_columns']):
        print("   ✅ PERFECT: All files have identical column structure")
    else:
        print("   ⚠️ WARNING: Column structure varies between files")
        print(f"   Common columns: {len(common_columns)}")
        print(f"   All unique columns: {len(analysis_results['global_columns'])}")
        print(f"   Extra columns: {analysis_results['global_columns'] - common_columns}")
    
    # 3. Data Type Analysis
    print(f"\n📊 DATA TYPE ANALYSIS:")
    for column, dtypes in analysis_results['data_types'].items():
        if len(dtypes) > 1:
            print(f"   ⚠️ Column '{column}' has inconsistent types: {dtypes}")
        else:
            dtype_str = next(iter(dtypes))
            print(f"   ✅ Column '{column}': {dtype_str}")
    
    # 4. Missing Values Report
    print(f"\n❌ MISSING VALUES ANALYSIS:")
    if analysis_results['missing_values']:
        for column, missing_count in analysis_results['missing_values'].items():
            if missing_count > 0:
                percentage = (missing_count / total_actual) * 100
                print(f"   Column '{column}': {missing_count:,} missing ({percentage:.2f}%)")
    else:
        print("   ✅ No missing values detected")
    
    # 5. Column-specific Recommendations for TFRecord
    print(f"\n🎯 TFRecord CONVERSION RECOMMENDATIONS:")
    
    tfrecord_schema = {}
    for column in analysis_results['global_columns']:
        dtypes = analysis_results['data_types'][column]
        main_dtype = next(iter(dtypes)) if dtypes else 'unknown'
        
        if 'int' in main_dtype:
            tfrecord_type = 'tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))'
            print(f"   '{column}': INT64 (use Int64List)")
            tfrecord_schema[column] = 'int64'
        elif 'float' in main_dtype:
            tfrecord_type = 'tf.train.Feature(float_list=tf.train.FloatList(value=[value]))'
            print(f"   '{column}': FLOAT32 (use FloatList)")
            tfrecord_schema[column] = 'float32'
        else:
            # Object/string columns - FIXED THE SYNTAX ERROR HERE
            tfrecord_type = 'tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))'
            print(f"   '{column}': STRING (use BytesList - will encode to bytes)")
            tfrecord_schema[column] = 'bytes'
    
    # 6. Memory and Performance Considerations
    print(f"\n💾 PERFORMANCE CONSIDERATIONS:")
    total_memory = sum(info['memory_usage_mb'] for info in analysis_results['file_info'])
    print(f"   Estimated total memory: {total_memory:.2f} MB")
    print(f"   Recommended shard size: {max(10000, total_actual // 10):,} examples per shard")
    
    # 7. Warnings
    if analysis_results['warnings']:
        print(f"\n🚨 CRITICAL WARNINGS:")
        for warning in analysis_results['warnings']:
            print(f"   {warning}")
    
    return tfrecord_schema

# Create a sample file list CSV first
def create_sample_file_list():
    """Create the file list CSV based on your provided data"""
    file_data = [
        "filename,start_line,end_line,rows_extracted",
        "F:\\cleaned\\DB\\DB2\\E2\\S18_E2_A1.csv,20377562,22919274,2541713",
        "F:\\cleaned\\DB\\DB2\\E2\\S1_E2_A1.csv,25469199,28022487,2553289",
        "F:\\cleaned\\DB\\DB2\\E1\\S20_E1_A1.csv,120913700,122715088,1801389",
        "F:\\cleaned\\DB\\DB2\\E1\\S21_E1_A1.csv,122715089,124528719,1813631",
        "F:\\cleaned\\DB\\DB3\\E1\\S6_E1_A1.csv,185251980,187057003,1805024",
        "F:\\cleaned\\DB\\DB3\\E2\\S7_E2_A1.csv,211295167,213775807,2480641",
        "F:\\cleaned\\DB\\DB3\\E2\\S1_E2_A1.csv,197326945,198608588,1281644",
        "F:\\cleaned\\DB\\DB3\\E1\\S5_E1_A1.csv,183447196,185251979,1804784"
    ]
    
    with open('file_list.csv', 'w') as f:
        for line in file_data:
            f.write(line + '\n')
    print("✅ Created file_list.csv with your provided data")

# Usage example:
if __name__ == "__main__":
    # First create the file list
    create_sample_file_list()
    
    print("Starting comprehensive CSV analysis...")
    analysis = analyze_csv_files("ready.csv")
    tfrecord_schema = generate_conversion_report(analysis)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"Your TFRecord schema will be: {tfrecord_schema}")
    
    # Save the schema for later use
    import json
    with open('tfrecord_schema.json', 'w') as f:
        json.dump(tfrecord_schema, f, indent=2)
    print("✅ Saved TFRecord schema to 'tfrecord_schema.json'")

```

---

## File: v6.0\logs\map\baseline_model.py
**Path:** `C:\stage\stage\v6.0\logs\map\baseline_model.py`

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def create_baseline_model():
    """Simple baseline model for EMG -> Hand kinematics"""
    inputs = tf.keras.Input(shape=(6,), name='emg_input')
    
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(22, activation='linear', name='glove_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# USAGE:
# model = create_baseline_model()
# model.summary()
# history = model.fit(dataset, epochs=50, validation_split=0.2)

```

---

## File: v6.0\logs\map\baseline_reader.py
**Path:** `C:\stage\stage\v6.0\logs\map\baseline_reader.py`

```python
import tensorflow as tf
import numpy as np

def parse_baseline_tfrecord(example_proto):
    """Parse baseline TFRecord examples."""
    feature_description = {
        # Input features (6 EMG NMF components)
        'emg_nmf_1': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_2': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_3': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_4': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_5': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_6': tf.io.FixedLenFeature([], tf.float32),
        
        # Output labels (22 Glove dimensions)
        'glove_1': tf.io.FixedLenFeature([], tf.float32),
        'glove_2': tf.io.FixedLenFeature([], tf.float32),
        'glove_3': tf.io.FixedLenFeature([], tf.float32),
        'glove_4': tf.io.FixedLenFeature([], tf.float32),
        'glove_5': tf.io.FixedLenFeature([], tf.float32),
        'glove_6': tf.io.FixedLenFeature([], tf.float32),
        'glove_7': tf.io.FixedLenFeature([], tf.float32),
        'glove_8': tf.io.FixedLenFeature([], tf.float32),
        'glove_9': tf.io.FixedLenFeature([], tf.float32),
        'glove_10': tf.io.FixedLenFeature([], tf.float32),
        'glove_11': tf.io.FixedLenFeature([], tf.float32),
        'glove_12': tf.io.FixedLenFeature([], tf.float32),
        'glove_13': tf.io.FixedLenFeature([], tf.float32),
        'glove_14': tf.io.FixedLenFeature([], tf.float32),
        'glove_15': tf.io.FixedLenFeature([], tf.float32),
        'glove_16': tf.io.FixedLenFeature([], tf.float32),
        'glove_17': tf.io.FixedLenFeature([], tf.float32),
        'glove_18': tf.io.FixedLenFeature([], tf.float32),
        'glove_19': tf.io.FixedLenFeature([], tf.float32),
        'glove_20': tf.io.FixedLenFeature([], tf.float32),
        'glove_21': tf.io.FixedLenFeature([], tf.float32),
        'glove_22': tf.io.FixedLenFeature([], tf.float32),
        
        # Metadata
        'timestamp': tf.io.FixedLenFeature([], tf.int64),
        'file_id': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Create input tensor (6 EMG features)
    inputs = tf.stack([
        parsed['emg_nmf_1'], parsed['emg_nmf_2'], parsed['emg_nmf_3'],
        parsed['emg_nmf_4'], parsed['emg_nmf_5'], parsed['emg_nmf_6']
    ], axis=0)
    
    # Create output tensor (22 Glove positions)
    outputs = tf.stack([parsed[f'glove_{i}'] for i in range(1, 23)], axis=0)
    
    return inputs, outputs

def create_baseline_dataset(tfrecord_pattern, batch_size=32, shuffle_buffer=10000, repeat=True):
    """Create baseline TensorFlow dataset."""
    files = tf.data.Dataset.list_files(tfrecord_pattern)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=4  # Process 4 files in parallel
    )
    dataset = dataset.map(parse_baseline_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size)
    
    # CRITICAL FIX: Repeat dataset indefinitely for training
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# USAGE:
# dataset = create_baseline_dataset('tfrecords_baseline/*/*.tfrecord')
# for inputs, outputs in dataset.take(1):
#     print(f"Input shape: {inputs.shape}")  # (batch_size, 6)
#     print(f"Output shape: {outputs.shape}") # (batch_size, 22)
```

---

## File: v6.0\logs\map\change.py
**Path:** `C:\stage\stage\v6.0\logs\map\change.py`

```python
import csv

# Input and output file names
input_file = 'cleaned_labels.csv'   # Replace with your actual input filename
output_file = 'ready.csv' # Replace with your desired output filename

# Open the input CSV and output CSV
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read and write the header
    header = next(reader)
    writer.writerow(header)

    # Process each row
    for row in reader:
        old_path = row[0]
        # Replace E:\DB with F:\data\DB
        new_path = old_path.replace(r'F:\data\DB', r'F:\cleaned\DB', 1)
        row[0] = new_path
        writer.writerow(row)

print(f"Path conversion complete. Output saved to {output_file}")
```

---

## File: v6.0\logs\map\convert_baseline.py
**Path:** `C:\stage\stage\v6.0\logs\map\convert_baseline.py`

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import time

class BaselineCSVToTFRecordConverter:
    def __init__(self, file_list_path):
        """Initialize with your file list - BASELINE VERSION"""
        self.file_df = pd.read_csv(file_list_path)
        self.input_columns = [f'emg_nmf_{i}' for i in range(1, 7)]
        self.output_columns = [f'glove_{i}' for i in range(1, 23)]
        
        print("🎯 BASELINE CONVERSION - No Feature Engineering")
        print(f"   Inputs: {len(self.input_columns)} EMG NMF components")
        print(f"   Outputs: {len(self.output_columns)} Glove dimensions")
        print("   Status: FAST PATH TO WORKING BASELINE")

    def csv_row_to_tfexample(self, row, timestamp, file_id):
        """Convert CSV row to simple tf.train.Example - BASELINE"""
        
        # Input features (EMG NMF components)
        emg_features = {}
        for col in self.input_columns:
            emg_features[col] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(row[col])]))
        
        # Output labels (Glove positions)
        glove_features = {}
        for col in self.output_columns:
            glove_features[col] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(row[col])]))
        
        # Metadata for sequential processing later
        metadata = {
            'timestamp': tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp])),
            'file_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_id.encode()])),
        }
        
        # Combine all features
        feature_dict = {**emg_features, **glove_features, **metadata}
        
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def convert_single_file(self, csv_file_path, output_dir, examples_per_shard=100000):
        """Convert single CSV file - OPTIMIZED FOR SPEED"""
        
        file_base_name = os.path.basename(csv_file_path).replace('.csv', '')
        file_output_dir = os.path.join(output_dir, file_base_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        print(f"📤 Converting: {file_base_name}")
        start_time = time.time()
        
        try:
            # Read ONLY the columns we need - MEMORY EFFICIENT
            use_columns = self.input_columns + self.output_columns
            df = pd.read_csv(csv_file_path, usecols=use_columns)
        except Exception as e:
            print(f"❌ Error reading {csv_file_path}: {e}")
            return 0, 0
        
        total_rows = len(df)
        num_shards = max(1, (total_rows + examples_per_shard - 1) // examples_per_shard)
        
        print(f"   Rows: {total_rows:,} | Shards: {num_shards}")
        
        total_written = 0
        
        for shard_id in range(num_shards):
            output_filename = os.path.join(
                file_output_dir, 
                f"{file_base_name}-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
            )
            
            # Use GZIP compression to save space
            options = tf.io.TFRecordOptions(compression_type='GZIP')
            
            start_idx = shard_id * examples_per_shard
            end_idx = min((shard_id + 1) * examples_per_shard, total_rows)
            
            shard_written = 0
            with tf.io.TFRecordWriter(output_filename, options=options) as writer:
                
                for i in tqdm(range(start_idx, end_idx), 
                            desc=f"   Shard {shard_id+1}/{num_shards}",
                            unit="rows",
                            leave=False):
                    
                    row = df.iloc[i]
                    example = self.csv_row_to_tfexample(
                        row, 
                        timestamp=i,
                        file_id=file_base_name
                    )
                    writer.write(example.SerializeToString())
                    shard_written += 1
            
            total_written += shard_written
            print(f"   ✅ Shard {shard_id+1}: {shard_written:,} rows")
        
        conversion_time = time.time() - start_time
        print(f"   🕒 Conversion time: {conversion_time:.2f}s")
        print(f"   📊 Rate: {total_written/conversion_time:.0f} rows/second")
        
        return total_written, num_shards

    def batch_convert_all_files(self, output_base_dir, examples_per_shard=100000):
        """Convert all files - SIMPLE AND RELIABLE"""
        
        # FIXED: Use raw string for Windows paths
        output_base_dir = os.path.normpath(output_base_dir)
        os.makedirs(output_base_dir, exist_ok=True)
        
        total_rows = 0
        total_shards = 0
        conversion_summary = {}
        
        print("🚀 STARTING BASELINE CONVERSION")
        print("=" * 60)
        
        for idx, row in self.file_df.iterrows():
            csv_file = row['filename']
            
            try:
                rows_written, shards_created = self.convert_single_file(
                    csv_file_path=csv_file,
                    output_dir=output_base_dir,
                    examples_per_shard=examples_per_shard
                )
                
                total_rows += rows_written
                total_shards += shards_created
                
                conversion_summary[os.path.basename(csv_file)] = {
                    'rows': rows_written,
                    'shards': shards_created,
                    'status': 'success'
                }
                
            except Exception as e:
                print(f"❌ Failed to convert {csv_file}: {str(e)}")
                conversion_summary[os.path.basename(csv_file)] = {
                    'rows': 0,
                    'shards': 0, 
                    'status': f'failed: {str(e)}'
                }
                continue
        
        # Save conversion info
        schema_info = {
            'input_columns': self.input_columns,
            'output_columns': self.output_columns, 
            'input_dim': len(self.input_columns),
            'output_dim': len(self.output_columns),
            'total_rows': total_rows,
            'total_shards': total_shards,
            'conversion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'BASELINE - No feature engineering'
        }
        
        with open(os.path.join(output_base_dir, 'baseline_info.json'), 'w') as f:
            json.dump(schema_info, f, indent=2)
        
        print("=" * 60)
        print(f"🎉 BASELINE CONVERSION COMPLETE!")
        print(f"📁 Total rows: {total_rows:,}")
        print(f"📁 Total shards: {total_shards}")
        print(f"🎯 Input dimension: {len(self.input_columns)}")
        print(f"🎯 Output dimension: {len(self.output_columns)}")
        print(f"💾 Output: {output_base_dir}")

def generate_baseline_reader():
    """Generate simple reader code for baseline data"""
    
    reader_code = '''import tensorflow as tf
import numpy as np

def parse_baseline_tfrecord(example_proto):
    """Parse baseline TFRecord examples."""
    feature_description = {
        # Input features (6 EMG NMF components)
        'emg_nmf_1': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_2': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_3': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_4': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_5': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_6': tf.io.FixedLenFeature([], tf.float32),
        
        # Output labels (22 Glove dimensions)
        'glove_1': tf.io.FixedLenFeature([], tf.float32),
        'glove_2': tf.io.FixedLenFeature([], tf.float32),
        'glove_3': tf.io.FixedLenFeature([], tf.float32),
        'glove_4': tf.io.FixedLenFeature([], tf.float32),
        'glove_5': tf.io.FixedLenFeature([], tf.float32),
        'glove_6': tf.io.FixedLenFeature([], tf.float32),
        'glove_7': tf.io.FixedLenFeature([], tf.float32),
        'glove_8': tf.io.FixedLenFeature([], tf.float32),
        'glove_9': tf.io.FixedLenFeature([], tf.float32),
        'glove_10': tf.io.FixedLenFeature([], tf.float32),
        'glove_11': tf.io.FixedLenFeature([], tf.float32),
        'glove_12': tf.io.FixedLenFeature([], tf.float32),
        'glove_13': tf.io.FixedLenFeature([], tf.float32),
        'glove_14': tf.io.FixedLenFeature([], tf.float32),
        'glove_15': tf.io.FixedLenFeature([], tf.float32),
        'glove_16': tf.io.FixedLenFeature([], tf.float32),
        'glove_17': tf.io.FixedLenFeature([], tf.float32),
        'glove_18': tf.io.FixedLenFeature([], tf.float32),
        'glove_19': tf.io.FixedLenFeature([], tf.float32),
        'glove_20': tf.io.FixedLenFeature([], tf.float32),
        'glove_21': tf.io.FixedLenFeature([], tf.float32),
        'glove_22': tf.io.FixedLenFeature([], tf.float32),
        
        # Metadata
        'timestamp': tf.io.FixedLenFeature([], tf.int64),
        'file_id': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Create input tensor (6 EMG features)
    inputs = tf.stack([
        parsed['emg_nmf_1'], parsed['emg_nmf_2'], parsed['emg_nmf_3'],
        parsed['emg_nmf_4'], parsed['emg_nmf_5'], parsed['emg_nmf_6']
    ], axis=0)
    
    # Create output tensor (22 Glove positions)
    outputs = tf.stack([parsed[f'glove_{i}'] for i in range(1, 23)], axis=0)
    
    return inputs, outputs

def create_baseline_dataset(tfrecord_pattern, batch_size=32, shuffle_buffer=10000):
    """Create baseline TensorFlow dataset."""
    files = tf.data.Dataset.list_files(tfrecord_pattern)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=4  # Process 4 files in parallel
    )
    dataset = dataset.map(parse_baseline_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# USAGE:
# dataset = create_baseline_dataset('tfrecords_baseline/*/*.tfrecord')
# for inputs, outputs in dataset.take(1):
#     print(f"Input shape: {inputs.shape}")  # (batch_size, 6)
#     print(f"Output shape: {outputs.shape}") # (batch_size, 22)
'''
    
    with open('baseline_reader.py', 'w') as f:
        f.write(reader_code)
    
    print("✅ Baseline TFRecord reader saved to 'baseline_reader.py'")

# SIMPLE TRAINING TEMPLATE
def generate_baseline_model():
    """Generate a simple baseline model template"""
    
    model_code = '''import tensorflow as tf
from tensorflow.keras import layers, Model

def create_baseline_model():
    """Simple baseline model for EMG -> Hand kinematics"""
    inputs = tf.keras.Input(shape=(6,), name='emg_input')
    
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(22, activation='linear', name='glove_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# USAGE:
# model = create_baseline_model()
# model.summary()
# history = model.fit(dataset, epochs=50, validation_split=0.2)
'''
    
    with open('baseline_model.py', 'w') as f:
        f.write(model_code)
    
    print("✅ Baseline model template saved to 'baseline_model.py'")

# MAIN EXECUTION
if __name__ == "__main__":
    # CONFIGURATION - SIMPLE AND CLEAN
    FILE_LIST_PATH = "ready.csv"
    # FIXED: Use raw string for Windows path to avoid escape character issues
    OUTPUT_BASE_DIR = r"F:\stage\v6.0\db\tfrecords_baseline"
    EXAMPLES_PER_SHARD = 100000
    
    print("🚀 LAUNCHING BASELINE CONVERSION")
    print("=" * 60)
    
    # Initialize converter
    converter = BaselineCSVToTFRecordConverter(FILE_LIST_PATH)
    
    # Generate reader and model templates
    generate_baseline_reader()
    generate_baseline_model()
    
    # Convert all files
    converter.batch_convert_all_files(
        output_base_dir=OUTPUT_BASE_DIR,
        examples_per_shard=EXAMPLES_PER_SHARD
    )
    
    print("\n🎯 NEXT STEPS - BASELINE PIPELINE:")
    print("1. Data converted to: tfrecords_baseline/")
    print("2. Use 'baseline_reader.py' to load data in TensorFlow") 
    print("3. Use 'baseline_model.py' to create and train your model")
    print("4. Get your FIRST RESULTS today!")
    print("\n💡 Remember: Working baseline > Perfect pipeline that never runs")
```

---

## File: v6.0\logs\map\size.py
**Path:** `C:\stage\stage\v6.0\logs\map\size.py`

```python
import os
import csv

# Path to your input CSV file (update this if needed)
input_csv = 'ready.csv'  # Replace with your actual CSV file path

total_size = 0
file_count = 0

try:
    with open(input_csv, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header if present (optional; remove if no header)

        for row in reader:
            if not row:  # Skip empty rows
                continue
            file_path = row[0].strip()
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
                print(f"Added: {file_path} ({file_size:,} bytes)")
            else:
                print(f"Warning: File not found - {file_path}")

    print(f"\nTotal files processed: {file_count}")
    print(f"Combined size: {total_size:,} bytes")
    print(f"Combined size: {total_size / (1024**2):.2f} MB")
    print(f"Combined size: {total_size / (1024**3):.2f} GB")

except FileNotFoundError:
    print(f"Error: The CSV file '{input_csv}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
```

---

## File: v6.0\logs\map\test_model.py
**Path:** `C:\stage\stage\v6.0\logs\map\test_model.py`

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from baseline_reader import create_baseline_dataset
import os

def test_loaded_model():
    """Comprehensive testing for the loaded baseline model"""
    
    print("🧪 STARTING COMPREHENSIVE MODEL TESTING")
    print("=" * 50)
    
    # Load the model with custom objects to handle metrics
    try:
        model = tf.keras.models.load_model(
            'best_baseline_model.h5',
            custom_objects={
                'mse': tf.keras.metrics.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError()
            }
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Display model architecture
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Create test dataset
    print("\n📁 Creating test dataset...")
    test_dataset = create_baseline_dataset(
        tfrecord_pattern=r'C:\stage\stage\v6.0\db\tfrecords_baseline/*/*.tfrecord',
        batch_size=1000,  # Larger batch for stable evaluation
        shuffle_buffer=10000
    )
    
    # Take a reasonable subset for testing (avoid using all data)
    test_dataset = test_dataset.take(50)  # 50 batches of 1000 = 50,000 samples
    
    print("\n📈 Evaluating model performance...")
    
    # Manual evaluation to get per-batch metrics
    total_loss = 0
    total_mae = 0
    num_batches = 0
    all_predictions = []
    all_true_values = []
    
    for batch_num, (inputs, true_outputs) in enumerate(test_dataset):
        predictions = model.predict(inputs, verbose=0)
        
        # Calculate batch metrics
        batch_loss = tf.keras.losses.MSE(true_outputs, predictions).numpy().mean()
        batch_mae = tf.keras.metrics.mae(true_outputs, predictions).numpy().mean()
        
        total_loss += batch_loss
        total_mae += batch_mae
        num_batches += 1
        
        # Store for visualization
        if batch_num < 3:  # Only store first few batches for visualization
            all_predictions.extend(predictions)
            all_true_values.extend(true_outputs.numpy())
        
        if batch_num % 10 == 0:
            print(f"   Processed batch {batch_num}, Loss: {batch_loss:.4f}, MAE: {batch_mae:.4f}")
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   Average Loss (MSE): {avg_loss:.4f}")
    print(f"   Average MAE: {avg_mae:.4f}")
    print(f"   Samples tested: {num_batches * 1000}")
    
    return model, np.array(all_predictions), np.array(all_true_values), avg_mae

def visualize_predictions(predictions, true_values, num_samples=5):
    """Visualize model predictions vs true values"""
    
    print(f"\n📊 Visualizing predictions for {num_samples} samples...")
    
    # Select random samples to visualize
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 1, i + 1)
        
        # Plot all 22 glove dimensions
        plt.plot(true_values[idx], 'b-', label='True Glove', alpha=0.7, linewidth=2)
        plt.plot(predictions[idx], 'r--', label='Predicted', alpha=0.7, linewidth=2)
        
        plt.title(f'Sample {i + 1} - Prediction vs Ground Truth')
        plt.ylabel('Hand Position')
        plt.legend()
        
        if i == num_samples - 1:
            plt.xlabel('Glove Dimension')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_error_distribution(predictions, true_values):
    """Analyze error distribution across different glove dimensions"""
    
    errors = np.abs(predictions - true_values)
    mean_errors_per_dimension = np.mean(errors, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, 23), mean_errors_per_dimension)
    plt.title('Average MAE per Glove Dimension')
    plt.xlabel('Glove Dimension')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🔍 ERROR ANALYSIS:")
    print(f"   Worst performing dimension: {np.argmax(mean_errors_per_dimension) + 1} "
          f"(MAE: {np.max(mean_errors_per_dimension):.4f})")
    print(f"   Best performing dimension: {np.argmin(mean_errors_per_dimension) + 1} "
          f"(MAE: {np.min(mean_errors_per_dimension):.4f})")
    print(f"   Overall average MAE: {np.mean(mean_errors_per_dimension):.4f}")

def test_single_prediction(model):
    """Test model on a single example"""
    
    print(f"\n🎯 Testing single prediction...")
    
    # Create a simple test dataset for single prediction
    single_test_dataset = create_baseline_dataset(
        tfrecord_pattern=r'C:\stage\stage\v6.0\db\tfrecords_baseline/*/*.tfrecord',
        batch_size=1,
        shuffle_buffer=1000
    ).take(1)
    
    for inputs, true_outputs in single_test_dataset:
        prediction = model.predict(inputs, verbose=0)
        
        print(f"   Input EMG shape: {inputs.shape}")
        print(f"   True output shape: {true_outputs.shape}")
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Sample prediction for first glove dimension: {prediction[0][0]:.4f}")
        print(f"   True value for first glove dimension: {true_outputs[0][0]:.4f}")

if __name__ == "__main__":
    # Suppress TensorFlow warnings for cleaner output :cite[2]
    tf.get_logger().setLevel('ERROR')
    
    print("🚀 EMG Gesture Recognition Model Testing")
    print("=" * 50)
    
    # Run comprehensive tests
    model, predictions, true_values, test_mae = test_loaded_model()
    
    if model is not None:
        # Visualize results
        visualize_predictions(predictions, true_values)
        analyze_error_distribution(predictions, true_values)
        test_single_prediction(model)
        
        print(f"\n✅ TESTING COMPLETE!")
        print(f"💡 Interpretation:")
        print(f"   - MAE of {test_mae:.4f} means average prediction error of {test_mae:.4f} units")
        print(f"   - Lower values indicate better performance")
        print(f"   - Check visualization files: 'model_predictions.png' and 'error_analysis.png'")
        
        # Compare with your training results
        print(f"\n📊 COMPARISON WITH TRAINING RESULTS:")
        print(f"   Your final validation MAE was: 0.2326")
        print(f"   Current test MAE is: {test_mae:.4f}")
        
        if test_mae <= 0.2326:
            print("   ✅ Model performance is consistent with training!")
        else:
            print("   ⚠️  Model performance differs from training results")
    else:
        print("❌ Testing failed due to model loading issues")
```

---

## File: v6.0\logs\map\test_pipeline.py
**Path:** `C:\stage\stage\v6.0\logs\map\test_pipeline.py`

```python
from baseline_reader import create_baseline_dataset
from baseline_model import create_baseline_model
import tensorflow as tf

# Test data loading
print("🧪 Testing data pipeline...")
dataset = create_baseline_dataset(
    tfrecord_pattern='F:/stage/v6.0/db/tfrecords_baseline/*/*.tfrecord',
    batch_size=32,
    shuffle_buffer=10000
)

# Check one batch
for inputs, outputs in dataset.take(1):
    print(f"✅ Input shape: {inputs.shape}")  # Should be (32, 6)
    print(f"✅ Output shape: {outputs.shape}") # Should be (32, 22)
    print(f"✅ Input range: {inputs.numpy().min():.3f} to {inputs.numpy().max():.3f}")
    print(f"✅ Output range: {outputs.numpy().min():.3f} to {outputs.numpy().max():.3f}")
    break

print("🎉 Data pipeline working correctly!")
```

---

## File: v6.0\logs\map\train_baseline.py
**Path:** `C:\stage\stage\v6.0\logs\map\train_baseline.py`

```python
from baseline_reader import create_baseline_dataset
from baseline_model import create_baseline_model
import tensorflow as tf
import os

# Configuration - OPTIMIZED FOR YOUR HARDWARE
BATCH_SIZE = 64
EPOCHS = 20
VALIDATION_SPLIT = 0.2

print("🚀 Starting Baseline Training - FIXED VERSION")
print("=" * 50)

# Calculate dataset sizes
dataset_size = 16082115  
train_size = int(dataset_size * (1 - VALIDATION_SPLIT))
val_size = dataset_size - train_size

# Calculate steps per epoch
steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE

print(f"📊 Dataset Info:")
print(f"   Total examples: {dataset_size:,}")
print(f"   Training examples: {train_size:,}")
print(f"   Validation examples: {val_size:,}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Training steps per epoch: {steps_per_epoch:,}")
print(f"   Validation steps: {validation_steps:,}")

# Create SEPARATE datasets for training and validation
print("\n📁 Creating training dataset...")
train_dataset = create_baseline_dataset(
    tfrecord_pattern=r'C:\stage\stage\v6.0\db\tfrecords_baseline/*/*.tfrecord',
    batch_size=BATCH_SIZE,
    shuffle_buffer=50000,
    repeat=True  # CRITICAL: Repeat indefinitely for training
)

print("📁 Creating validation dataset...")
val_dataset = create_baseline_dataset(
    tfrecord_pattern=r'C:\stage\stage\v6.0\db\tfrecords_baseline/*/*.tfrecord',
    batch_size=BATCH_SIZE,
    shuffle_buffer=0,  # No shuffle for validation
    repeat=True  # CRITICAL: Repeat for validation too
)

# But we'll manually split by taking appropriate steps
# Training dataset: take only training portion
train_dataset = train_dataset.take(steps_per_epoch)
# Validation dataset: skip training portion and take validation portion  
val_dataset = val_dataset.skip(steps_per_epoch).take(validation_steps)

# Create model
model = create_baseline_model()
print("\n📋 Model Summary:")
model.summary()

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_baseline_model.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    tf.keras.callbacks.CSVLogger('training_log.csv'),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
]

print("\n🏋️ Starting training...")
print("   This will take a while. Grab some coffee! ☕")

# Train the model with EXPLICIT steps
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('final_baseline_model.h5')
print("💾 Models saved:")
print("   - best_baseline_model.h5 (best validation performance)")
print("   - final_baseline_model.h5 (final epoch)")
print("   - training_log.csv (training history)")

# Quick evaluation
print("\n📈 Final Results:")
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_mae = history.history['mae'][-1]
final_val_mae = history.history['val_mae'][-1]

print(f"   Final Training Loss: {final_train_loss:.4f}")
print(f"   Final Validation Loss: {final_val_loss:.4f}")
print(f"   Final Training MAE: {final_train_mae:.4f}")
print(f"   Final Validation MAE: {final_val_mae:.4f}")

# Check for overfitting
if final_val_loss > final_train_loss * 1.2:
    print("⚠️  Warning: Possible overfitting detected")
else:
    print("✅ Good: Training and validation losses are close")

print("\n🎉 Baseline training complete!")
print("Next: Analyze results and visualize predictions")
```

---

## File: v6.0\scripts\data_base_creation\check_val.py
**Path:** `C:\stage\stage\v6.0\scripts\data_base_creation\check_val.py`

```python
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
```

---

## File: v6.0\scripts\data_base_creation\creation_final.py
**Path:** `C:\stage\stage\v6.0\scripts\data_base_creation\creation_final.py`

```python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Configuration
FILE_LIST_CSV = r"D:\stage\v5.0\logs\map\train_map.csv"
TFRecord_OUTPUT_DIR = r"D:\stage\v5.0\db\tfrecords"
LOG_OUTPUT = r"D:\stage\v5.0\logs\creation\processing_log.csv"
ERROR_LOG = r"D:\stage\v5.0\logs\creation\processing_errors.log"
CHUNK_SIZE = 10000
EXAMPLES_PER_TFRECORD = 50000

# Define the correct base path for your files
CORRECT_BASE_PATH = r"E:\DB"

# Define glove column mappings for different databases
GLOVE_COLUMNS_DB7 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_7", "glove_8", "glove_9", "glove_10",
    "glove_11", "glove_12", "glove_13", "glove_14", "glove_15",
    "glove_16", "glove_17", "glove_18"
]

GLOVE_COLUMNS_DB23 = [
    "glove_1", "glove_2", "glove_3", "glove_4", "glove_5",
    "glove_6", "glove_8", "glove_9", "glove_11", "glove_12",
    "glove_13", "glove_15", "glove_16", "glove_17", "glove_19",
    "glove_20", "glove_21", "glove_22"
]

def correct_file_path(original_path):
    if "DB2" in original_path or "DB3" in original_path or "DB7" in original_path:
        parts = original_path.split("DB\\", 1)
        if len(parts) > 1:
            relative_path = parts[1]
            return os.path.join(CORRECT_BASE_PATH, relative_path)
    return original_path

def get_db_type(file_path):
    if "DB7" in file_path:
        return "DB7"
    elif "DB2" in file_path or "DB3" in file_path:
        return "DB23"
    else:
        raise ValueError(f"Unknown DB type in path: {file_path}")

def get_glove_columns(db_type):
    if db_type == "DB7":
        return GLOVE_COLUMNS_DB7
    elif db_type == "DB23":
        return GLOVE_COLUMNS_DB23
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

def create_tf_example(glove_row):
    feature_dict = {
        f'glove_{i}': tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(x)]))
        for i, x in enumerate(glove_row)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def process_file_glove_only(file_path, glove_columns, chunk_size=CHUNK_SIZE):
    try:
        total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1
        data_chunks = []
        rows_processed = 0

        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size),
                         total=int(total_rows/chunk_size)+1,
                         desc=f"Processing {os.path.basename(file_path)}"):
            missing_cols = [col for col in glove_columns if col not in chunk.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
            glove_data = chunk[glove_columns].values.astype(np.float32)
            data_chunks.append(glove_data)
            rows_processed += len(chunk)
        return data_chunks, rows_processed
    except Exception as e:
        raise Exception(f"Error processing {file_path}: {str(e)}")

def main():
    print("Loading file list...")
    df_log = pd.read_csv(FILE_LIST_CSV)
    df_log['corrected_filename'] = df_log['filename'].apply(correct_file_path)
    os.makedirs(TFRecord_OUTPUT_DIR, exist_ok=True)

    processing_log = []
    files_with_errors = []
    tfrecord_mapping_log = []  # NEW: logs which source file rows went into which TFRecord

    tfrecord_counter = 0
    examples_in_current_file = 0
    writer = None
    current_tfrecord_path = ""

    global_row_offset = 0  # Tracks absolute row index across all files

    for idx, row in tqdm(df_log.iterrows(), total=len(df_log), desc="Processing Files"):
        original_path = row['filename']
        file_path = row['corrected_filename']
        expected_count = row.get('rows_extracted', -1)

        print(f"Processing: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            with open(ERROR_LOG, 'a') as elog:
                elog.write(f"{error_msg}\n")
            files_with_errors.append(file_path)
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'status': 'FILE_NOT_FOUND',
                'has_nulls': False
            })
            continue

        try:
            db_type = get_db_type(file_path)
            glove_columns = get_glove_columns(db_type)
            print(f"  → DB Type: {db_type}, Using {len(glove_columns)} glove columns")

            data_chunks, rows_processed = process_file_glove_only(file_path, glove_columns)

            file_start_global = global_row_offset
            file_end_global = global_row_offset + rows_processed - 1

            local_row_in_file = 0

            for chunk in data_chunks:
                for glove_row in chunk:
                    if writer is None or examples_in_current_file >= EXAMPLES_PER_TFRECORD:
                        if writer:
                            writer.close()
                        current_tfrecord_path = os.path.join(TFRecord_OUTPUT_DIR, f"data_{tfrecord_counter:04d}.tfrecord")
                        writer = tf.io.TFRecordWriter(current_tfrecord_path)
                        tfrecord_counter += 1
                        examples_in_current_file = 0

                    example = create_tf_example(glove_row)
                    writer.write(example.SerializeToString())
                    examples_in_current_file += 1

                    # Log mapping: this source file's row → TFRecord file + global index
                    tfrecord_mapping_log.append({
                        'source_file': file_path,
                        'source_row_local': local_row_in_file,
                        'source_row_global': global_row_offset,
                        'tfrecord_file': os.path.basename(current_tfrecord_path),
                        'tfrecord_example_index': examples_in_current_file - 1,
                        'tfrecord_file_index': tfrecord_counter - 1
                    })

                    local_row_in_file += 1
                    global_row_offset += 1

            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': rows_processed,
                'status': 'SUCCESS',
                'has_nulls': False,
                'global_row_start': file_start_global,
                'global_row_end': file_end_global
            })
            print(f"✓ Processed {rows_processed} rows from {os.path.basename(file_path)}")

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            with open(ERROR_LOG, 'a') as elog:
                elog.write(f"{error_msg}\n")
            files_with_errors.append(file_path)
            processing_log.append({
                'filename': file_path,
                'original_filename': original_path,
                'rows_expected': expected_count,
                'rows_processed': 0,
                'status': f'ERROR: {str(e)}',
                'has_nulls': False
            })

    if writer:
        writer.close()

    # Save all logs
    log_df = pd.DataFrame(processing_log)
    log_df.to_csv(LOG_OUTPUT, index=False)

    mapping_df = pd.DataFrame(tfrecord_mapping_log)
    mapping_log_path = os.path.join(TFRecord_OUTPUT_DIR, "tfrecord_mapping.csv")
    mapping_df.to_csv(mapping_log_path, index=False)
    print(f"TFRecord ↔ Source file mapping saved to: {mapping_log_path}")

    total_processed = sum(log['rows_processed'] for log in processing_log if log['status'] == 'SUCCESS')
    print(f"Total rows processed: {total_processed}")
    print(f"TFRecord files created: {tfrecord_counter}")

    if files_with_errors:
        print(f"⚠️  {len(files_with_errors)} files had errors. See {ERROR_LOG}")
    else:
        print("✅ All files processed successfully.")

    metadata = {
        'total_rows': total_processed,
        'glove_columns_DB7': GLOVE_COLUMNS_DB7,
        'glove_columns_DB23': GLOVE_COLUMNS_DB23,
        'tfrecord_files': tfrecord_counter,
        'examples_per_tfrecord': EXAMPLES_PER_TFRECORD,
        'output_feature_names': [f'glove_{i}' for i in range(len(GLOVE_COLUMNS_DB7))]
    }

    with open(os.path.join(TFRecord_OUTPUT_DIR, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2)

    print(f"TFRecord files saved to: {TFRecord_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

---

## File: v7.0\baseline_reader.py
**Path:** `C:\stage\stage\v7.0\baseline_reader.py`

```python
import tensorflow as tf
import numpy as np

def parse_baseline_tfrecord(example_proto):
    """Parse baseline TFRecord examples."""
    feature_description = {
        # Input features (6 EMG NMF components)
        'emg_nmf_1': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_2': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_3': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_4': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_5': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_6': tf.io.FixedLenFeature([], tf.float32),
        
        # Output labels (22 Glove dimensions)
        'glove_1': tf.io.FixedLenFeature([], tf.float32),
        'glove_2': tf.io.FixedLenFeature([], tf.float32),
        'glove_3': tf.io.FixedLenFeature([], tf.float32),
        'glove_4': tf.io.FixedLenFeature([], tf.float32),
        'glove_5': tf.io.FixedLenFeature([], tf.float32),
        'glove_6': tf.io.FixedLenFeature([], tf.float32),
        'glove_7': tf.io.FixedLenFeature([], tf.float32),
        'glove_8': tf.io.FixedLenFeature([], tf.float32),
        'glove_9': tf.io.FixedLenFeature([], tf.float32),
        'glove_10': tf.io.FixedLenFeature([], tf.float32),
        'glove_11': tf.io.FixedLenFeature([], tf.float32),
        'glove_12': tf.io.FixedLenFeature([], tf.float32),
        'glove_13': tf.io.FixedLenFeature([], tf.float32),
        'glove_14': tf.io.FixedLenFeature([], tf.float32),
        'glove_15': tf.io.FixedLenFeature([], tf.float32),
        'glove_16': tf.io.FixedLenFeature([], tf.float32),
        'glove_17': tf.io.FixedLenFeature([], tf.float32),
        'glove_18': tf.io.FixedLenFeature([], tf.float32),
        'glove_19': tf.io.FixedLenFeature([], tf.float32),
        'glove_20': tf.io.FixedLenFeature([], tf.float32),
        'glove_21': tf.io.FixedLenFeature([], tf.float32),
        'glove_22': tf.io.FixedLenFeature([], tf.float32),
        
        # Metadata
        'timestamp': tf.io.FixedLenFeature([], tf.int64),
        'file_id': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Create input tensor (6 EMG features)
    inputs = tf.stack([
        parsed['emg_nmf_1'], parsed['emg_nmf_2'], parsed['emg_nmf_3'],
        parsed['emg_nmf_4'], parsed['emg_nmf_5'], parsed['emg_nmf_6']
    ], axis=0)
    
    # Create output tensor (22 Glove positions)
    outputs = tf.stack([parsed[f'glove_{i}'] for i in range(1, 23)], axis=0)
    
    return inputs, outputs

def create_baseline_dataset(tfrecord_pattern, batch_size=32, shuffle_buffer=10000, repeat=True):
    """Create baseline TensorFlow dataset."""
    files = tf.data.Dataset.list_files(tfrecord_pattern)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=4  # Process 4 files in parallel
    )
    dataset = dataset.map(parse_baseline_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size)
    
    # CRITICAL FIX: Repeat dataset indefinitely for training
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# USAGE:
# dataset = create_baseline_dataset('tfrecords_baseline/*/*.tfrecord')
# for inputs, outputs in dataset.take(1):
#     print(f"Input shape: {inputs.shape}")  # (batch_size, 6)
#     print(f"Output shape: {outputs.shape}") # (batch_size, 22)
```

---

## File: v7.0\check_results.py
**Path:** `C:\stage\stage\v7.0\check_results.py`

```python
# check_results.py
import os
import json
import glob

def check_experiment_results():
    """Check what experiment results and models were saved"""
    
    print("🔍 CHECKING EXPERIMENT RESULTS")
    print("=" * 50)
    
    # Find all experiment folders
    experiment_folders = glob.glob("experiment_*")
    
    if not experiment_folders:
        print("❌ No experiment folders found!")
        return
    
    all_results = {}
    
    for folder in experiment_folders:
        print(f"\n📁 Checking: {folder}")
        
        # Check for models
        model_files = glob.glob(os.path.join(folder, "models", "*.keras"))
        print(f"   Models: {len(model_files)}")
        for model_file in model_files:
            print(f"     - {os.path.basename(model_file)}")
        
        # Check for results JSON
        results_file = os.path.join(folder, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                all_results[folder] = results
                print(f"   ✅ Results: {results.get('model_version', 'Unknown')}")
                print(f"      Test MAE: {results.get('test_mae', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Error reading results: {e}")
        else:
            print("   ❌ No results.json found")
        
        # Check for training logs
        log_file = os.path.join(folder, "training_log.csv")
        if os.path.exists(log_file):
            print(f"   ✅ Training log found")
        else:
            print("   ❌ No training log found")
    
    return all_results

if __name__ == "__main__":
    results = check_experiment_results()
```

---

## File: v7.0\compare_models.py
**Path:** `C:\stage\stage\v7.0\compare_models.py`

```python
# compare_models.py
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import json

def compare_all_models():
    """Compare all trained models and generate visualizations"""
    
    print("📊 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 50)
    
    # Find all experiment folders
    experiment_folders = glob.glob("experiment_*")
    
    if not experiment_folders:
        print("❌ No experiment folders found!")
        return
    
    comparison_data = []
    all_training_logs = {}
    
    # Collect data from each experiment
    for folder in experiment_folders:
        print(f"📁 Processing: {folder}")
        
        # Try to get results from JSON
        results_file = os.path.join(folder, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                model_name = results.get('model_version', folder)
                comparison_data.append({
                    'Model': model_name,
                    'Test MAE': results.get('test_mae', 'N/A'),
                    'Best Test MAE': results.get('best_model_test_mae', 'N/A'),
                    'Validation Loss': results.get('final_validation_loss', 'N/A'),
                    'Training Loss': results.get('final_training_loss', 'N/A'),
                    'Training Time (min)': round(results.get('training_time_seconds', 0) / 60, 1),
                    'Best Epoch': results.get('best_epoch', 'N/A'),
                    'Total Epochs': results.get('total_epochs_trained', 'N/A'),
                    'Folder': folder
                })
            except Exception as e:
                print(f"   ❌ Error reading JSON: {e}")
        
        # Try to get training history from CSV
        log_file = os.path.join(folder, "training_log.csv")
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                model_name = folder.replace('experiment_', '').replace('series_', '')
                all_training_logs[model_name] = df
                print(f"   ✅ Loaded training log: {len(df)} epochs")
            except Exception as e:
                print(f"   ❌ Error reading CSV: {e}")
    
    # Display comparison table
    if comparison_data:
        print(f"\n{'='*100}")
        print(f"{'Model':<15} {'Test MAE':<10} {'Best Test MAE':<12} {'Val Loss':<10} {'Train Loss':<10} {'Time (min)':<10} {'Best Epoch':<10}")
        print(f"{'='*100}")
        for data in comparison_data:
            print(f"{data['Model']:<15} {data['Test MAE']:<10} {data['Best Test MAE']:<12} {data['Validation Loss']:<10} {data['Training Loss']:<10} {data['Training Time (min)']:<10} {data['Best Epoch']:<10}")
        print(f"{'='*100}")
    
    # Generate comparison plots if we have training logs
    if all_training_logs:
        generate_comparison_plots(all_training_logs)
    
    # Find best model
    if comparison_data:
        try:
            # Filter out models with valid MAE values
            valid_models = [m for m in comparison_data if isinstance(m['Best Test MAE'], (int, float))]
            if valid_models:
                best_model = min(valid_models, key=lambda x: x['Best Test MAE'])
                print(f"\n🏆 BEST MODEL: {best_model['Model']}")
                print(f"   Best Test MAE: {best_model['Best Test MAE']:.4f}")
                print(f"   Validation Loss: {best_model['Validation Loss']:.4f}")
                print(f"   Located in: {best_model['Folder']}")
                
                # Save comparison results
                with open('model_comparison_results.json', 'w') as f:
                    json.dump(comparison_data, f, indent=2)
                print(f"\n💾 Comparison saved to: model_comparison_results.json")
        except Exception as e:
            print(f"❌ Error finding best model: {e}")
    
    return comparison_data

def generate_comparison_plots(training_logs):
    """Generate comparison plots from training logs"""
    
    print(f"\n📈 Generating comparison plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training Loss Comparison
    plt.subplot(2, 2, 1)
    for model_name, df in training_logs.items():
        if 'loss' in df.columns:
            plt.plot(df['loss'], label=model_name, alpha=0.8)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss Comparison
    plt.subplot(2, 2, 2)
    for model_name, df in training_logs.items():
        if 'val_loss' in df.columns:
            plt.plot(df['val_loss'], label=model_name, alpha=0.8)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Training MAE Comparison
    plt.subplot(2, 2, 3)
    for model_name, df in training_logs.items():
        if 'mae' in df.columns:
            plt.plot(df['mae'], label=model_name, alpha=0.8)
    plt.title('Training MAE Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Validation MAE Comparison
    plt.subplot(2, 2, 4)
    for model_name, df in training_logs.items():
        if 'val_mae' in df.columns:
            plt.plot(df['val_mae'], label=model_name, alpha=0.8)
    plt.title('Validation MAE Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comparison plots saved as: model_comparison_plots.png")

if __name__ == "__main__":
    results = compare_all_models()
    
    if results:
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("Next steps:")
        print("1. Review the comparison table above")
        print("2. Check 'model_comparison_plots.png' for training curves")
        print("3. Look at 'model_comparison_results.json' for detailed data")
        print("4. Use the best model for further testing or create improved versions")
    else:
        print("❌ No valid results found for comparison")
```

---

## File: v7.0\train.py
**Path:** `C:\stage\stage\v7.0\train.py`

```python
# complete_training_script.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
from baseline_reader import create_baseline_dataset

def create_improved_model_v1(model_name="improved_v1"):
    """
    Improved model with regularization to reduce overfitting
    This is MODEL VERSION 1 - Enhanced Baseline
    """
    inputs = tf.keras.Input(shape=(6,), name='emg_input')
    
    # Enhanced architecture with regularization
    x = tf.keras.layers.Dense(128, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(22, activation='linear', name='glove_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_improved_model_v2(model_name="improved_v2"):
    """
    MODEL VERSION 2 - Deeper Architecture
    """
    inputs = tf.keras.Input(shape=(6,), name='emg_input')
    
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    outputs = tf.keras.layers.Dense(22, activation='linear', name='glove_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_improved_model_v3(model_name="improved_v3"):
    """
    MODEL VERSION 3 - Wider but Shallower
    """
    inputs = tf.keras.Input(shape=(6,), name='emg_input')
    
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(22, activation='linear', name='glove_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def setup_experiment_directory(experiment_name):
    """Create directory structure for experiment results"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    model_dir = os.path.join(exp_dir, "models")
    plot_dir = os.path.join(exp_dir, "plots")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    return exp_dir, model_dir, plot_dir

def train_model(model, model_version, experiment_name="baseline"):
    """
    Complete training function with automatic model saving and evaluation
    """
    print(f"🚀 STARTING TRAINING FOR {model_version}")
    print("=" * 60)
    
    # Setup directories
    exp_dir, model_dir, plot_dir = setup_experiment_directory(experiment_name)
    
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 50
    
    # Calculate dataset sizes
    dataset_size = 16082115
    train_size = int(dataset_size * 0.7)   # 70% training
    val_size = int(dataset_size * 0.15)    # 15% validation
    test_size = int(dataset_size * 0.15)   # 15% testing
    
    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE
    test_steps = test_size // BATCH_SIZE
    
    print(f"📊 Dataset Split:")
    print(f"   Training: {train_size:,} samples")
    print(f"   Validation: {val_size:,} samples") 
    print(f"   Test: {test_size:,} samples")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    
    # Create datasets
    full_dataset = create_baseline_dataset(
        tfrecord_pattern=r'C:\stage\stage\v6.0\db\tfrecords_baseline/*/*.tfrecord',
        batch_size=BATCH_SIZE,
        shuffle_buffer=50000
    ).repeat()
    
    # Split datasets
    train_dataset = full_dataset.take(steps_per_epoch)
    val_dataset = full_dataset.skip(steps_per_epoch).take(validation_steps)
    test_dataset = full_dataset.skip(steps_per_epoch + validation_steps).take(test_steps)
    
    # Model file paths
    best_model_path = os.path.join(model_dir, f"best_{model_version}.keras")
    final_model_path = os.path.join(model_dir, f"final_{model_version}.keras")
    
    # Enhanced callbacks :cite[6]:cite[10]
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_dir, 'logs'))
    ]
    
    print(f"📋 Model Summary for {model_version}:")
    model.summary()
    
    # Train the model
    print(f"\n🏋️ Training {model_version}...")
    start_time = datetime.datetime.now()
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.datetime.now() - start_time
    
    # Save final model
    model.save(final_model_path)
    
    # Evaluate on test set
    print(f"\n🧪 Evaluating {model_version} on test set...")
    test_loss, test_mae = model.evaluate(test_dataset, steps=test_steps, verbose=0)
    
    # Load best model for final evaluation
    best_model = tf.keras.models.load_model(best_model_path)
    best_test_loss, best_test_mae = best_model.evaluate(test_dataset, steps=test_steps, verbose=0)
    
    # Save results
    results = {
        'model_version': model_version,
        'experiment_name': experiment_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'training_time_seconds': training_time.total_seconds(),
        'final_training_loss': history.history['loss'][-1],
        'final_training_mae': history.history['mae'][-1],
        'final_validation_loss': history.history['val_loss'][-1],
        'final_validation_mae': history.history['val_mae'][-1],
        'test_loss': test_loss,
        'test_mae': test_mae,
        'best_model_test_loss': best_test_loss,
        'best_model_test_mae': best_test_mae,
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': np.argmin(history.history['val_loss']) + 1,
        'model_architecture': model.to_json()
    }
    
    # Save results to JSON
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create training history plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_version} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_version} - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_version}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results summary
    print(f"\n🎯 {model_version} - FINAL RESULTS:")
    print(f"   Training Loss: {results['final_training_loss']:.4f}")
    print(f"   Validation Loss: {results['final_validation_loss']:.4f}")
    print(f"   Test Loss: {results['test_loss']:.4f}")
    print(f"   Test MAE: {results['test_mae']:.4f}")
    print(f"   Best Model Test MAE: {results['best_model_test_mae']:.4f}")
    print(f"   Training Time: {training_time}")
    print(f"   Best Epoch: {results['best_epoch']}")
    
    # Overfitting check
    overfitting_ratio = results['final_validation_loss'] / results['final_training_loss']
    if overfitting_ratio > 1.2:
        print(f"   ⚠️  Overfitting detected (ratio: {overfitting_ratio:.2f})")
    else:
        print(f"   ✅ Good generalization (ratio: {overfitting_ratio:.2f})")
    
    print(f"\n💾 Models and results saved to: {exp_dir}")
    print(f"   - Best model: {best_model_path}")
    print(f"   - Final model: {final_model_path}")
    print(f"   - Training log: {os.path.join(exp_dir, 'training_log.csv')}")
    print(f"   - Results: {os.path.join(exp_dir, 'results.json')}")
    print(f"   - Plots: {os.path.join(plot_dir, f'{model_version}_training_history.png')}")
    
    return results, best_model_path

def run_experiment_series():
    """
    Run a series of model experiments
    """
    print("🔬 STARTING MODEL EXPERIMENT SERIES")
    print("=" * 60)
    
    models_to_train = {
        'improved_v1': create_improved_model_v1,
        'improved_v2': create_improved_model_v2, 
        'improved_v3': create_improved_model_v3,
    }
    
    all_results = {}
    
    for model_name, model_creator in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"🎯 TRAINING {model_name.upper()}")
        print(f"{'='*50}")
        
        try:
            model = model_creator()
            results, best_model_path = train_model(model, model_name, f"series_{model_name}")
            all_results[model_name] = {
                'results': results,
                'best_model_path': best_model_path
            }
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Compare all models
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SERIES COMPARISON")
    print(f"{'='*60}")
    
    comparison_data = []
    for model_name, data in all_results.items():
        results = data['results']
        comparison_data.append({
            'Model': model_name,
            'Test MAE': f"{results['test_mae']:.4f}",
            'Best Test MAE': f"{results['best_model_test_mae']:.4f}",
            'Validation Loss': f"{results['final_validation_loss']:.4f}",
            'Training Time': f"{results['training_time_seconds']/60:.1f} min",
            'Best Epoch': results['best_epoch']
        })
    
    # Print comparison table
    print("\n" + "="*90)
    print(f"{'Model':<12} {'Test MAE':<10} {'Best Test MAE':<14} {'Val Loss':<10} {'Time':<12} {'Best Epoch':<10}")
    print("="*90)
    for data in comparison_data:
        print(f"{data['Model']:<12} {data['Test MAE']:<10} {data['Best Test MAE']:<14} {data['Validation Loss']:<10} {data['Time']:<12} {data['Best Epoch']:<10}")
    print("="*90)
    
    # Save comparison results
    with open('experiment_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Comparison saved to: experiment_comparison.json")
    
    # Find best model
    best_model = min(comparison_data, key=lambda x: float(x['Best Test MAE']))
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
    print(f"   Best Test MAE: {best_model['Best Test MAE']}")
    
    return all_results

if __name__ == "__main__":
    # Suppress TensorFlow warnings for cleaner output
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("🤖 EMG GESTURE RECOGNITION - MODEL EXPERIMENTATION PLATFORM")
    print("=" * 60)
    
    # Option 1: Train a single model
    # model = create_improved_model_v1()
    # results, best_model_path = train_model(model, "improved_v1", "single_run")
    
    # Option 2: Run the full experiment series
    all_results = run_experiment_series()
    
    print(f"\n🎉 EXPERIMENTATION COMPLETE!")
    print("Next steps:")
    print("1. Review the results in the experiment_* folders")
    print("2. Check training plots and metrics") 
    print("3. Use the best model for further testing")
    print("4. Modify model architectures and run again!")
```

---

## File: v8.0\baseline_reader.py
**Path:** `C:\stage\stage\v8.0\baseline_reader.py`

```python
# baseline_reader.py
import tensorflow as tf

def parse_baseline_tfrecord(example_proto):
    """
    Parses a single tf.train.Example proto for the baseline EMG->Glove data.
    This function defines the data schema and converts the serialized data
    back into tensors.
    """
    # Define the structure of the data saved in the TFRecord files.
    feature_description = {
        'emg_nmf_1': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_2': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_3': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_4': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_5': tf.io.FixedLenFeature([], tf.float32),
        'emg_nmf_6': tf.io.FixedLenFeature([], tf.float32),

        'glove_1': tf.io.FixedLenFeature([], tf.float32),
        'glove_2': tf.io.FixedLenFeature([], tf.float32),
        'glove_3': tf.io.FixedLenFeature([], tf.float32),
        'glove_4': tf.io.FixedLenFeature([], tf.float32),
        'glove_5': tf.io.FixedLenFeature([], tf.float32),
        'glove_6': tf.io.FixedLenFeature([], tf.float32),
        'glove_7': tf.io.FixedLenFeature([], tf.float32),
        'glove_8': tf.io.FixedLenFeature([], tf.float32),
        'glove_9': tf.io.FixedLenFeature([], tf.float32),
        'glove_10': tf.io.FixedLenFeature([], tf.float32),
        'glove_11': tf.io.FixedLenFeature([], tf.float32),
        'glove_12': tf.io.FixedLenFeature([], tf.float32),
        'glove_13': tf.io.FixedLenFeature([], tf.float32),
        'glove_14': tf.io.FixedLenFeature([], tf.float32),
        'glove_15': tf.io.FixedLenFeature([], tf.float32),
        'glove_16': tf.io.FixedLenFeature([], tf.float32),
        'glove_17': tf.io.FixedLenFeature([], tf.float32),
        'glove_18': tf.io.FixedLenFeature([], tf.float32),
        'glove_19': tf.io.FixedLenFeature([], tf.float32),
        'glove_20': tf.io.FixedLenFeature([], tf.float32),
        'glove_21': tf.io.FixedLenFeature([], tf.float32),
        'glove_22': tf.io.FixedLenFeature([], tf.float32),

        # Metadata is not used by the model but must be in the parser.
        'timestamp': tf.io.FixedLenFeature([], tf.int64),
        'file_id': tf.io.FixedLenFeature([], tf.string),
    }

    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Group the features into an input tensor (shape: [6])
    inputs = tf.stack([
        parsed['emg_nmf_1'], parsed['emg_nmf_2'], parsed['emg_nmf_3'],
        parsed['emg_nmf_4'], parsed['emg_nmf_5'], parsed['emg_nmf_6']
    ], axis=0)

    # Group the labels into an output tensor (shape: [22])
    outputs = tf.stack([parsed[f'glove_{i}'] for i in range(1, 23)], axis=0)

    return inputs, outputs
```

---

## File: v8.0\train_final.py
**Path:** `C:\stage\stage\v8.0\train_final.py`

```python
# train_final.py
# This script is a complete, robust framework for EMG model experimentation.
# It includes:
#   1. Automatic, one-time creation of a fixed data split for reproducibility.
#   2. Loading of the fixed split for all training runs to ensure fair comparison.
#   3. Scientifically valid data pipeline to prevent data leakage.
#   4. Fixes for common TensorFlow 'OUT_OF_RANGE' and JSON serialization errors.
#   5. Orchestration for training and comparing multiple model architectures.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import glob
import random
from baseline_reader import parse_baseline_tfrecord # Ensure baseline_reader.py is in the same directory

# --- ⚙️ GLOBAL CONFIGURATION ---
# Define your project paths and settings here for easy access.
TFRECORD_PATTERN = r'C:\stage\stage\v6.0\db\tfrecords_baseline\*\*.tfrecord' # IMPORTANT: Update with your path
SPLIT_FILE_PATH = 'data_split.json' # The file to store our fixed data split
TOTAL_EXAMPLES = 16082115 # From your 'baseline_info.json'
BATCH_SIZE = 64
EPOCHS = 50

# --- 🗂️ DATA PREPARATION & SPLITTING ---
def prepare_and_load_data_split():
    """
    Manages the data split. Creates a new random split and saves it if one doesn't exist.
    If a split file exists, it loads it to ensure reproducibility.
    """
    if not os.path.exists(SPLIT_FILE_PATH):
        print(f"INFO: No data split file found. Creating a new one at '{SPLIT_FILE_PATH}'...")
        all_files = glob.glob(TFRECORD_PATTERN)
        if not all_files:
            raise FileNotFoundError(f"FATAL: No TFRecord files found matching the pattern: {TFRECORD_PATTERN}. Please check the path.")
        
        random.shuffle(all_files)

        train_split = 0.7
        val_split = 0.15
        train_size = int(len(all_files) * train_split)
        val_size = int(len(all_files) * val_split)

        split_data = {
            'train': all_files[:train_size],
            'validation': all_files[train_size : train_size + val_size],
            'test': all_files[train_size + val_size:]
        }

        with open(SPLIT_FILE_PATH, 'w') as f:
            json.dump(split_data, f, indent=2)
        print("✅ Data split created and saved successfully.")
    else:
        print(f"✅ Using existing data split file from '{SPLIT_FILE_PATH}'.")

    # Load and return the split data
    with open(SPLIT_FILE_PATH, 'r') as f:
        split_data = json.load(f)
    return split_data['train'], split_data['validation'], split_data['test']

def create_dataset_from_files(file_list, batch_size, shuffle_buffer=0, repeat=False):
    """Creates a TensorFlow dataset from a pre-defined list of TFRecord files."""
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=4
    )
    dataset = dataset.map(parse_baseline_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 🏛️ MODEL ARCHITECTURES ---
def create_improved_model_v1(model_name="improved_v1"):
    """MODEL VERSION 1 - Enhanced Baseline with Regularization"""
    inputs = tf.keras.Input(shape=(6,), name='emg_input')
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(22, activation='linear', name='glove_output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model

def create_improved_model_v2(model_name="improved_v2"):
    """MODEL VERSION 2 - Deeper Architecture"""
    inputs = tf.keras.Input(shape=(6,), name='emg_input')
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(22, activation='linear', name='glove_output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='mse', metrics=['mae'])
    return model

def create_improved_model_v3(model_name="improved_v3"):
    """MODEL VERSION 3 - Wider but Shallower"""
    inputs = tf.keras.Input(shape=(6,), name='emg_input')
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(22, activation='linear', name='glove_output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model

# --- 🏋️ TRAINING & EVALUATION ---
def setup_experiment_directory(experiment_name):
    """Creates a unique directory for storing all artifacts of a training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{experiment_name}_{timestamp}"
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def train_model(model, model_version, train_files, val_files, test_files, experiment_name="series"):
    """The main function to train a single model and save all results."""
    print(f"🚀 STARTING TRAINING FOR {model_version}")
    print("=" * 60)

    exp_dir = setup_experiment_directory(f"{experiment_name}_{model_version}")
    
    # Create datasets
    train_dataset = create_dataset_from_files(train_files, BATCH_SIZE, shuffle_buffer=50000, repeat=True)
    val_dataset = create_dataset_from_files(val_files, BATCH_SIZE)
    test_dataset = create_dataset_from_files(test_files, BATCH_SIZE)

    # Calculate steps_per_epoch
    train_examples = int(TOTAL_EXAMPLES * 0.7) # Based on 70% split
    steps_per_epoch = train_examples // BATCH_SIZE
    print(f"   Calculated steps_per_epoch: {steps_per_epoch:,}")

    # Callbacks
    best_model_path = os.path.join(exp_dir, "models", f"best_{model_version}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(exp_dir, 'training_log.csv')),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_dir, 'logs'))
    ]

    print(f"\nTraining {model_version}...")
    model.summary()
    start_time = datetime.datetime.now()

    history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, callbacks=callbacks, verbose=1)

    training_time = datetime.datetime.now() - start_time
    
    # Evaluate on the unseen test set
    print(f"\n🧪 Evaluating {model_version} on unseen test data...")
    best_model = tf.keras.models.load_model(best_model_path)
    best_test_loss, best_test_mae = best_model.evaluate(test_dataset, verbose=0)

    # Save results with JSON serialization fix
    results = {
        'model_version': model_version,
        'best_model_test_loss': float(best_test_loss),
        'best_model_test_mae': float(best_test_mae),
        'total_epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_loss']) + 1),
        'final_validation_loss': float(history.history['val_loss'][-1]),
        'training_time_seconds': training_time.total_seconds(),
        'model_architecture': json.loads(model.to_json()) # Load to ensure it's a valid dict
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Create and save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_version} - Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE'); plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_version} - MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f'{model_version}_training_history.png'))
    plt.close()

    print(f"\n🎯 {model_version} - FINAL RESULTS:")
    print(f"   Best Model Test MAE: {results['best_model_test_mae']:.4f}")
    print(f"   Training Time: {str(training_time).split('.')[0]}")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"💾 Results and artifacts saved to: {exp_dir}")
    return results

# --- 🔬 EXPERIMENT ORCHESTRATION ---
def run_experiment_series():
    """Main function to run the entire experiment series."""
    print("="*70)
    print("🔬 EMG-TO-GLOVE MODEL EXPERIMENTATION PLATFORM 🔬")
    print("="*70)

    # Step 1: Prepare the data split (load existing or create new)
    train_files, val_files, test_files = prepare_and_load_data_split()
    print(f"   Data partitions loaded: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files.\n")

    # Step 2: Define models to train
    models_to_train = {
        #'improved_v1': create_improved_model_v1,
        #'improved_v2': create_improved_model_v2,
        'improved_v3': create_improved_model_v3,
    }

    # Step 3: Iterate and train each model
    all_results = []
    for model_name, model_creator in models_to_train.items():
        print(f"\n{'='*50}\n>>>>>  TRAINING MODEL: {model_name.upper()}  <<<<<\n{'='*50}")
        try:
            model = model_creator()
            results = train_model(model, model_name, train_files, val_files, test_files)
            all_results.append(results)
        except Exception as e:
            print(f"❌❌❌ CRITICAL ERROR training {model_name}: {e} ❌❌❌")
            import traceback
            traceback.print_exc() # Print full error for debugging
            continue
    
    print(f"\n{'='*70}\n📊 EXPERIMENT SERIES COMPLETE - FINAL COMPARISON 📊\n{'='*70}")
    if not all_results:
        print("No models were trained successfully. Exiting.")
        return

    # Print summary table
    print(f"{'Model':<15} {'Test MAE':<12} {'Best Epoch':<12} {'Training Time (min)':<20}")
    print('-'*70)
    for res in sorted(all_results, key=lambda x: x['best_model_test_mae']):
        time_min = res['training_time_seconds'] / 60
        print(f"{res['model_version']:<15} {res['best_model_test_mae']:.4f}{'':<8} {res['best_epoch']:<12} {time_min:<20.1f}")
    
    # Identify and announce the best model
    best_model_results = min(all_results, key=lambda x: x['best_model_test_mae'])
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model_results['model_version']} (Test MAE: {best_model_results['best_model_test_mae']:.4f})\n")


if __name__ == "__main__":
    # Suppress verbose TensorFlow logging for cleaner output
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    run_experiment_series()
    
    print(f"🎉 All experiments are complete!")
    print("Next steps: Review the 'experiment_*' folders for detailed plots and logs.")
```

---

## File: v9.0\convert_baseline.py
**Path:** `C:\stage\stage\v9.0\convert_baseline.py`

```python
# Save this as convert_sequential.py

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import time
import argparse

# --- CONFIGURATION ---
# All key hyperparameters are here.
CONFIG = {
    "FILE_LIST_PATH": "ready.csv",
    "OUTPUT_DIR": r"F:\stage\v6.0\db\tfrecords_sequential", # <-- NEW Directory!
    "EXAMPLES_PER_SHARD": 20000, # Sequences are larger, so fewer per shard is fine
    "SEQUENCE_LENGTH": 500,      # (Hyperparameter) The number of time steps in one example
    "TARGET_SHIFT": 300,         # (Hyperparameter) How many steps *after* the sequence to predict
    "INPUT_COLS": [f'emg_nmf_{i}' for i in range(1, 7)],
    "OUTPUT_COLS": [f'glove_{i}' for i in range(1, 23)]
}
# ---------------------

def make_sequence_example(nmf_seq: np.ndarray, glove_vec: np.ndarray) -> tf.train.Example:
    """
    Serializes a sequence (array) and a target vector (array) into a tf.train.Example.
    We use tf.io.serialize_tensor for efficient array storage.
    """
    nmf_seq_bytes = tf.io.serialize_tensor(nmf_seq.astype(np.float32)).numpy()
    glove_vec_bytes = tf.io.serialize_tensor(glove_vec.astype(np.float32)).numpy()

    feature_dict = {
        'nmf_sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[nmf_seq_bytes])),
        'glove_target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[glove_vec_bytes])),
        # Add shape info for validation
        'nmf_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=nmf_seq.shape)),
        'glove_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=glove_vec.shape)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def process_file_to_sequences(csv_path: str, writer: tf.io.TFRecordWriter, cfg: dict) -> int:
    """
    Reads a single CSV, windows it into sequences, and writes them to the TFRecordWriter.
    """
    seq_len = cfg["SEQUENCE_LENGTH"]
    shift = cfg["TARGET_SHIFT"]
    
    try:
        df = pd.read_csv(csv_path, usecols=cfg["INPUT_COLS"] + cfg["OUTPUT_COLS"])
    except Exception as e:
        print(f"   ❌ Error reading {csv_path}: {e}")
        return 0

    X = df[cfg["INPUT_COLS"]].values
    Y = df[cfg["OUTPUT_COLS"]].values
    n = X.shape[0]
    written = 0
    
    # Calculate the last possible start index
    # We need L steps for the sequence and S steps for the shift
    last_start_index = n - seq_len - shift
    
    if last_start_index <= 0:
        print(f"   ⚠️ Warning: File {csv_path} is too short ({n} rows) for sequence length {seq_len} + shift {shift}. Skipping.")
        return 0

    # Slide the window over the file
    for start in range(last_start_index):
        seq_end = start + seq_len
        target_idx = seq_end + shift
        
        # Extract the sequence and the target
        nmf_seq = X[start:seq_end, :]
        glove_target = Y[target_idx, :]
        
        # Create and write the example
        example = make_sequence_example(nmf_seq, glove_target)
        writer.write(example.SerializeToString())
        written += 1

    return written

def main(config):
    """Main conversion function."""
    print("🚀 STARTING SEQUENTIAL CONVERSION (PHASE 2)")
    print("=" * 60)
    print(f"   Sequence Length: {config['SEQUENCE_LENGTH']}")
    print(f"   Target Shift: {config['TARGET_SHIFT']}")
    print(f"   Outputting to: {config['OUTPUT_DIR']}")
    
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    
    try:
        file_df = pd.read_csv(config["FILE_LIST_PATH"])
        files_to_process = file_df['filename'].tolist()
    except Exception as e:
        print(f"❌ FATAL: Could not read file list '{config['FILE_LIST_PATH']}': {e}")
        return

    shard_idx = 0
    examples_in_shard = 0
    total_written = 0
    writer = None
    metadata = {
        "params": {
            "sequence_length": config["SEQUENCE_LENGTH"],
            "target_shift": config["TARGET_SHIFT"],
            "examples_per_shard": config["EXAMPLES_PER_SHARD"],
            "input_cols": config["INPUT_COLS"],
            "output_cols": config["OUTPUT_COLS"]
        },
        "files": {},
        "total_examples": 0
    }

    def open_new_writer(idx: int):
        path = os.path.join(config["OUTPUT_DIR"], f'data_{idx:04d}.tfrecord')
        # We will use GZIP compression for this one
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        return tf.io.TFRecordWriter(path, options=options), path

    writer, current_path = open_new_writer(shard_idx)
    print(f"   Writing to shard: {current_path}")

    for fp in tqdm(files_to_process, desc='Total Files', unit='file'):
        print(f"\n📤 Processing: {os.path.basename(fp)}")
        
        written_for_file = process_file_to_sequences(fp, writer, config)
        
        if written_for_file == 0:
            metadata["files"][fp] = {"examples": 0, "status": "skipped_too_short"}
            continue
            
        total_written += written_for_file
        examples_in_shard += written_for_file
        metadata["files"][fp] = {"examples": written_for_file, "status": "success"}
        print(f"   ✅ Wrote {written_for_file:,} sequences from this file.")
        
        # Check if we need to roll over to a new shard
        if examples_in_shard >= config["EXAMPLES_PER_SHARD"]:
            writer.close()
            print(f"   ... Shard {shard_idx:04d} complete ({examples_in_shard:,} examples).")
            shard_idx += 1
            examples_in_shard = 0
            writer, current_path = open_new_writer(shard_idx)
            print(f"   Writing to new shard: {current_path}")

    if writer:
        writer.close()
    print(f"   ... Final shard {shard_idx:04d} complete ({examples_in_shard:,} examples).")
    
    metadata["total_examples"] = int(total_written)
    
    # Save metadata
    meta_path = os.path.join(config["OUTPUT_DIR"], 'metadata.json')
    with open(meta_path, 'w') as jf:
        json.dump(metadata, jf, indent=2)
        
    print("\n" + "=" * 60)
    print(f"🎉 SEQUENTIAL CONVERSION COMPLETE!")
    print(f"   Total sequences written: {total_written:,}")
    print(f"   Total shards: {shard_idx + 1}")
    print(f"   Metadata saved to: {meta_path}")

if __name__ == "__main__":
    # This allows you to override config from the command line if you want
    # e.g.: python convert_sequential.py --SEQUENCE_LENGTH 300
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEQUENCE_LENGTH', type=int, default=CONFIG['SEQUENCE_LENGTH'])
    parser.add_argument('--TARGET_SHIFT', type=int, default=CONFIG['TARGET_SHIFT'])
    parser.add_argument('--OUTPUT_DIR', type=str, default=CONFIG['OUTPUT_DIR'])
    args = parser.parse_args()

    # Update config with any command-line arguments
    CONFIG['SEQUENCE_LENGTH'] = args.SEQUENCE_LENGTH
    CONFIG['TARGET_SHIFT'] = args.TARGET_SHIFT
    CONFIG['OUTPUT_DIR'] = args.OUTPUT_DIR
    
    main(CONFIG)
```

---

## File: v9.0\sequence_reader.py
**Path:** `C:\stage\stage\v9.0\sequence_reader.py`

```python
# Save this as sequence_reader.py

import tensorflow as tf

# These must match the config in your conversion script
SEQUENCE_LENGTH = 500
NUM_FEATURES = 6
NUM_TARGETS = 22

def parse_sequence_tfrecord(example_proto):
    """
    Parses TFRecord examples created by convert_sequential.py.
    This reads the serialized tensors for sequence data.
    """
    feature_description = {
        'nmf_sequence': tf.io.FixedLenFeature([], tf.string),
        'glove_target': tf.io.FixedLenFeature([], tf.string),
        'nmf_shape': tf.io.FixedLenFeature([2], tf.int64), 
        'glove_shape': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the byte strings back into tensors
    nmf_sequence = tf.io.parse_tensor(parsed['nmf_sequence'], out_type=tf.float32)
    glove_target = tf.io.parse_tensor(parsed['glove_target'], out_type=tf.float32)
    
    # Set the shapes explicitly, as parsing from bytes loses shape information
    # This is critical for the model
    nmf_sequence = tf.reshape(nmf_sequence, [SEQUENCE_LENGTH, NUM_FEATURES]) 
    glove_target = tf.reshape(glove_target, [NUM_TARGETS])
    
    return nmf_sequence, glove_target

def create_sequence_dataset(tfrecord_pattern, batch_size, shuffle_buffer=10000, repeat=False):
    """
    Creates a TensorFlow dataset from a file pattern of sequential TFRecords.
    """
    files = tf.data.Dataset.list_files(tfrecord_pattern)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), # Use GZIP
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=4
    )
    dataset = dataset.map(parse_sequence_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size)
    
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# USAGE:
# from sequence_reader import create_sequence_dataset
# train_ds = create_sequence_dataset('path/to/tfrecords_sequential/*.tfrecord', batch_size=64)
# for seq, target in train_ds.take(1):
#     print(f"Sequence batch shape: {seq.shape}") # (64, 500, 6)
#     print(f"Target batch shape: {target.shape}") # (64, 22)
```

---

## File: v9.0\data_creation\analyze_10hz_data.py
**Path:** `C:\stage\stage\v9.0\data_creation\analyze_10hz_data.py`

```python
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
    
    move_df, glove_data_scaled = load_and_prepare_data(Config.DATA_CSV_FILE)
    
    if move_df is not None:
        analyze_movement_clusters(move_df, glove_data_scaled)
        
    print("\n" + "=" * 60)
    print("🎉 PLOTTING COMPLETE!")
    print("   Ensure the new 'movement_clusters_pca.png' is used in Figure 3.13.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v9.0\data_creation\analyze_10hz_data_v2.py
**Path:** `C:\stage\stage\v9.0\data_creation\analyze_10hz_data_v2.py`

```python
# Save this as analyze_10hz_data_v2.py
# This script loads the 10Hz feature data, creates new "Virtual Motor"
# columns, and calculates their true Min (rest) and Max (movement) values.

import pandas as pd
import numpy as np
import os
from typing import Dict, List

# --- ⚙️ CONFIGURATION ---
DATA_FILE = "10hz_feature_data.parquet" # The file created by Phase 2a

# 1. DEFINE VIRTUAL MOTOR GROUPINGS
# Based on the Cyberglove image and the correlation heatmap,
# we define which sensors belong to which "motor".
# We use the 0-indexed column names ('glove_0', 'glove_1', etc.)
VIRTUAL_MOTORS: dict[str, list[str]] = {
    # A. Primary Flexion/Abduction (7 Motors)
    "motor_thumb_flex": ["glove_3", "glove_4"],
    "motor_index_flex": ["glove_5", "glove_6", "glove_7"],
    "motor_middle_flex": ["glove_8", "glove_9", "glove_10"],
    "motor_ring_flex": ["glove_12", "glove_13", "glove_14"],
    "motor_pinky_flex": ["glove_16", "glove_17", "glove_18"],
    "motor_thumb_abduct": ["glove_2"],
    "motor_wrist_flex_ext": ["glove_21"], 

    # B. Secondary Adduction/Waving (5 Motors - Enhanced)
    "motor_index_adduct": ["glove_11"],
    "motor_middle_adduct": ["glove_15"],
    "motor_ring_adduct": ["glove_19"],
    "motor_pinky_adduct": ["glove_20"],
    "motor_hand_waving": ["glove_1", "glove_22"]
}

# --- 1. LOAD THE DATASET ---
def load_data(file_path):
    """Loads the 10Hz data from the parquet file."""
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ FATAL: File not found: {file_path}")
        return None
    
    try:
        df = pd.read_parquet(file_path)
        print(f"   ... Success. Loaded {len(df):,} total 10Hz samples.")
        return df
    except Exception as e:
        print(f"❌ FATAL: Could not read parquet file. Error: {e}")
        return None

# --- 2. CREATE VIRTUAL MOTOR COLUMNS ---
def create_virtual_motors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Averages the raw sensor columns to create the new "motor" columns.
    """
    print("\n--- 🔬 1. Creating Virtual Motors ---")
    
    motor_cols = []
    for motor_name, sensor_list in VIRTUAL_MOTORS.items():
        print(f"   ... Creating '{motor_name}' from {len(sensor_list)} sensors.")
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
        print("   ⚠️ Warning: Data sample is missing either 'rest' or 'movement' data.")
        print("   ... Analysis will be incomplete. Please use a larger data sample.")
        return

    # Calculate "Min" (the average value at rest)
    rest_mins = rest_df[motor_cols].mean()
    
    # Calculate "Max" (the 99th percentile of movement)
    # Using 99th percentile is more robust to outliers than a hard .max()
    move_maxs = move_df[motor_cols].quantile(0.99)
    
    # Combine into a final report
    analysis_report = pd.DataFrame({
        "Rest_Value (Min)": rest_mins,
        "99th_Percentile_Move (Max)": move_maxs
    })
    
    # Add a "Range" column for our information
    analysis_report["Range (Max - Min)"] = analysis_report["99th_Percentile_Move (Max)"] - analysis_report["Rest_Value (Min)"]

    print("\n✅ ANALYSIS COMPLETE. These are the values for our 0-to-1 normalization:\n")
    print(analysis_report.to_string(float_format="%.2f"))
    
    # Save this report to a file for our next script
    report_file = "motor_normalization_params.json"
    analysis_report.to_json(report_file, indent=2)
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
        print(f"   Please review the printed table and the '{motor_cols[0]}_normalization_params.json' file.")
        print("   These Min/Max values are what we will use to build our final, normalized TFRecord dataset.")
        print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## File: v9.0\data_creation\convert_to_10hz_csv.py
**Path:** `C:\stage\stage\v9.0\data_creation\convert_to_10hz_csv.py`

```python
# Save this as convert_to_10hz_csv.py
# This script converts raw 2000Hz CSVs into a single 10Hz feature-engineered CSV
# for Exploratory Data Analysis (EDA).

import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- ⚙️ CONFIGURATION ---
class Config:
    # We will define the file list in the main() function.
    OUTPUT_CSV_PATH = "10hz_feature_data.csv" # The SINGLE output CSV for analysis
    
    # Signal Processing
    FS = 2000.0 # 2000 Hz
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    BAND_LOW = 20.0
    BAND_HIGH = 450.0
    BUTTER_ORDER = 4
    
    # Feature Engineering (Downsampling)
    WINDOW_MS = 100 # 100ms window
    WINDOW_SAMPLES = int(FS * (WINDOW_MS / 1000.0)) # = 200 samples

# --- 🛠️ HELPER FUNCTIONS ---

def design_filters(fs: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the 50Hz notch and 20-450Hz bandpass filters."""
    b_notch, a_notch = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)
    nyq = 0.5 * fs
    low = Config.BAND_LOW / nyq
    high = Config.BAND_HIGH / nyq
    b_band, a_band = signal.butter(Config.BUTTER_ORDER, [low, high], btype='band')
    return (b_notch, a_notch), (b_band, a_band)

def calculate_hudgins_features(window: np.ndarray) -> np.ndarray:
    """
    Calculates the 4 Hudgins' features (RMS, WL, ZC, SSC) for a 12-channel window.
    Input window shape: (200, 12)
    Output feature vector shape: (48,)
    """
    n_samples, n_channels = window.shape
    features = []
    
    # 1. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2, axis=0))
    features.append(rms)
    
    # 2. Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    features.append(wl)
    
    # 3. Zero Crossings (ZC)
    zc = np.sum((window[:-1] * window[1:] < 0) & (np.abs(window[:-1] - window[1:]) > 1e-6), axis=0)
    features.append(zc)
    
    # 4. Slope Sign Changes (SSC)
    ssc = np.sum((window[1:-1] - window[:-2]) * (window[1:-1] - window[2:]) > 1e-6, axis=0)
    features.append(ssc)
    
    return np.concatenate(features, axis=0).astype(np.float32)

# --- 1. DATA GATHERING (2000Hz -> 10Hz) ---
def build_10hz_list_of_rows(file_list: List[str]) -> List[Dict]:
    """
    Processes all raw CSVs, extracts features, and returns a list of rows
    (as dictionaries) for the 10Hz data.
    """
    (b_notch, a_notch), (b_band, a_band) = design_filters(Config.FS)
    
    all_rows = [] # List to hold all our new 10Hz rows (as dicts)
    
    for fp in tqdm(file_list, desc="Processing Raw Files (2000Hz -> 10Hz)", unit="file"):
        try:
            # Load all necessary columns
            use_cols = [f"emg_{i}" for i in range(1, 13)] + [f"glove_{i}" for i in range(1, 23)] + ["restimulus"]
            df = pd.read_csv(fp, usecols=lambda c: c in use_cols)
            
            emg_cols = [f"emg_{i}" for i in range(1, 13)]
            glove_cols = [f"glove_{i}" for i in range(1, 23)]
            
            emg_data = df[emg_cols].values
            glove_data = df[glove_cols].values
            labels = df['restimulus'].values.astype(int)
            
            # Filter the raw EMG data
            for i in range(emg_data.shape[1]):
                emg_data[:, i] = signal.filtfilt(b_notch, a_notch, emg_data[:, i])
                emg_data[:, i] = signal.filtfilt(b_band, a_band, emg_data[:, i])
            
            # Find boundaries of "pure" chunks
            boundaries = np.where(np.diff(labels) != 0)[0] + 1
            chunk_starts = np.concatenate(([0], boundaries))
            chunk_stops = np.concatenate((boundaries, [len(labels)]))
            
            # Process each "pure" chunk
            for start, stop in zip(chunk_starts, chunk_stops):
                if start >= stop: continue
                
                chunk_emg = emg_data[start:stop]
                chunk_glove = glove_data[start:stop]
                chunk_label = labels[start] # All labels in this chunk are the same
                
                n_samples = len(chunk_emg)
                # Use non-overlapping windows, discard partial remainder
                n_windows = n_samples // Config.WINDOW_SAMPLES 
                
                if n_windows == 0: continue
                
                # Process all *full* non-overlapping windows in this chunk
                for i in range(n_windows):
                    win_start = i * Config.WINDOW_SAMPLES
                    win_stop = win_start + Config.WINDOW_SAMPLES
                    
                    emg_window = chunk_emg[win_start:win_stop]
                    glove_window = chunk_glove[win_start:win_stop]
                    
                    # Calculate features for the window
                    features_48 = calculate_hudgins_features(emg_window)
                    
                    # Get the *last* glove position as the target
                    target_22 = glove_window[-1, :].astype(np.float32)
                    
                    # Create a dictionary for this 10Hz row
                    row_data = {'restimulus': chunk_label}
                    # Add features
                    for f_idx in range(48):
                        row_data[f'feat_{f_idx}'] = features_48[f_idx]
                    # Add targets
                    for t_idx in range(22):
                        row_data[f'glove_{t_idx}'] = target_22[t_idx]
                        
                    all_rows.append(row_data)
                
        except Exception as e:
            print(f"\n⚠️ Warning: Failed to process file {fp}. Error: {e}")
            continue
            
    print(f"\n... 10Hz data extraction complete. Found {len(all_rows)} total 10Hz samples.")
    return all_rows

# --- MAIN EXECUTION ---
def main():
    print("🚀 STARTING CONVERSION TO 10Hz CSV (PHASE 2a)")
    print("=" * 60)
    
    # ==================== EDIT THIS LIST ====================
    # Paste your 3 file paths here.
    # Use 'r' before the string to handle Windows backslashes '\'.
    FILES_TO_PROCESS = [
        r"C:\stage\data\DB\DB2\E1\S20_E1_A1.csv",
        r"C:\stage\data\DB\DB2\E2\S1_E2_A1.csv",
        r"C:\stage\data\DB\DB3\E2\S7_E2_A1.csv"
    ]
    # ========================================================
    
    print(f"Processing {len(FILES_TO_PROCESS)} manually specified files...")

    # --- STEP 1: Process 2000Hz -> 10Hz ---
    list_of_rows = build_10hz_list_of_rows(FILES_TO_PROCESS)
    
    if not list_of_rows:
        print("❌ FATAL: No 10Hz data could be extracted. Check file paths and data.")
        return

    # --- STEP 2: Convert list of dicts to DataFrame and save ---
    print(f"\n... Converting to DataFrame and saving to {Config.OUTPUT_CSV_PATH}...")
    try:
        final_df = pd.DataFrame(list_of_rows)
        # Use a more efficient format if pyarrow is installed, otherwise CSV
        try:
            import pyarrow
            final_df.to_parquet(Config.OUTPUT_CSV_PATH.replace('.csv', '.parquet'), index=False)
            print("   (Saved as .parquet for speed and size efficiency)")
            Config.OUTPUT_CSV_PATH = Config.OUTPUT_CSV_PATH.replace('.csv', '.parquet')
        except ImportError:
            final_df.to_csv(Config.OUTPUT_CSV_PATH, index=False)
            print("   (Saved as .csv. Consider `pip install pyarrow` for faster saves/loads)")

    except Exception as e:
        print(f"❌ FATAL: Failed to save DataFrame. Error: {e}")
        return

    print("\n" + "=" * 60)
    print(f"🎉 CONVERSION TO 10Hz CSV COMPLETE!")
    print(f"   Total 10Hz samples created: {len(list_of_rows):,}")
    print(f"   Output file: {Config.OUTPUT_CSV_PATH}")
    print("\nNext step: Run the 'Phase 2b' analysis script on this new file.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

