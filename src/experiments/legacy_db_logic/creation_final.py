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