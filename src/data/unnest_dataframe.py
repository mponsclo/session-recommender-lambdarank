import pandas as pd
import ast
import os
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def unnest_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unnest a dataframe with list-like columns into a flat dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with list-like columns.

    Returns:
        pd.DataFrame: The unnested dataframe with each list element as a separate row.
    """
    # Ensure list-like columns are properly converted from string representation to lists
    for col in df.columns[1:]:  # Exclude 'user_id'
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Unnest the dataframe
    unnested_data = []
    for _, row in df.iterrows():
        max_length = max(len(row[col]) for col in df.columns[1:])  # Find the longest list
        for i in range(max_length):
            new_row = {col: row[col][i] if i < len(row[col]) else None for col in df.columns[1:]}  # Handle shorter lists
            new_row['user_id'] = row['user_id']
            unnested_data.append(new_row)
    
    # Convert unnested data back to a dataframe
    unnested_df = pd.DataFrame(unnested_data)
    
    # Ensure all columns except 'user_id' are numeric
    for col in unnested_df.columns:
        if col != 'user_id':
            unnested_df[col] = pd.to_numeric(unnested_df[col], errors='coerce')
    
    return unnested_df

# Directory path
data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
os.makedirs(data_dir, exist_ok=True)

# Process and concatenate all files
all_dataframes = []
for i in range(1, 14):  # Iterate from batch_1.csv to batch_13.csv
    file_path = os.path.join(data_dir, f"batch_{i}.csv")
    if os.path.exists(file_path):  # Check if the file exists
        logger.info(f"Processing {file_path}...")
        raw_df = pd.read_csv(file_path)  # Read the file
        unnested_df = unnest_dataframe(raw_df)  # Unnest the dataframe
        all_dataframes.append(unnested_df)  # Add to the list
    else:
        logger.warning(f"File {file_path} not found.")

# Concatenate all the dataframes
if all_dataframes:
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)
    # Save the concatenated dataframe to a new CSV file
    output_path = os.path.join(os.path.dirname(__file__), '../../data/processed/unnested_data.csv')
    os.makedirs(output_path, exist_ok=True)
    final_dataframe.to_csv(output_path, index=False)
    logger.info(f"All dataframes concatenated and saved to {output_path}")
else:
    logger.error("No dataframes were processed. Exiting.")