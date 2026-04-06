import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import logging
import sys
from itertools import islice

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_user_data(user_id: str, timeout: int = 10, error_log_file: str = "error_log.json") -> dict | None:
    """
    Fetch user data from the API.

    Args:
        user_id (str): The ID of the user to fetch data for.
        timeout (int, optional): The timeout for the request in seconds. Defaults to 10.
        error_log_file (str, optional): The file to log errors to. Defaults to "error_log.json".

    Returns:
        dict | None: A dictionary containing user data if successful, None otherwise.
    """
    url_user = f"https://zara-boost-hackathon.nuwe.io/users/{user_id}"
    try:
        response = requests.get(url_user, timeout=timeout)
        response.raise_for_status()
        user_data = response.json()
        return {
            'user_id': user_data['user_id'],
            'country': user_data['values']['country'],
            'R': user_data['values']['R'],
            'F': user_data['values']['F'],
            'M': user_data['values']['M']
        }
    except requests.exceptions.RequestException as e:
        error_data = {'user_id': user_id, 'error': str(e)}
        with open(error_log_file, "a") as log_file:
            log_file.write(json.dumps(error_data) + "\n")
        logger.error(f"Error fetching data for user {user_id}: {e}")
        return None

def process_batch(
    user_list: list[str], 
    batch_number: int, 
    max_workers: int = 10, 
    timeout: int = 10, 
    error_log_file: str = "error_log.json"
) -> list[dict]:
    """
    Process a batch of users by fetching their data concurrently.

    Args:
        user_list (list[str]): List of user IDs to process.
        batch_number (int): The batch number being processed.
        max_workers (int, optional): The maximum number of threads to use. Defaults to 10.
        timeout (int, optional): The timeout for each request in seconds. Defaults to 10.
        error_log_file (str, optional): The file to log errors to. Defaults to "error_log.json".

    Returns:
        list[dict]: A list of dictionaries containing user data.
    """
    all_users_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_user = {executor.submit(fetch_user_data, user, timeout, error_log_file): user for user in user_list}
        for cnt, future in enumerate(as_completed(future_to_user), 1):
            try:
                user_data = future.result()
                if user_data:
                    all_users_data.append(user_data)
                percentage = (cnt / len(user_list)) * 100
                # Refresh the line in console
                sys.stdout.write(f"\rBatch {batch_number}: Processed {cnt}/{len(user_list)} users ({percentage:.2f}%)")
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
    sys.stdout.write("\n")  # Move to the next line after the batch
    return all_users_data

def batched(iterable: list, batch_size: int) -> iter:
    """
    Yield successive batches from an iterable.

    Args:
        iterable (list): The list to be divided into batches.
        batch_size (int): The size of each batch.

    Yields:
        iter: An iterator over batches of the specified size.
    """
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

def fetch_all_users(
    user_list: list[str], 
    output_dir: str, 
    batch_size: int = 100, 
    max_workers: int = 10, 
    timeout: int = 10, 
    error_log_file: str = "error_log.json"
) -> None:
    """
    Fetch data for all users and save it in batches.

    Args:
        user_list (list[str]): List of user IDs to fetch data for.
        output_dir (str): Directory to save the output files.
        batch_size (int, optional): Number of users to process in each batch. Defaults to 100.
        max_workers (int, optional): Maximum number of threads to use for concurrent processing. Defaults to 10.
        timeout (int, optional): Timeout for each request in seconds. Defaults to 10.
        error_log_file (str, optional): File to log errors to. Defaults to "error_log.json".

    Returns:
        None
    """
    for batch_num, batch in enumerate(batched(user_list, batch_size), 1):
        logger.info(f"Processing batch {batch_num} with {len(batch)} users...")
        batch_data = process_batch(batch, batch_num, max_workers=max_workers, timeout=timeout, error_log_file=error_log_file)
        if batch_data:
            batch_df = pd.DataFrame(batch_data)
            output_file = os.path.join(output_dir, f"batch_{batch_num}.csv")
            batch_df.to_csv(output_file, index=False)
            logger.info(f"Batch {batch_num} saved to {output_file}.")
        else:
            logger.warning(f"Batch {batch_num} contained no valid data.")