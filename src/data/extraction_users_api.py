import os
import requests
import logging
from fetch_user_data import fetch_all_users

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("collection_call.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory for saving files
output_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE = 500
MAX_WORKERS = 10
TIMEOUT = 10
ERROR_LOG_FILE = os.path.join(output_dir, "error_log.json")

response = requests.get("https://zara-boost-hackathon.nuwe.io/users")
user_list = response.json()

open(ERROR_LOG_FILE, "w").close()

logger.info("Starting data collection...")
fetch_all_users(user_list, output_dir, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS, timeout=TIMEOUT, error_log_file=ERROR_LOG_FILE)
logger.info("Data collection complete.")