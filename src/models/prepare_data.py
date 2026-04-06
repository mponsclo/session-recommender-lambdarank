import os
import pandas as pd
from user_segmentation import normalize_rfm, segment_users

# Directory for saving files
data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
output_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')

# Path to the CSV file
users_path = os.path.join(data_dir, 'users.csv')
users = pd.read_csv(users_path)

def process_users(users: pd.DataFrame, output_dir: str) -> None:
  
  users = normalize_rfm(users)
  users = segment_users(users)

  output_path = os.path.join(output_dir, 'processed_users.csv')
  users = users[['country', 'user_id', 'segment_1', 'segment_2', 'segment_3']]
  users.to_csv(output_path, index=False)

if __name__ == "__main__":
  # Process the users and save the output
  process_users(users, output_dir)