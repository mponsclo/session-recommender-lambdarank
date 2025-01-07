import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

unique_users_train = train['user_id'].unique()
unique_users_test = test['user_id'].unique()
users_not_in_train = test[~test['user_id'].isin(unique_users_train)]
#users_not_in_train.to_csv(os.path.join(data_dir, 'users_not_in_train.csv'), index=False)

# Get user sessions that only have 1 row
single_session_users = users_not_in_train.groupby('session_id').filter(lambda x: len(x) == 1)
single_session_users.to_csv(os.path.join(data_dir, 'new_session_activity_not_in_train.csv'), index=False)
multiple_session_users = users_not_in_train.groupby('session_id').filter(lambda x: len(x) != 1)
multiple_session_users.to_csv(os.path.join(data_dir, 'returning_user_not_in_train.csv'), index=False)

def split_test_set(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Splits user sessions into single and multiple session users, both for users not in the training set and users in the training set.

  Args:
    train (pd.DataFrame): The training dataset containing user sessions.
    test (pd.DataFrame): The test dataset containing user sessions.

  Returns:
    tuple: A tuple containing four DataFrames:
      - single_session_users_not_in_db (pd.DataFrame): Users not in the training set with only one session.
      - multiple_session_users_not_in_db (pd.DataFrame): Users not in the training set with multiple sessions.
      - single_session_users_in_db (pd.DataFrame): Users in the training set with only one session.
      - multiple_session_users_in_db (pd.DataFrame): Users in the training set with multiple sessions.
  """
  unique_users_db = train['user_id'].unique()
  users_not_in_db = test[~test['user_id'].isin(unique_users_db)]

  single_session_users_not_in_db = users_not_in_db.groupby('session_id').filter(lambda x: len(x) == 1)
  multiple_session_users_not_in_db = users_not_in_db.groupby('session_id').filter(lambda x: len(x) != 1)

  users_in_db = test[test['user_id'].isin(unique_users_db)]

  single_session_users_in_db = users_in_db.groupby('session_id').filter(lambda x: len(x) == 1)
  multiple_session_users_in_db = users_in_db.groupby('session_id').filter(lambda x: len(x) != 1)

  return (single_session_users_not_in_db, multiple_session_users_not_in_db, 
      single_session_users_in_db, multiple_session_users_in_db)

single_session_users_not_in_db, multiple_session_users_not_in_db, single_session_users_in_train, multiple_session_users_in_train = split_test_set(train, test)

def predict_single_session_users_not_in_db(data: pd.DataFrame) -> pd.DataFrame:
  """
  Predicts the target variable for users not in the training set with only one session.

  Args:
    data (pd.DataFrame): The dataset containing user sessions with only one row.

  Returns:
    pd.DataFrame: The dataset with the target variable predicted.
  """
  
  aggregated_data = data.groupby('user_id').agg({'session_id': 'first'}).reset_index()
  aggregated_data['products'] = [[11024] * 5] * len(aggregated_data)
  result = aggregated_data[['user_id', 'products']]
  return result

def predict_multiple_session_users_not_in_db(data: pd.DataFrame) -> pd.DataFrame:
  """
  Predicts the target variable for users not in the training set with multiple sessions.

  Args:
    data (pd.DataFrame): The dataset containing user sessions with multiple rows.

  Returns:
    pd.DataFrame: The dataset with the target variable predicted.
  """
  aggregated_data = data.groupby('user_id').agg({'session_id': 'first'}).reset_index()
  aggregated_data['products'] = [[11024] * 5] * len(aggregated_data)
  result = aggregated_data[['user_id', 'products']]
  return result

def predict_single_session_users_in_train(data: pd.DataFrame) -> pd.DataFrame:
  """
  Predicts the target variable for users in the training set with only one session.

  Args:
    data (pd.DataFrame): The dataset containing user sessions with only one row.

  Returns:
    pd.DataFrame: The dataset with the target variable predicted.
  """
  aggregated_data = data.groupby('user_id').agg({'session_id': 'first'}).reset_index()
  aggregated_data['products'] = [[11024] * 5] * len(aggregated_data)
  result = aggregated_data[['user_id', 'products']]
  return result

def predict_multiple_session_users_in_train(data: pd.DataFrame) -> pd.DataFrame:
  """
  Predicts the target variable for users in the training set with multiple sessions.

  Args:
    data (pd.DataFrame): The dataset containing user sessions with multiple rows.

  Returns:
    pd.DataFrame: The dataset with the target variable predicted.
  """
  aggregated_data = data.groupby('user_id').agg({'session_id': 'first'}).reset_index()
  aggregated_data['products'] = [[11024] * 5] * len(aggregated_data)
  result = aggregated_data[['user_id', 'products']]
  return result

single_session_users_not_in_db = predict_single_session_users_not_in_db(single_session_users_not_in_db)
assert single_session_users_not_in_db['user_id'].is_unique, "user_id is not unique"

multiple_session_users_not_in_db = predict_multiple_session_users_not_in_db(multiple_session_users_not_in_db)
assert multiple_session_users_not_in_db['user_id'].is_unique, "user_id is not unique"

single_session_users_in_train = predict_single_session_users_in_train(single_session_users_in_train)
assert single_session_users_in_train['user_id'].is_unique, "user_id is not unique"

multiple_session_users_in_train = predict_multiple_session_users_in_train(multiple_session_users_in_train)
assert multiple_session_users_in_train['user_id'].is_unique, "user_id is not unique"


# Concatenate all the predicted dataframes
all_predictions = pd.concat([
  single_session_users_not_in_db,
  multiple_session_users_not_in_db,
  single_session_users_in_train,
  multiple_session_users_in_train
])

# Ensure user_id is unique
# Keep just one of the duplicate user_id
all_predictions = all_predictions.drop_duplicates(subset='user_id', keep='first')
assert all_predictions['user_id'].is_unique, "user_id is not unique"

# Sort by user_id
all_predictions = all_predictions.sort_values(by='user_id')

# Convert to JSON format with user_id as key and list of products as value
predictions_json = all_predictions.set_index('user_id')['products'].to_json(orient='index')

# Save to a JSON file
output_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
  f.write(predictions_json)