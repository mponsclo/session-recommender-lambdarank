
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def normalize_rfm(users: pd.DataFrame) -> pd.DataFrame:
  """
  Normalize the R, F, and M columns in the users DataFrame using MinMaxScaler.

  Args:
    users (pd.DataFrame): DataFrame containing the R, F, and M columns to be normalized.

  Returns:
    pd.DataFrame: DataFrame with normalized R, F, and M columns.
  """
  scaler = MinMaxScaler()
  users[["R_normalized", "F_normalized", "M_normalized"]] = scaler.fit_transform(users[["R", "F", "M"]])
  return users

def segment_users(users: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
  """
  Segment users into clusters based on normalized R, F, and M columns using KMeans clustering.
  One-hot encode the segment column and concatenate it to the original DataFrame.

  Args:
    users (pd.DataFrame): DataFrame containing the normalized R, F, and M columns.
    n_clusters (int): Number of clusters for KMeans. Default is 3.

  Returns:
    pd.DataFrame: DataFrame with user segments and one-hot encoded segment columns.
  """
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  users["segment"] = kmeans.fit_predict(users[["R_normalized", "F_normalized", "M_normalized"]])
  segment_names = {0: "1", 1: "2", 2: "3"}
  users["segment"] = users["segment"].map(segment_names)
  
  # One-hot encode the segment column
  encoder = OneHotEncoder(sparse_output=False)
  segment_encoded = encoder.fit_transform(users[["segment"]])
  segment_encoded_df = pd.DataFrame(segment_encoded, columns=encoder.get_feature_names_out(["segment"]))
  
  # Concatenate the original dataframe with the one-hot encoded segments
  users = pd.concat([users, segment_encoded_df], axis=1)

  return users