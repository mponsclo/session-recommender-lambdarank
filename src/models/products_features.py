import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import os

# Directory for saving files
data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')

# Path to the CSV file
train_path = os.path.join(data_dir, 'train.csv')

# Load product data
products_file =  os.path.join(data_dir,"products.pkl")
with open(products_file, "rb") as file:
    products = pickle.load(file)  # Assuming products is a DataFrame

# Load train dataset
train = pd.read_csv(train_path)

# ----- 1. Calculate Product Similarity -----
# Extract product embeddings and ensure they have the same length
product_embeddings = products["embedding"].apply(lambda x: x if isinstance(x, list) and len(x) == len(products["embedding"].iloc[0]) else [0]*len(products["embedding"].iloc[0])).tolist()
product_ids = products["partnumber"].tolist()

# Compute cosine similarity between product embeddings
similarity_matrix = cosine_similarity(product_embeddings)

# Store similarity scores in a DataFrame
similarity_df = pd.DataFrame(
  similarity_matrix, index=product_ids, columns=product_ids
)

# Extract top 5 similar products for each product
def get_top_similar(row, top_n=5):
  # Sort similarity scores in descending order, exclude the product itself
  similar_products = row.sort_values(ascending=False).iloc[1:top_n+1]
  return list(similar_products.index)

# Apply function to each row
similarity_df["top_5_similar"] = similarity_df.apply(get_top_similar, axis=1)

# ----- 2. Statistics from Train -----
# Popularity (frequency of interaction)
product_popularity = train.groupby("partnumber").size().reset_index(name="frequency")

# Add-to-cart ratio
add_to_cart_stats = train.groupby("partnumber")["add_to_cart"].mean().reset_index(name="add_to_cart_ratio")

# Merge stats with products data
products = products.merge(product_popularity, on="partnumber", how="left").fillna(0)
products = products.merge(add_to_cart_stats, on="partnumber", how="left").fillna(0)

# ----- 3. Encode Categorical Features -----
# One-hot encode 'color_id', 'cod_section', 'family'
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(products[["color_id", "cod_section", "family"]])

# Add encoded features back to the DataFrame
encoded_columns = encoder.get_feature_names_out(["color_id", "cod_section", "family"])
encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)
products = pd.concat([products.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Drop original categorical columns (optional)
products = products.drop(columns=["color_id", "cod_section", "family"])

# Display processed product data
print("Processed product data:\n", products.head())
