import  os
import duckdb

# Directory for saving files
data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')

# Path to the CSV file
products_path = os.path.join(data_dir, 'products.csv')
users_path = os.path.join(data_dir, 'users.csv')
train_path = os.path.join(data_dir, 'train.csv')

# Create a DuckDB connection
con = duckdb.connect()

def get_product_with_lowest_family_code_with_discount(con, csv_file_path):
  """
    Which product (`partnumber`) with `color_id` equal to 3 
    belongs to the lowest `familiy` code with a `discount`?
  """ 
  query = f"""
    SELECT *
    FROM read_csv_auto('{csv_file_path}')
    WHERE user_id is not null and session_id = 33052
    --LIMIT 10
  """
  
  result = con.execute(query).fetchdf()
  return result

result = get_product_with_lowest_family_code_with_discount(con, train_path)
print(result)