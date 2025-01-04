import  os
import duckdb
import pandas as pd

# Directory for saving files
data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')


# Path to the CSV file
products_path = os.path.join(data_dir, 'products.csv')

# Create a DuckDB connection
con = duckdb.connect()

def get_product_with_lowest_family_code_with_discount(con, csv_file_path):
  """
    Which product (`partnumber`) with `color_id` equal to 3 
    belongs to the lowest `familiy` code with a `discount`?
  """ 
  query = f"""
    SELECT partnumber, family, color_id, discount
    FROM read_csv_auto('{csv_file_path}')
    WHERE true
      AND color_id = 3
      AND discount = 1
    QUALIFY ROW_NUMBER() OVER (ORDER BY family ASC) = 1
  """
  
  result = con.execute(query).fetchdf()
  return result

# Call the function and print the result
product = get_product_with_lowest_family_code_with_discount(con, products_path)
print(product)