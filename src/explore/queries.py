import  os
import duckdb
import pandas as pd

# Directory for saving files
data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')


# Path to the CSV file
products_path = os.path.join(data_dir, 'products.csv')
users_path = os.path.join(data_dir, 'users.csv')

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

def get_user_with_lowest_purchase_frequency(con, csv_file_path):
  """
  In the country where most users have made purchases totaling less than 500 (`M`), 
  which is the user who has the lowest purchase frequency (`F`), the most recent purchase (highest `R`) 
  and the lowest `user_id`? Follow the given order of variables as the sorting priority.
  """
  query = f"""
    WITH country_purchases AS (
      SELECT 
        country,
        COUNT(CASE WHEN M < 500 THEN 1 END) AS nb_users_condition,
        COUNT(*) AS nb_users
      FROM read_csv_auto('{csv_file_path}')
      GROUP BY country
    )

    SELECT user_id, R, F, M
    FROM read_csv_auto('{csv_file_path}')
    WHERE country = (
      SELECT country
      FROM country_purchases
      ORDER BY nb_users_condition DESC
      LIMIT 1
    )
    ORDER BY F ASC, R DESC, user_id ASC
    LIMIT 1
  """
  
  result = con.execute(query).fetchdf()
  return result

# Get the product with the lowest family code with a discount
# result = get_product_with_lowest_family_code_with_discount(con, products_path)
result = get_user_with_lowest_purchase_frequency(con, users_path)
print(result)