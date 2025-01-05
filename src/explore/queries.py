import  os
import duckdb
import pandas as pd

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

def get_average_visits_before_adding_to_cart(con, csv_file_path):
  """
  Among the products that were added to the cart at least once, 
  how many times is a product visited before it is added to the cart on average? 
  Give the answer with 2 decimals.
  """
  
  query = f"""
    WITH add_to_cart_events AS (
      SELECT
        session_id,
        user_id,
        partnumber,
        timestamp_local AS add_to_cart_time,
        ROW_NUMBER() OVER (PARTITION BY session_id, user_id, partnumber ORDER BY timestamp_local) AS add_to_cart_event_id
      FROM read_csv_auto('{csv_file_path}')
      WHERE true
        AND add_to_cart = 1
        AND user_id = 324152.0
        AND user_id is not null
        AND partnumber = 23268
    ),

    visits_with_cart_mapping AS (
      SELECT
          v.session_id,
          v.partnumber,
          v.user_id,
          v.timestamp_local AS visit_time,
          c.add_to_cart_time,
          c.add_to_cart_event_id
      FROM read_csv_auto('{csv_file_path}') v
      LEFT JOIN add_to_cart_events c
          ON v.session_id = c.session_id
          AND v.user_id = c.user_id
          AND v.partnumber = c.partnumber
          AND v.timestamp_local < c.add_to_cart_time -- Visits must be before the add-to-cart event
      WHERE true
        AND v.add_to_cart = 0
        AND v.user_id = 324152.0
        AND v.user_id is not null
        AND v.partnumber = 23268
    ),

    pre_cart_visits AS (
        SELECT
            session_id,
            user_id,
            partnumber,
            add_to_cart_event_id,
            COUNT(*) AS pre_cart_visit_count
        FROM visits_with_cart_mapping
        GROUP BY session_id, user_id, partnumber, add_to_cart_event_id
    ),

    base AS (
      SELECT
        AVG(pre_cart_visit_count) AS avg_pre_cart_visits
      FROM pre_cart_visits
      WHERE true
        --AND add_to_cart = 1
        AND user_id = 324152.0
        AND user_id is not null
        AND partnumber = 23268
    )

    SELECT *
    FROM visits_with_cart_mapping
  """

  query2 = f"""
    --SELECT user_id, partnumber, MAX(add_to_cart) AS add_to_cart, COUNT(*) AS visit_count
    SELECT *
    FROM read_csv_auto('{csv_file_path}')
    WHERE true
      --AND add_to_cart = 1
      AND user_id = 324152.0
      AND user_id is not null
      AND partnumber = 23268
    --GROUP BY user_id, partnumber
    --HAVING MAX(add_to_cart) = 1 AND COUNT(*) = 11
    ORDER BY timestamp_local ASC
  """

  result = con.execute(query).fetchdf()
  return result

def get_device_most_frequently_used_for_purchases(con, csv_file_path, csv_file_path2):
  """
  Which device (`device_type`) is most frequently used by users to make purchases (`add_to_cart` = 1) 
  of discounted products (`discount` = 1)?
  """

  query = f"""
    WITH discounted_products AS (
      SELECT partnumber
      FROM read_csv_auto('{csv_file_path}')
      WHERE true
        AND discount = 1
    )

    SELECT
      device_type,
      COUNT(*) AS nb_purchases
    FROM read_csv_auto('{csv_file_path2}')
    WHERE true
      AND add_to_cart = 1
      AND EXISTS (
        SELECT 1
        FROM discounted_products dp
        WHERE dp.partnumber = read_csv_auto.partnumber
      )
    GROUP BY device_type
    ORDER BY nb_purchases DESC
  """
  
  result = con.execute(query).fetchdf()
  return result


# Get the product with the lowest family code with a discount
# result = get_product_with_lowest_family_code_with_discount(con, products_path)
# result = get_user_with_lowest_purchase_frequency(con, users_path)
# result = get_average_visits_before_adding_to_cart(con, train_path)
result = get_device_most_frequently_used_for_purchases(con, products_path, train_path)
print(result)