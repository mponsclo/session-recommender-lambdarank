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

# QUERY 1
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

# QUERY 2
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

# QUERY 3
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

# QUERY 4
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

# QUERY 5
def get_user_with_most_interactions_in_sessions_from_device(con, csv_file_path, csv_file_path2):
  """
  Among users with purchase frequency (`F`) in the top 3 within their purchase country, 
  who has interacted with the most products (`partnumber`) in sessions conducted from a 
  device with identifier 3 (`device_type` = 3)?
  """
  
  query = f"""
    WITH top_users_by_country AS (
      SELECT user_id
      FROM read_csv_auto('{csv_file_path}')
      QUALIFY true
        AND ROW_NUMBER() OVER (PARTITION BY country ORDER BY F DESC) <= 3
    )

    SELECT
      u.user_id
      , COUNT(DISTINCT s.partnumber) AS interaction_count
    FROM top_users_by_country u
    INNER JOIN read_csv_auto('{csv_file_path2}') s ON u.user_id = s.user_id
    WHERE true
      AND s.device_type = 3
    GROUP BY u.user_id
    ORDER BY interaction_count DESC
    LIMIT 1
    """
  
  result = con.execute(query).fetchdf()
  return result

# QUERY 6
def get_unique_family_identifiers_outside_user_country(con, csv_file_path, csv_file_path2, csv_file_path3):
  """
  For interactions that occurred outside the user's country of residence, how many 
  unique family identifiers are there? Take into account any registered country for 
  each user, as there may be more than one country per user.
  """

  query = f"""
    WITH user_countries AS (
      SELECT 
        user_id
        , country
      FROM read_csv_auto('{csv_file_path}')
    ),
    
    product_interactions AS (
      SELECT
        DISTINCT partnumber
      FROM read_csv_auto('{csv_file_path2}') s
      LEFT JOIN user_countries u ON s.user_id = u.user_id AND s.country = u.country
      WHERE true
        AND s.user_id IS NOT NULL
        AND u.country IS NULL
    )

    SELECT
      COUNT(DISTINCT family) AS unique_family_count
    FROM read_csv_auto('{csv_file_path3}')
    INNER JOIN product_interactions pi ON read_csv_auto.partnumber = pi.partnumber
  """

  result = con.execute(query).fetchdf()
  return result

# QUERY 7


## Execute the queries
# result = get_product_with_lowest_family_code_with_discount(con, products_path)
# result = get_user_with_lowest_purchase_frequency(con, users_path)
# result = get_average_visits_before_adding_to_cart(con, train_path) # TODO: Fix the query
# result = get_device_most_frequently_used_for_purchases(con, products_path, train_path)
# result = get_user_with_most_interactions_in_sessions_from_device(con, users_path, train_path)
result = get_unique_family_identifiers_outside_user_country(con, users_path, train_path, products_path)
print(result)