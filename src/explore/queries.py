import os
import json
import duckdb

# Connect to dbt DuckDB pipeline
DB_PATH = os.path.join(os.path.dirname(__file__), '../../transform/target/inditex_recommender.duckdb')
con = duckdb.connect(DB_PATH, read_only=True)


def get_query_1(con):
    """
    Q1: Which product (partnumber) with color_id equal to 3 belongs to
    the lowest family code with a discount?
    """
    return con.sql('''
        SELECT product_id as partnumber
        FROM staging.stg_products
        WHERE color_id = 3 AND has_discount = 1
        ORDER BY family_id ASC
        LIMIT 1
    ''').fetchone()[0]


def get_query_2(con):
    """
    Q2: In the country where most users have made purchases totaling less than 500 (M),
    which is the user who has the lowest purchase frequency (F), the most recent purchase
    (highest R) and the lowest user_id? Sort priority: F ASC, R DESC, user_id ASC.
    """
    return con.sql('''
        WITH country_low_m AS (
            SELECT country_id
            FROM staging.stg_users
            WHERE monetary_value < 500
            GROUP BY country_id
            ORDER BY COUNT(*) DESC
            LIMIT 1
        )
        SELECT user_id
        FROM staging.stg_users
        WHERE country_id = (SELECT country_id FROM country_low_m)
        ORDER BY frequency ASC, recency DESC, user_id ASC
        LIMIT 1
    ''').fetchone()[0]


def get_query_3(con):
    """
    Q3: Among the products that were added to the cart at least once,
    how many times is a product visited before it is added to the cart on average?
    (2 decimal places)
    """
    return con.sql('''
        WITH first_cart AS (
            SELECT product_id, MIN(interaction_timestamp) as first_cart_ts
            FROM staging.stg_interactions_train
            WHERE is_added_to_cart = 1
            GROUP BY product_id
        ),
        views_before AS (
            SELECT fc.product_id, COUNT(t.product_id) as view_count
            FROM first_cart fc
            LEFT JOIN staging.stg_interactions_train t
                ON fc.product_id = t.product_id
                AND t.interaction_timestamp < fc.first_cart_ts
                AND t.is_added_to_cart = 0
            GROUP BY fc.product_id
        )
        SELECT ROUND(AVG(view_count), 2) FROM views_before
    ''').fetchone()[0]


def get_query_4(con):
    """
    Q4: Which device (device_type) is most frequently used by users to make purchases
    (add_to_cart = 1) of discounted products (discount = 1)?
    """
    return con.sql('''
        SELECT t.device_type_id as device_type
        FROM staging.stg_interactions_train t
        INNER JOIN staging.stg_products p ON t.product_id = p.product_id
        WHERE t.is_added_to_cart = 1 AND p.has_discount = 1
        GROUP BY t.device_type_id
        ORDER BY COUNT(*) DESC
        LIMIT 1
    ''').fetchone()[0]


def get_query_5(con):
    """
    Q5: Among users with purchase frequency (F) in the top 3 within their purchase country,
    who has interacted with the most products (partnumber) in sessions conducted from
    device_type = 3?
    """
    return con.sql('''
        WITH ranked_users AS (
            SELECT user_id,
                DENSE_RANK() OVER (PARTITION BY country_id ORDER BY frequency DESC) as rank
            FROM staging.stg_users
        ),
        top3 AS (SELECT DISTINCT user_id FROM ranked_users WHERE rank <= 3)
        SELECT t.user_id
        FROM staging.stg_interactions_train t
        INNER JOIN top3 u ON t.user_id = u.user_id
        WHERE t.device_type_id = 3
        GROUP BY t.user_id
        ORDER BY COUNT(DISTINCT t.product_id) DESC
        LIMIT 1
    ''').fetchone()[0]


def get_query_6(con):
    """
    Q6: For interactions that occurred outside the user's country of residence,
    how many unique family identifiers are there? Take into account any registered
    country for each user, as there may be more than one country per user.
    """
    return con.sql('''
        WITH user_countries AS (
            SELECT user_id, country_id FROM staging.stg_users
        ),
        outside_products AS (
            SELECT DISTINCT t.product_id
            FROM staging.stg_interactions_train t
            LEFT JOIN user_countries uc
                ON t.user_id = uc.user_id AND t.country_id = uc.country_id
            WHERE t.user_id IS NOT NULL
              AND uc.country_id IS NULL
        )
        SELECT COUNT(DISTINCT p.family_id)
        FROM outside_products op
        INNER JOIN staging.stg_products p ON op.product_id = p.product_id
    ''').fetchone()[0]


def get_query_7(con):
    """
    Q7: Among interactions from the first 7 days of June, which is the most frequent
    page type where each family is added to the cart?
    Format: {family: pagetype}. Ties broken by smallest pagetype.
    """
    df = con.sql('''
        WITH june_cart AS (
            SELECT p.family_id, t.page_type_id, COUNT(*) as cnt
            FROM staging.stg_interactions_train t
            INNER JOIN staging.stg_products p ON t.product_id = p.product_id
            WHERE t.is_added_to_cart = 1
              AND t.interaction_date >= DATE '2024-06-01'
              AND t.interaction_date <= DATE '2024-06-07'
              AND t.page_type_id IS NOT NULL
            GROUP BY p.family_id, t.page_type_id
        )
        SELECT family_id, page_type_id
        FROM june_cart
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY family_id ORDER BY cnt DESC, page_type_id ASC
        ) = 1
        ORDER BY family_id
    ''').df()
    return {str(int(r['family_id'])): int(r['page_type_id']) for _, r in df.iterrows()}


def generate_predictions(con, output_path):
    """Generate predictions_1.json with all query answers."""
    predictions = {
        "target": {
            "query_1": {"partnumber": int(get_query_1(con))},
            "query_2": {"user_id": int(get_query_2(con))},
            "query_3": {"average_previous_visits": float(get_query_3(con))},
            "query_4": {"device_type": int(get_query_4(con))},
            "query_5": {"user_id": int(get_query_5(con))},
            "query_6": {"unique_families": int(get_query_6(con))},
            "query_7": get_query_7(con),
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    return predictions


if __name__ == '__main__':
    output_path = os.path.join(os.path.dirname(__file__), '../../predictions/predictions_1.json')
    results = generate_predictions(con, output_path)
    for key, value in results['target'].items():
        print(f"{key}: {value}")
    print(f"\nSaved to {output_path}")
    con.close()
