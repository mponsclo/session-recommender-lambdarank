# Data Pipeline (dbt + DuckDB)

All transformations live in [transform/](../transform/) and are managed by dbt against a local DuckDB file ([transform/target/inditex_recommender.duckdb](../transform/)). Four materialization layers, each with a single responsibility.

## Layer overview

```
Raw CSVs ──> Staging (views) ──> Intermediate (tables) ──> Marts (tables) ──> Features (tables) ──> ML
```

Materialization strategy from [dbt_project.yml](../transform/dbt_project.yml):

| Layer | Materialization | Purpose |
|-------|-----------------|---------|
| Staging | view | Type casting, column renaming, split tagging |
| Intermediate | table | Aggregation, deduplication, row-level enrichment |
| Marts | table | Dimension/fact model for analytics reuse |
| Features | table | Wide feature tables for ML handoff |

## Staging ([transform/models/staging/](../transform/models/staging/))

| Model | Source | Purpose |
|-------|--------|---------|
| `stg_interactions_train` | `train.csv` (46.5M rows) | Type cast, column rename, flag anonymous sessions |
| `stg_interactions_test` | `test.csv` (29K rows) | Same cleaning, no cart labels |
| `stg_products` | `products.csv` (43,692 rows) | Normalize discount flag, family, section, color, raw embedding |
| `stg_users` | `users.csv` (577K rows) | User RFM data from CSV |
| `stg_api_users` | batched JSON API dumps | Unnest + flatten the API user payload |

## Intermediate ([transform/models/intermediate/](../transform/models/intermediate/))

| Model | Purpose |
|-------|---------|
| `int_sessions` | Session-level rollup: `session_duration_seconds`, `total_interactions`, `products_added_to_cart`, `cart_addition_ratio` |
| `int_user_product_interactions` | User-product pairs with aggregated view/cart signals |
| `int_user_profiles` | Merge + dedupe user profiles from CSV + API sources |
| `int_product_stats` | Per-product global stats: `total_interactions`, `total_cart_additions`, `cart_addition_rate`, `trend_ratio` (3-day window) |

## Marts ([transform/models/marts/](../transform/models/marts/))

Dimensional model for reuse by both analytics (Task 1) and ML (Task 3):

| Model | Purpose |
|-------|---------|
| `dim_products` | Product dimension: stats + family context (`family_avg_cart_rate`, `family_popularity_rank`, `global_popularity_rank`, `cart_rate_vs_family_avg`, `top_co_viewed_products`) |
| `dim_users` | User dimension: RFM + behavioral stats + user type classification |
| `dim_sessions` | Session dimension: temporal features + user context |
| `fct_interactions` | Union of train + test interactions enriched with product attributes (family, section, color, discount) |

## Features ([transform/models/features/](../transform/models/features/))

The handoff layer — wide tables consumed directly by [src/models/predict_model.py](../src/models/predict_model.py):

| Model | Purpose |
|-------|---------|
| `feat_product_popularity` | Popularity + trend: ranks, `trend_ratio`, family averages |
| `feat_session_context` | Per-session: `products_viewed` list, `device_type`, `families_in_session`, `discount_view_ratio`, `is_returning_user` |
| `feat_user_behavior` | Per-user: RFM, totals, preferred family/section/color, `discount_interaction_ratio`, `families_explored`, `days_since_last_activity` |
| `feat_recommendation_input` | Final wide table — 37 columns combining session + user + viewed-product stats |

### `feat_recommendation_input` columns

Session identity: `session_id`, `user_id`, `is_anonymous`, `is_returning_user`, `country_id`

Session context: `session_interaction_count`, `unique_products_in_session`, `products_viewed` (list), `device_type`, `page_type`, `session_date`, `session_duration_seconds`, `families_in_session`, `sections_in_session`, `discount_view_ratio`, `dominant_family_in_session`, `dominant_section_in_session`

User features: `user_type`, `recency`, `frequency`, `monetary_value`, `user_total_sessions`, `user_total_interactions`, `user_total_cart_additions`, `user_overall_cart_ratio`, `user_avg_session_duration`, `user_avg_products_per_session`, `user_preferred_family`, `user_preferred_section`, `user_preferred_color`, `user_discount_affinity`, `user_families_explored`, `user_days_since_last_activity`

Aggregated viewed-product stats: `avg_viewed_product_cart_rate`, `avg_viewed_product_popularity`, `avg_viewed_product_rank`, `has_viewed_discounted_product`

## Running the pipeline

```bash
make dbt-build
# Equivalent to:
cd transform && dbt run && dbt test
```

Requires the raw CSVs at `data/raw/` (proprietary, not included — see [README Data Notice](../README.md#data-notice)).
