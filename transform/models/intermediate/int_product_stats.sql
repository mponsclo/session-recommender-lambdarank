{{ config(materialized='table') }}

with interactions as (
    select * from {{ ref('stg_interactions_train') }}
)

select
    product_id,
    count(*)                                                                        as total_interactions,
    count(distinct session_id)                                                      as unique_sessions,
    count(distinct user_id) filter (where not is_anonymous)                         as unique_known_users,
    sum(is_added_to_cart)                                                           as total_cart_additions,
    round(sum(is_added_to_cart)::double / nullif(count(*), 0), 4)                  as cart_addition_rate,
    count(distinct country_id)                                                      as countries_seen_in,
    count(distinct device_type_id)                                                  as device_types_used,
    min(interaction_date)                                                            as first_seen_date,
    max(interaction_date)                                                            as last_seen_date,
    -- Temporal: interactions in last 3 days vs first 3 days of the dataset
    -- Note: dates are specific to the hackathon dataset (2024-06-01 to 2024-06-16)
    count(*) filter (where interaction_date >= '2024-06-14')                        as last_3_days_interactions,
    count(*) filter (where interaction_date <= '2024-06-03')                        as first_3_days_interactions
from interactions
group by product_id
