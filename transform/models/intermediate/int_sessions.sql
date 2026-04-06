{{ config(materialized='table') }}

with interactions as (
    select * from {{ ref('stg_interactions_train') }}
),

session_agg as (
    select
        session_id,
        user_id,
        is_anonymous,
        country_id,
        min(interaction_timestamp)                                          as session_start,
        max(interaction_timestamp)                                          as session_end,
        epoch(max(interaction_timestamp) - min(interaction_timestamp))      as session_duration_seconds,
        count(*)                                                            as total_interactions,
        count(distinct product_id)                                          as unique_products_viewed,
        sum(is_added_to_cart)                                               as products_added_to_cart,
        round(sum(is_added_to_cart)::double / count(*) * 100, 2)           as cart_addition_ratio,
        min(interaction_date)                                               as session_date,
        mode(device_type_id)                                                as primary_device_type,
        mode(page_type_id)                                                  as primary_page_type,
        count(distinct device_type_id)                                      as device_types_used,
        count(distinct page_type_id)                                        as page_types_visited
    from interactions
    group by session_id, user_id, is_anonymous, country_id
)

select * from session_agg
