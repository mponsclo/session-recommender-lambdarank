{{ config(materialized='table') }}

with interactions as (
    select
        *,
        sum(is_added_to_cart) over (
            partition by user_id, product_id
            order by interaction_timestamp
            rows between unbounded preceding and current row
        ) as cumulative_cart_adds
    from {{ ref('stg_interactions_train') }}
    where not is_anonymous
)

select
    user_id,
    product_id,
    count(*)                                                        as view_count,
    sum(is_added_to_cart)                                           as add_to_cart_count,
    min(interaction_timestamp)                                      as first_interaction,
    max(interaction_timestamp)                                      as last_interaction,
    count(distinct session_id)                                      as sessions_with_product,
    count(distinct interaction_date)                                 as days_with_product,
    -- Views before the first cart addition
    count(*) filter (
        where cumulative_cart_adds = 0
        and is_added_to_cart = 0
    )                                                               as views_before_first_cart
from interactions
group by user_id, product_id
