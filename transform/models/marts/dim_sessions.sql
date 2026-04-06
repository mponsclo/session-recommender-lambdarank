{{ config(materialized='table') }}

with sessions as (
    select * from {{ ref('int_sessions') }}
),

user_dim as (
    select user_id, user_type
    from {{ ref('dim_users') }}
)

select
    s.session_id,
    s.user_id,
    s.is_anonymous,
    s.country_id,
    s.session_start,
    s.session_end,
    s.session_date,
    s.session_duration_seconds,
    s.total_interactions,
    s.unique_products_viewed,
    s.products_added_to_cart,
    s.cart_addition_ratio,
    s.primary_device_type,
    s.primary_page_type,
    s.device_types_used,
    s.page_types_visited,

    -- Temporal features
    dayofweek(s.session_date)           as day_of_week,
    hour(s.session_start)               as hour_of_day,
    dayofweek(s.session_date) in (6, 7) as is_weekend,

    -- User context
    coalesce(u.user_type, 'anonymous')  as user_type

from sessions s
left join user_dim u on s.user_id = u.user_id
