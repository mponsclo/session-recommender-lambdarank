{{ config(materialized='table') }}

with user_profiles as (
    select * from {{ ref('int_user_profiles') }}
),

user_behavior as (
    select
        user_id,
        count(distinct session_id)      as total_sessions,
        count(*)                        as total_interactions,
        sum(is_added_to_cart)           as total_cart_additions,
        count(distinct product_id)      as unique_products_interacted,
        min(interaction_date)           as first_activity_date,
        max(interaction_date)           as last_activity_date
    from {{ ref('stg_interactions_train') }}
    where not is_anonymous
    group by user_id
)

select
    coalesce(up.user_id, ub.user_id)                    as user_id,
    up.primary_country_id,
    up.recency,
    up.frequency,
    up.monetary_value,
    up.all_country_ids,
    up.country_count,
    coalesce(ub.total_sessions, 0)                      as total_sessions,
    coalesce(ub.total_interactions, 0)                   as total_interactions,
    coalesce(ub.total_cart_additions, 0)                 as total_cart_additions,
    coalesce(ub.unique_products_interacted, 0)           as unique_products_interacted,
    ub.first_activity_date,
    ub.last_activity_date,

    -- Overall conversion rate
    case
        when coalesce(ub.total_interactions, 0) = 0 then 0
        else round(ub.total_cart_additions::double / ub.total_interactions * 100, 2)
    end as overall_cart_ratio,

    -- User type classification
    case
        when up.user_id is not null and ub.user_id is not null then 'known_active'
        when up.user_id is not null and ub.user_id is null then 'known_inactive'
        when up.user_id is null and ub.user_id is not null then 'unregistered_active'
        else 'unknown'
    end as user_type

from user_profiles up
full outer join user_behavior ub on up.user_id = ub.user_id
