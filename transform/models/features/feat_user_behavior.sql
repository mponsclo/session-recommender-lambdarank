{{ config(materialized='table') }}

with user_dim as (
    select * from {{ ref('dim_users') }}
),

session_stats as (
    select
        user_id,
        avg(session_duration_seconds)       as avg_session_duration,
        avg(unique_products_viewed)         as avg_products_per_session,
        avg(cart_addition_ratio)            as avg_cart_ratio,
        stddev(session_duration_seconds)    as stddev_session_duration,
        max(session_date)                   as most_recent_session_date,
        median(total_interactions)          as median_interactions_per_session
    from {{ ref('int_sessions') }}
    where user_id is not null
    group by user_id
),

-- Product category preferences
category_prefs as (
    select
        i.user_id,
        mode(p.family_id)       as preferred_family_id,
        mode(p.section_id)      as preferred_section_id,
        mode(p.color_id)        as preferred_color_id,
        -- Discount affinity
        round(
            count(*) filter (where p.has_discount = 1)::double / nullif(count(*), 0), 4
        ) as discount_interaction_ratio,
        -- Diversity: how many distinct families does this user interact with
        count(distinct p.family_id)     as families_explored,
        count(distinct p.color_id)      as colors_explored
    from {{ ref('stg_interactions_train') }} i
    join {{ ref('stg_products') }} p on i.product_id = p.product_id
    where not i.is_anonymous
    group by i.user_id
),

-- Device preferences
device_prefs as (
    select
        user_id,
        mode(device_type_id) as preferred_device_type
    from {{ ref('stg_interactions_train') }}
    where not is_anonymous
    group by user_id
)

select
    u.user_id,
    u.primary_country_id,
    u.user_type,

    -- RFM features
    u.recency,
    u.frequency,
    u.monetary_value,

    -- Activity features
    u.total_sessions,
    u.total_interactions,
    u.total_cart_additions,
    u.unique_products_interacted,
    u.overall_cart_ratio,

    -- Session behavior
    round(coalesce(ss.avg_session_duration, 0), 2)      as avg_session_duration,
    round(coalesce(ss.avg_products_per_session, 0), 2)   as avg_products_per_session,
    round(coalesce(ss.avg_cart_ratio, 0), 2)             as avg_cart_ratio,
    round(coalesce(ss.stddev_session_duration, 0), 2)    as stddev_session_duration,
    coalesce(ss.median_interactions_per_session, 0)      as median_interactions_per_session,

    -- Category preferences
    cp.preferred_family_id,
    cp.preferred_section_id,
    cp.preferred_color_id,
    coalesce(cp.discount_interaction_ratio, 0)           as discount_interaction_ratio,
    coalesce(cp.families_explored, 0)                    as families_explored,
    coalesce(cp.colors_explored, 0)                      as colors_explored,

    -- Device preference
    dp.preferred_device_type,

    -- Recency: days since last activity (relative to dataset end: 2024-06-16)
    date_diff('day', u.last_activity_date, '2024-06-16'::date) as days_since_last_activity

from user_dim u
left join session_stats ss on u.user_id = ss.user_id
left join category_prefs cp on u.user_id = cp.user_id
left join device_prefs dp on u.user_id = dp.user_id
where u.user_id is not null
