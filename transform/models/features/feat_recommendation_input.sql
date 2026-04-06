{{ config(materialized='table') }}

-- Final wide table joining all features per test session
-- This is the handoff point between dbt and the Python ML code

with session_ctx as (
    select * from {{ ref('feat_session_context') }}
),

user_feat as (
    select * from {{ ref('feat_user_behavior') }}
),

-- For each product viewed in a test session, get its popularity features
session_product_stats as (
    select
        t.session_id,
        -- Aggregate product-level stats across all products viewed in the session
        avg(pp.cart_addition_rate)           as avg_viewed_product_cart_rate,
        avg(pp.total_interactions)           as avg_viewed_product_popularity,
        avg(pp.global_popularity_rank)       as avg_viewed_product_rank,
        max(pp.has_discount)                 as has_viewed_discounted_product,
        mode(pp.family_id)                   as dominant_family_in_session,
        mode(pp.section_id)                  as dominant_section_in_session
    from {{ ref('stg_interactions_test') }} t
    join {{ ref('feat_product_popularity') }} pp on t.product_id = pp.product_id
    group by t.session_id
)

select
    -- Session identification
    sc.session_id,
    sc.user_id,
    sc.is_anonymous,
    sc.is_returning_user,
    sc.country_id,

    -- Session context features
    sc.session_interaction_count,
    sc.unique_products_in_session,
    sc.products_viewed,
    sc.device_type,
    sc.page_type,
    sc.session_date,
    sc.session_duration_seconds,
    sc.families_in_session,
    sc.sections_in_session,
    sc.discount_view_ratio,

    -- User behavior features (NULL for anonymous/cold-start users)
    uf.user_type,
    uf.recency,
    uf.frequency,
    uf.monetary_value,
    uf.total_sessions                       as user_total_sessions,
    uf.total_interactions                    as user_total_interactions,
    uf.total_cart_additions                  as user_total_cart_additions,
    uf.overall_cart_ratio                    as user_overall_cart_ratio,
    uf.avg_session_duration                  as user_avg_session_duration,
    uf.avg_products_per_session             as user_avg_products_per_session,
    uf.preferred_family_id                   as user_preferred_family,
    uf.preferred_section_id                  as user_preferred_section,
    uf.preferred_color_id                    as user_preferred_color,
    uf.discount_interaction_ratio            as user_discount_affinity,
    uf.families_explored                     as user_families_explored,
    uf.days_since_last_activity             as user_days_since_last_activity,

    -- Session product context
    sps.avg_viewed_product_cart_rate,
    sps.avg_viewed_product_popularity,
    sps.avg_viewed_product_rank,
    sps.has_viewed_discounted_product,
    sps.dominant_family_in_session,
    sps.dominant_section_in_session

from session_ctx sc
left join user_feat uf on sc.user_id = uf.user_id
left join session_product_stats sps on sc.session_id = sps.session_id
