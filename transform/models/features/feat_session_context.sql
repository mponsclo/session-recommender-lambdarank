{{ config(materialized='table') }}

with test_interactions as (
    select * from {{ ref('stg_interactions_test') }}
),

train_users as (
    select distinct user_id
    from {{ ref('stg_interactions_train') }}
    where user_id is not null
),

session_features as (
    select
        t.session_id,
        t.user_id,
        t.is_anonymous,
        t.country_id,
        count(*)                            as session_interaction_count,
        count(distinct t.product_id)        as unique_products_in_session,
        list(distinct t.product_id)         as products_viewed,
        mode(t.device_type_id)              as device_type,
        mode(t.page_type_id)               as page_type,
        min(t.interaction_date)             as session_date,
        epoch(
            max(t.interaction_timestamp) - min(t.interaction_timestamp)
        )                                   as session_duration_seconds,

        -- Whether user exists in training data
        max(case when tu.user_id is not null then 1 else 0 end)::boolean as is_returning_user,

        -- Product diversity within session
        count(distinct p.family_id)         as families_in_session,
        count(distinct p.section_id)        as sections_in_session,

        -- Discount interest signal
        round(
            count(*) filter (where p.has_discount = 1)::double / nullif(count(*), 0), 4
        ) as discount_view_ratio

    from test_interactions t
    left join train_users tu on t.user_id = tu.user_id
    left join {{ ref('stg_products') }} p on t.product_id = p.product_id
    group by t.session_id, t.user_id, t.is_anonymous, t.country_id
)

select * from session_features
