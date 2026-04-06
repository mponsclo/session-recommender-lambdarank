{{ config(materialized='table') }}

with train as (
    select * from {{ ref('stg_interactions_train') }}
),

test as (
    select * from {{ ref('stg_interactions_test') }}
),

unioned as (
    select * from train
    union all
    select * from test
)

select
    u.session_id,
    u.interaction_date,
    u.interaction_timestamp,
    u.is_added_to_cart,
    u.user_id,
    u.is_anonymous,
    u.country_id,
    u.product_id,
    u.device_type_id,
    u.page_type_id,
    u.data_split,

    -- Product context
    p.has_discount,
    p.family_id,
    p.section_id,
    p.color_id

from unioned u
left join {{ ref('stg_products') }} p on u.product_id = p.product_id
