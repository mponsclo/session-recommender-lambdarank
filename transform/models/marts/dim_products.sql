{{ config(materialized='table') }}

with products as (
    select * from {{ ref('stg_products') }}
),

product_stats as (
    select * from {{ ref('int_product_stats') }}
),

family_stats as (
    select
        family_id,
        round(avg(cart_addition_rate), 4)   as family_avg_cart_rate,
        count(*)                            as family_product_count,
        sum(total_interactions)              as family_total_interactions
    from {{ ref('int_product_stats') }} ps
    join {{ ref('stg_products') }} p on ps.product_id = p.product_id
    group by family_id
)

select
    p.product_id,
    p.has_discount,
    p.color_id,
    p.section_id,
    p.family_id,
    p.embedding_raw,

    -- Interaction stats
    coalesce(ps.total_interactions, 0)      as total_interactions,
    coalesce(ps.total_cart_additions, 0)    as total_cart_additions,
    coalesce(ps.cart_addition_rate, 0)      as cart_addition_rate,
    coalesce(ps.unique_sessions, 0)         as unique_sessions,
    coalesce(ps.unique_known_users, 0)      as unique_known_users,
    coalesce(ps.countries_seen_in, 0)       as countries_seen_in,
    ps.first_seen_date,
    ps.last_seen_date,

    -- Temporal trend
    coalesce(ps.last_3_days_interactions, 0)    as last_3_days_interactions,
    coalesce(ps.first_3_days_interactions, 0)   as first_3_days_interactions,
    case
        when coalesce(ps.first_3_days_interactions, 0) = 0 then null
        else round(ps.last_3_days_interactions::double / ps.first_3_days_interactions, 2)
    end as trend_ratio,

    -- Family context
    coalesce(fs.family_avg_cart_rate, 0)        as family_avg_cart_rate,
    coalesce(fs.family_product_count, 0)        as family_product_count,

    -- Popularity rank within family
    rank() over (
        partition by p.family_id
        order by coalesce(ps.total_interactions, 0) desc
    ) as family_popularity_rank

from products p
left join product_stats ps on p.product_id = ps.product_id
left join family_stats fs on p.family_id = fs.family_id
