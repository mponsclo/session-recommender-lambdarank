{{ config(materialized='table') }}

with product_dim as (
    select * from {{ ref('dim_products') }}
),

-- Co-viewed products: most common other products viewed in same session
session_co_views as (
    select
        a.product_id as product_id,
        b.product_id as co_viewed_product_id,
        count(distinct a.session_id) as co_view_sessions
    from {{ ref('stg_interactions_train') }} a
    join {{ ref('stg_interactions_train') }} b
        on a.session_id = b.session_id
        and a.product_id != b.product_id
    -- Limit to products with reasonable interaction counts to keep this tractable
    where a.product_id in (
        select product_id from {{ ref('int_product_stats') }}
        where total_interactions >= 10
    )
    group by a.product_id, b.product_id
    qualify row_number() over (
        partition by a.product_id
        order by count(distinct a.session_id) desc
    ) <= 10
)

select
    pd.product_id,
    pd.has_discount,
    pd.color_id,
    pd.section_id,
    pd.family_id,
    pd.total_interactions,
    pd.total_cart_additions,
    pd.cart_addition_rate,
    pd.unique_sessions,
    pd.unique_known_users,
    pd.countries_seen_in,
    pd.trend_ratio,
    pd.family_avg_cart_rate,
    pd.family_product_count,
    pd.family_popularity_rank,

    -- Global popularity rank
    rank() over (order by pd.total_interactions desc) as global_popularity_rank,

    -- Cart rate relative to family average
    case
        when pd.family_avg_cart_rate = 0 then null
        else round(pd.cart_addition_rate / pd.family_avg_cart_rate, 2)
    end as cart_rate_vs_family_avg,

    -- Top co-viewed products (as list)
    (
        select list(co_viewed_product_id order by co_view_sessions desc)
        from session_co_views cv
        where cv.product_id = pd.product_id
    ) as top_co_viewed_products

from product_dim pd
