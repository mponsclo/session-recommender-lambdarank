{{ config(materialized='view') }}

with source as (
    select * from {{ source('raw', 'users') }}
),

cleaned as (
    select
        user_id::bigint         as user_id,
        country::integer        as country_id,
        "R"::integer            as recency,
        "F"::integer            as frequency,
        "M"::double             as monetary_value
    from source
)

select * from cleaned
