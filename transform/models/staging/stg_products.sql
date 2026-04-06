{{ config(materialized='view') }}

with source as (
    select * from {{ source('raw', 'products') }}
),

cleaned as (
    select
        partnumber::bigint          as product_id,
        discount::integer           as has_discount,
        color_id::integer           as color_id,
        cod_section::integer        as section_id,
        family::integer             as family_id,
        embedding                   as embedding_raw
    from source
)

select * from cleaned
