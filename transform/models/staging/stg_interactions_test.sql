{{ config(materialized='view') }}

with source as (
    select * from {{ source('raw', 'test') }}
),

cleaned as (
    select
        session_id::integer                     as session_id,
        date::date                              as interaction_date,
        timestamp_local::timestamp              as interaction_timestamp,
        null::integer                           as is_added_to_cart,
        {{ safe_cast_user_id('user_id') }}      as user_id,
        user_id is null                         as is_anonymous,
        country::integer                        as country_id,
        partnumber::bigint                      as product_id,
        device_type::integer                    as device_type_id,
        pagetype::integer                       as page_type_id,
        'test'                                  as data_split
    from source
)

select * from cleaned
