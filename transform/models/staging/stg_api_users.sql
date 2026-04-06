{{ config(materialized='table') }}

with raw_batches as (
    select *
    from read_csv_auto('../data/raw/api_extraction/batch_*.csv')
),

parsed as (
    select
        user_id::bigint as user_id,
        string_split(trim(replace(replace(country, '[', ''), ']', '')), ',') as country_list,
        string_split(trim(replace(replace("R", '[', ''), ']', '')), ',') as r_list,
        string_split(trim(replace(replace("F", '[', ''), ']', '')), ',') as f_list,
        string_split(trim(replace(replace("M", '[', ''), ']', '')), ',') as m_list
    from raw_batches
),

unnested as (
    select
        user_id,
        unnest(country_list)::integer   as country_id,
        unnest(r_list)::integer         as recency,
        unnest(f_list)::integer         as frequency,
        unnest(m_list)::double          as monetary_value
    from parsed
)

select * from unnested
