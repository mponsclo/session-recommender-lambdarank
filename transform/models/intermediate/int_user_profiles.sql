{{ config(materialized='table') }}

with csv_users as (
    select *, 'csv' as source from {{ ref('stg_users') }}
),

api_users as (
    select *, 'api' as source from {{ ref('stg_api_users') }}
),

all_users as (
    select user_id, country_id, recency, frequency, monetary_value, source
    from csv_users
    union all
    select user_id, country_id, recency, frequency, monetary_value, source
    from api_users
),

-- Deduplicate: one primary row per user (prefer CSV, then highest frequency)
deduplicated as (
    select
        *,
        row_number() over (
            partition by user_id
            order by
                case when source = 'csv' then 0 else 1 end,
                frequency desc,
                monetary_value desc
        ) as rn
    from all_users
),

-- Preserve all country associations per user
user_countries as (
    select
        user_id,
        list(distinct country_id order by country_id) as all_country_ids,
        count(distinct country_id) as country_count
    from all_users
    group by user_id
)

select
    d.user_id,
    d.country_id        as primary_country_id,
    d.recency,
    d.frequency,
    d.monetary_value,
    uc.all_country_ids,
    uc.country_count,
    d.source             as primary_source
from deduplicated d
join user_countries uc on d.user_id = uc.user_id
where d.rn = 1
