{% macro safe_cast_user_id(column_name) %}
    case
        when {{ column_name }} is not null then {{ column_name }}::bigint
        else null
    end
{% endmacro %}
