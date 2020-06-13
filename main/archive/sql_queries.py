# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:23:30 2020

@author: advai
"""

# Some old queries

# Query for total sessions and total engagement time per user
# Scraped because counting session_starts gives false account of sessions
# Some sessions start due to some background functions and no engagement
query_string = """
SELECT t1.user_id, t1.sessions, t2.total_eng_time, t3.last_interaction, t3.first_open
FROM (
    SELECT user_id, COUNT(event_name) AS sessions
    FROM `analytics_157832975.events_202005*` AS t1
    WHERE event_name = "session_start"
    GROUP BY user_id
) AS t1
INNER JOIN (
    SELECT user_id, SUM(param.value.int_value) AS total_eng_time
    FROM `analytics_157832975.events_202005*`,
    UNNEST(event_params) AS param
    WHERE param.key = "engagement_time_msec"
    AND event_name = "user_engagement"
    GROUP BY user_id
) AS t2
ON t1.user_id = t2.user_id
INNER JOIN (
    SELECT user_id, MAX(event_timestamp) AS last_interaction, MAX(properties.value.int_value) AS first_open
    FROM `analytics_157832975.events_202005*`,
    UNNEST(user_properties) AS properties
    WHERE properties.key = "first_open_time"
    AND event_name = "user_engagement"
    GROUP BY user_id
) AS t3
ON t2.user_id = t3.user_id
"""

# Query for sessions, eng_time, avg_session_time and last_interaction
# Here sessions more than 1 second included
query_string_1 = """
SELECT
    t1.user_id,
    t2.sessions,
    t2.total_eng_time,
    t2.avg_session_time,
    t3.last_interaction,
    t3.first_open
-- Only select users with app version greater than 3.0.0
FROM (
    SELECT DISTINCT
        user_id
    FROM `analytics_157832975.events_202005*` AS f1
    WHERE event_timestamp = (
        SELECT MIN(event_timestamp)
        FROM `analytics_157832975.events_202005*` AS f2
        WHERE f2.user_id = f1.user_id
    )
    AND app_info.version LIKE '3%'
) AS t1
-- Join user stats with number of sessions and engagement times
-- Sessions with abnormally low engagement times due to background funcs
INNER JOIN (
    SELECT
        user_id,
        COUNT(session_id) AS sessions,
        SUM(session_time) AS total_eng_time,
        AVG(session_time) AS avg_session_time
    FROM (
        SELECT
            user_id,
            session_id,
            SUM(eng_time) AS session_time,
        FROM (
            SELECT
                user_id,
                event_timestamp,
                (SELECT value.int_value FROM UNNEST (event_params)
                 WHERE key = 'engagement_time_msec') AS eng_time,
                (SELECT value.int_value FROM UNNEST (event_params)
                 WHERE key = 'ga_session_id') AS session_id,
            FROM `analytics_157832975.events_202005*`
            WHERE event_name = "user_engagement"
        )
        GROUP BY session_id, user_id
        HAVING session_time > 1000
    )
    GROUP BY user_id
) AS t2
ON t1.user_id = t2.user_id
-- Join recency that is last interaction with app
-- and when user first started using app
INNER JOIN (
    SELECT
        user_id,
        MAX(event_timestamp) AS last_interaction,
        MAX(properties.value.int_value) AS first_open
    FROM `analytics_157832975.events_202005*`,
    UNNEST(user_properties) AS properties
    WHERE properties.key = "first_open_time"
    AND event_name = "user_engagement"
    GROUP BY user_id
) AS t3
ON t1.user_id = t3.user_id
"""

# Visits and engagement times per screen
query_string_3 = """
SELECT
    screen_name,
    COUNT(event_name) AS screen_visits,
    SUM(eng_time) AS screen_eng
FROM (
    SELECT
        event_name,
        (SELECT value.string_value FROM UNNEST (event_params)
         WHERE key = 'firebase_screen') AS screen_name,
        (SELECT value.int_value FROM UNNEST (event_params)
         WHERE key = 'engagement_time_msec') AS eng_time
    FROM `analytics_157832975.events_202005*`
    WHERE event_name = "user_engagement"
)
GROUP BY screen_name
"""