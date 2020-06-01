# -*- coding: utf-8 -*-
"""
Created on Wed May 27 2020

@author: Advait Hasabnis
"""

import os
import pandas as pd
from pathlib import Path
from google.cloud import bigquery

from target_and_market.config import data_folder

# Uncomment following options for better viewability in IPython console
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 600)

#%% BigQuery client

SERVICE_ACCOUNT = os.environ['GOOGLE_API_KEY']
bqclient = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT)

#%% Users who have at least one session with no session_id
query = """
SELECT
    user_id,
    (SELECT value.int_value FROM UNNEST (event_params)
     WHERE key = 'ga_session_id') AS session_id,
FROM `analytics_157832975.events_202005*`
WHERE event_name = "user_engagement"
GROUP BY session_id, user_id
HAVING session_id IS NULL
"""

user_error_status = pd.DataFrame(bqclient.query(query)
                                 .result()
                                 .to_dataframe()
                                 )
user_error_status.to_csv(data_folder/'user_error_status.csv')


#%% Query user statistics from BigQuery
# Session considered between 10 seconds and 30 mins
query_string_1 = """
SELECT
    t1.user_id,
    t2.sessions,
    t2.last_session,
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
        MAX(last_event) AS last_session,
        SUM(session_time) AS total_eng_time,
        AVG(session_time) AS avg_session_time
    FROM (
        SELECT
            user_id,
            session_id,
            MAX(event_timestamp) AS last_event,
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
        HAVING session_time > 10000 AND session_time < 1800000
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

user_analytics = pd.DataFrame(bqclient.query(query_string_1)
                              .result()
                              .to_dataframe()
                              )
user_analytics.to_csv(data_folder/'user_analytics.csv')



