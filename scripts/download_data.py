# -*- coding: utf-8 -*-
"""
Created on Wed May 27 2020

@author: Advait Hasabnis
"""

import os
import pandas as pd
from pathlib import Path
from google.cloud import bigquery

from segment.config import data_folder

#%% BigQuery Client

SERVICE_ACCOUNT = os.environ['GOOGLE_API_KEY']
bqclient = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT)

#%% Query for total sessions per user
query_string_1 = """
SELECT user_id, COUNT(event_name) AS sessions
FROM `delta-178517.analytics_157832975.events_202005*`
WHERE event_name = "session_start"
GROUP BY user_id
"""

frequency = pd.DataFrame(bqclient.query(query_string_1)
                         .result()
                         .to_dataframe()
                         )
frequency.to_csv(data_folder/'frequency.csv')

#%% Query for total engagement time and avg session time per user
query_string_2 = """
SELECT t1.user_id, t1.total_eng_time, t2.avg_session_time
FROM (
    SELECT user_id, SUM(param.value.int_value) AS total_eng_time
    FROM `analytics_157832975.events_202005*`,
    UNNEST(event_params) AS param
    WHERE param.key = "engagement_time_msec"
    AND event_name = "user_engagement"
    GROUP BY user_id
) AS t1
INNER JOIN (
    SELECT
        user_id,
        AVG(session_time) AS avg_session_time
    FROM (
        SELECT
            user_id,
            SUM(eng_time) AS session_time,
            session_id
        FROM (
            SELECT
                user_id,
                (SELECT value.int_value FROM UNNEST (event_params)
                 WHERE key = 'engagement_time_msec') AS eng_time,
                (SELECT value.int_value FROM UNNEST (event_params)
                 WHERE key = 'ga_session_id') AS session_id
            FROM `analytics_157832975.events_202005*`
            WHERE event_name = "user_engagement"
        )
        GROUP BY session_id, user_id
    )
    GROUP BY user_id
) AS t2
ON t1.user_id = t2.user_id
"""

engagement = pd.DataFrame(bqclient.query(query_string_2)
                          .result()
                          .to_dataframe()
                          )
engagement.to_csv(data_folder/'engagement.csv')

#%% Visits and engagement times per screen
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

screen_stats = pd.DataFrame(bqclient.query(query_string_3)
                          .result()
                          .to_dataframe()
                          )
screen_stats.to_csv(data_folder/'screen_stats.csv')


#%% Query for engagement time per user per screen for top screens