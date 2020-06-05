# -*- coding: utf-8 -*-
"""
Created on Wed May 27 2020

@author: Advait Hasabnis
"""

import os
import pandas as pd
from pathlib import Path
from google.cloud import bigquery

from targetandmarket.config import data_folder

# Uncomment following options for better viewability in IPython console
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 600)

#%% BigQuery client

SERVICE_ACCOUNT = os.environ['GOOGLE_API_KEY']
bqclient = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT)

#%% Users who have at least one session with no session_id
# These users will be removed from the dataset
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

#%% Query user statistics - average session time, total engagement time, last session, first session
# Session considered between 10 seconds and 30 mins
query_string_1 = """
SELECT
    t1.user_id,
    t2.sessions,
    t2.last_session,
    t2.total_time,
    t2.avg_session,
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
        MAX(last_event)/1000000 AS last_session,
        SUM(session_time)/1000 AS total_time,
        AVG(session_time)/1000 AS avg_session
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
        MAX(properties.value.int_value)/1000 AS first_open
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
user_analytics.to_csv(data_folder/'user_analytics_raw.csv')

#%% Engagement time per day for each user
query_string_2 = """
SELECT
    user_id,
    date,
    SUM(session_time) AS day_eng_time,
FROM (
    SELECT
        user_id,
        MIN(event_date) AS date,
        SUM(eng_time)/1000 AS session_time,
    FROM (
        SELECT
            user_id,
            event_date,
            (SELECT value.int_value FROM UNNEST (event_params)
             WHERE key = 'engagement_time_msec') AS eng_time,
            (SELECT value.int_value FROM UNNEST (event_params)
             WHERE key = 'ga_session_id') AS session_id,
        FROM `analytics_157832975.events_202005*`
        WHERE event_name = "user_engagement"
    )
    GROUP BY session_id, user_id
    HAVING session_time > 10 AND session_time < 1800
)
GROUP BY user_id, date
"""
timeseries = pd.DataFrame(bqclient.query(query_string_2)
                          .result()
                          .to_dataframe()
                          )
timeseries.to_csv(data_folder/'timeseries.csv')

#%% Users and geolocation
query_string_3 = """
SELECT DISTINCT user_id, geo.continent, geo.country
FROM `analytics_157832975.events_202005*`
"""

user_geo = pd.DataFrame(bqclient.query(query_string_3)
                                .result()
                                .to_dataframe()
                                )
user_geo = user_geo.drop_duplicates(subset=['user_id'], keep=False)
user_geo.to_csv(data_folder/'user_geo.csv')

#%% Users who tried and purchased pro during May 
query_string_4="""
SELECT user_id, event_name, param.value.string_value AS product_id
FROM `analytics_157832975.events_202005*`,
UNNEST(event_params) AS param
WHERE event_name IN ('in_app_purchase', 'ecommerce_purchase')
AND param.key = 'product_id'
"""
user_purchases = pd.DataFrame(bqclient.query(query_string_4)
                                      .result()
                                      .to_dataframe()
                                      )
user_purchases.to_csv(data_folder/'user_purchases_may.csv')

#%% Users who tried and purchased pro during June to validate model in the future
query_string_5="""
SELECT user_id, event_name, param.value.string_value AS product_id
FROM `analytics_157832975.events_202006*`,
UNNEST(event_params) AS param
WHERE event_name IN ('in_app_purchase', 'ecommerce_purchase')
AND param.key = 'product_id'
"""
user_purchases_total_june = pd.DataFrame(bqclient.query(query_string_5)
                                                 .result()
                                                 .to_dataframe()
                                                 )
user_purchases_total_june.to_csv(data_folder/'user_purchases_total_june.csv')