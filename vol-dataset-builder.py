# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

from feature_functions import Binarizer, return_proba

import requests
import pandas as pd
import numpy as np
import sqlalchemy
import mysql.connector
import matplotlib.pyplot as plt

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar

polygon_api_key = "your polygon.io API key, use 'QUANTGALORE' for 10% off"
calendar = get_calendar("NYSE")

start_date = "2006-01-01"
end_date = (datetime.today() - timedelta(days = 1)).strftime("%Y-%m-%d")

trade_dates = pd.DataFrame({"trade_dates": calendar.schedule(start_date = start_date, end_date = end_date).index.strftime("%Y-%m-%d")})

underlying_ticker = "SPY"

volatility_list = []
times = []

for date in trade_dates["trade_dates"]:
    
    try:

        start_time = datetime.now()
        
        underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
        underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time()) & (underlying.index.time < pd.Timestamp("16:01").time())]
        
        if len(underlying) < 350:
            continue
        
        pre_14_session = underlying[underlying.index.hour < 14].copy()
        pre_14_session["returns"] = abs(pre_14_session["c"].pct_change().cumsum())
        
        post_14_session = underlying[underlying.index.hour >= 14].copy()
        post_14_session["returns"] = abs(post_14_session["c"].pct_change().cumsum())
        
        volatility_dataframe = pd.DataFrame([{"date": pd.to_datetime(date), 
                                              "pre_14_vol": round(pre_14_session["returns"].iloc[-1]*100, 2),
                                              "pre_14_volume": round(pre_14_session["v"].sum()),
                                              "post_14_vol": round(post_14_session["returns"].iloc[-1]*100, 2),
                                              "post_14_volume": round(post_14_session["v"].sum())}])
    
        volatility_list.append(volatility_dataframe)
        end_time = datetime.now()
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(trade_dates["trade_dates"]==date)[0][0]/len(trade_dates.index))*100,2)
        iterations_remaining = len(trade_dates["trade_dates"]) - np.where(trade_dates["trade_dates"]==date)[0][0]
        average_time_to_complete = np.mean(times)
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
    except Exception as error:
        print(error)
        continue
    
volatility_dataset = pd.concat(volatility_list).set_index("date")
volatility_dataset["volatility_change"] = volatility_dataset["post_14_vol"] - volatility_dataset["pre_14_vol"]

# how often volatility was higher
len(volatility_dataset[volatility_dataset["volatility_change"] > 0]) / len(volatility_dataset)

training_dataset = volatility_dataset.copy()
training_dataset["year"] = training_dataset.index.year
training_dataset["month"] = training_dataset.index.month
training_dataset["day"] = training_dataset.index.day

# storing the dataset

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')

# with engine.connect() as conn:
#     result = conn.execute(sqlalchemy.text(f'DROP TABLE vol_dataset'))

training_dataset.to_sql(f"vol_dataset", con = engine, if_exists = "replace")