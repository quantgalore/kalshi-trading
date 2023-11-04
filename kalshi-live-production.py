# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

from feature_functions import Binarizer, return_proba, round_to_multiple

import requests
import pandas as pd
import numpy as np
import sqlalchemy
import mysql.connector
import matplotlib.pyplot as plt
import pytz
import kalshi_python
import uuid

from kalshi_python.models import *
from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestRegressor
from pandas_market_calendars import get_calendar

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
polygon_api_key = "your polygon.io API key, use 'QUANTGALORE' for 10% off"
calendar = get_calendar("NYSE")

vol_dataset = pd.read_sql(sql = "vol_dataset", con = engine).set_index("date")
features = ["year", "month", "day", "pre_14_volume", "pre_14_vol"]
target = "volatility_change"

date = datetime.now(tz=pytz.timezone("America/New_York")).date()

training_dataset = vol_dataset[vol_dataset.index < date.strftime('%Y-%m-%d')].copy()

#
X_Regression = training_dataset[features].values
Y_Regression = training_dataset[target].values

RandomForest_Regression_Model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None).fit(X_Regression, Y_Regression)
#

underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{date.strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time()) & (underlying.index.time < pd.Timestamp("16:01").time())].add_prefix("spy_")
underlying["year"] = underlying.index.year
underlying["month"] = underlying.index.month
underlying["day"] = underlying.index.day

spx_index = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/I:SPX/range/1/second/{date.strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
spx_index.index = pd.to_datetime(spx_index.index, unit = "ms", utc = True).tz_convert("America/New_York")
spx_index = spx_index[(spx_index.index.time >= pd.Timestamp("14:00").time()) & (spx_index.index.time < pd.Timestamp("16:01").time())]

pre_14_session = underlying[underlying.index.hour < 14].copy()
pre_14_session["returns"] = abs(pre_14_session["spy_c"].pct_change().cumsum())

production_data = pd.DataFrame([{"year": pre_14_session["year"].iloc[-1], "month": pre_14_session["month"].iloc[-1],
                                  "day": pre_14_session["day"].iloc[-1],
                                  "pre_14_volume": round(pre_14_session["spy_v"].sum()),
                                "pre_14_vol": round(pre_14_session["returns"].iloc[-1]*100, 2)}])

session_vol = production_data["pre_14_vol"].iloc[0]

X_prod = production_data[features].values

# regression

random_forest_regression_prediction = RandomForest_Regression_Model.predict(X_prod)[0]
regression_expected_vol = session_vol + random_forest_regression_prediction

last_price = spx_index["c"].iloc[-1]

spx_bounds = round((last_price - (last_price * abs(regression_expected_vol/100))),2), round(last_price * (1+abs(regression_expected_vol/100)), 2)
   
# get contract

config = kalshi_python.Configuration()
kalshi_api = kalshi_python.ApiInstance(email='kalshi-login@email.com',password='kalshi-password',configuration=config)

kalshi_event_string = "INXD-" + date.strftime("%y%b%d").upper()

markets = kalshi_api.get_event(event_ticker=kalshi_event_string).markets

option_list = []

for market in markets:
    
    market_ticker = market.ticker
    
    market_info = pd.json_normalize(kalshi_api.get_market(ticker=market_ticker).to_dict()['market'])
    
    strike_range = market_info["subtitle"].iloc[0].replace(',', '').split(' to ')
    
    if "or" in strike_range[0]:
        
        if "above" in strike_range[0]:
        
            strike_range = market_info["subtitle"].iloc[0].replace(',', '').split(' or ')
    
            floor = float(strike_range[0])
            cap = floor + 24.99
            
        elif "below" in strike_range[0]:
            
            strike_range = market_info["subtitle"].iloc[0].replace(',', '').split(' or ')
    
            cap = float(strike_range[0])        
            floor = cap - 24.99
            
    else:
            
        floor = float(strike_range[0])
        cap = float(strike_range[1])

    option_dataframe = market_info.copy().reset_index()
    option_dataframe["floor"] = floor
    option_dataframe["cap"] = cap
    option_dataframe["ticker"] = market_ticker
    
    option_dataframe = option_dataframe[["no_bid", "no_ask", "floor", "cap", "yes_bid", "yes_ask", "ticker"]].copy()
    option_list.append(option_dataframe)

option_chain = pd.concat(option_list)
    
option_chain["spx_floor"] = spx_bounds[0]
option_chain["price"] = last_price
option_chain["spx_cap"] = spx_bounds[1]

# the strikes which best represent the expected price move

option_chain["floor_distance"] = abs(option_chain["floor"] - option_chain["spx_floor"])
option_chain["cap_distance"] = abs(option_chain["cap"] - option_chain["spx_cap"])
option_chain["distance"] = option_chain["floor_distance"] + option_chain["cap_distance"]

chosen_option = option_chain.nsmallest(1, "distance")