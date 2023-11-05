# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Local User
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

start_date = "2023-04-26"
end_date = (datetime.today() - timedelta(days = 1)).strftime("%Y-%m-%d")

trade_dates = pd.DataFrame({"trade_dates": calendar.schedule(start_date = start_date, end_date = end_date).index})

config = kalshi_python.Configuration()
kalshi_api = kalshi_python.ApiInstance(email='kalshi-login@email.com',password='kalshi-password',configuration=config)

trades = []

for date in trade_dates["trade_dates"]:
    
    try:
        
        start_time = datetime.now()
        
        underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{date.strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
        underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time())].add_prefix("spy_")
        underlying["year"] = underlying.index.year
        underlying["month"] = underlying.index.month
        underlying["day"] = underlying.index.day
        
        if len(underlying) < 350:
            continue
        
        spx_index = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/I:SPX/range/1/minute/{date.strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        spx_index.index = pd.to_datetime(spx_index.index, unit = "ms", utc = True).tz_convert("America/New_York")
        spx_index = spx_index[(spx_index.index.time >= pd.Timestamp("14:00").time()) & (spx_index.index.time < pd.Timestamp("16:01").time())]
        
        if len(spx_index) < 1:
            continue
        
        pre_14_session = underlying[underlying.index.hour < 14].copy()
        pre_14_session["returns"] = abs(pre_14_session["spy_c"].pct_change().cumsum())
        
        training_dataset = vol_dataset[vol_dataset.index < date].copy()
    
        X_Regression = training_dataset[features].values
        Y_Regression = training_dataset[target].values
        
        production_data = pd.DataFrame([{"year": pre_14_session["year"].iloc[-1], "month": pre_14_session["month"].iloc[-1],
                                          "day": pre_14_session["day"].iloc[-1],
                                          "pre_14_volume": round(pre_14_session["spy_v"].sum()),
                                        "pre_14_vol": round(pre_14_session["returns"].iloc[-1]*100, 2)}])
    
        session_vol = production_data["pre_14_vol"].iloc[0]
    
        X_prod = production_data[features].values
        
        #
        
        RandomForest_Regression_Model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None).fit(X_Regression, Y_Regression)
        
        # regression
    
        random_forest_regression_prediction = RandomForest_Regression_Model.predict(X_prod)[0]
        regression_expected_vol = session_vol + random_forest_regression_prediction
        
        last_price = spx_index["c"].iloc[0]
    
        spx_bounds = round((last_price - (last_price * abs(regression_expected_vol/100))),2), round(last_price * (1+abs(regression_expected_vol/100)), 2)
        
        ################
    
        kalshi_event_string = "INXD-" + date.strftime("%y%b%d").upper()
        trade_open_timestamp = int((date.tz_localize("America/New_York") + timedelta(hours=14)).timestamp())
        trade_close_timestamp = int((date.tz_localize("America/New_York") + timedelta(hours=16)).timestamp())
        
        try:
            markets = kalshi_api.get_event(event_ticker=kalshi_event_string).markets
        except:
            continue
        
        option_list = []
        
        for market in markets:
            
            market_ticker = market.ticker
            
            market_info = pd.json_normalize(kalshi_api.get_market(ticker=market_ticker).to_dict()['market'])
            
            try:
                market_history = pd.json_normalize(kalshi_api.get_market_history(ticker=market_ticker,min_ts = trade_open_timestamp, max_ts = trade_close_timestamp, limit = 10).to_dict()["history"]).set_index("ts")
            except: continue
            market_history.index = pd.to_datetime(market_history.index, unit = "s", utc = True).tz_convert("America/New_York")
    
            trade = market_history.head(1)
            market_info
            
            strike_range =strike_range = market_info["subtitle"].iloc[0].replace(',', '').split(' to ')
            
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
    
            option_dataframe = trade.copy().reset_index()
            option_dataframe["floor"] = floor
            option_dataframe["cap"] = cap
            option_dataframe["ticker"] = market_ticker
            
            option_dataframe = option_dataframe[["no_bid", "no_ask", "floor", "cap", "yes_bid", "yes_ask", "ticker"]].copy()
            option_list.append(option_dataframe)
        
        option_chain = pd.concat(option_list)    
        option_chain["floor_distance"] = abs(option_chain["floor"] - spx_bounds[0])
        option_chain["cap_distance"] = abs(option_chain["cap"] - spx_bounds[1])
        option_chain["distance"] = option_chain["floor_distance"] + option_chain["cap_distance"]
        
        chosen_option = option_chain.nsmallest(1, "distance")
        
        yes_price = chosen_option["yes_ask"].iloc[0]
        
        final_yes_price = np.nan
        
        trade_dataframe = pd.DataFrame([{"date": date, "r_prediction": random_forest_regression_prediction,
                                         "open_price": yes_price, "closing_price": final_yes_price,
                                         "floor": chosen_option["floor"].iloc[0],
                                         "cap": chosen_option["cap"].iloc[0],
                                         "final_spx": spx_index["c"].iloc[-1]}])
        
        trades.append(trade_dataframe)
        
        end_time = datetime.now()
        iteration = round((np.where(trade_dates==date)[0][0]/len(trade_dates))*100,2)
        iterations_remaining = len(trade_dates) - np.where(trade_dates==date)[0][0]
        average_time_to_complete = (end_time - start_time).total_seconds()
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
        
    except Exception as error:
        print(error)
        continue

    #
    
complete_trades = pd.concat(trades).set_index("date")
# un-comment to check if you only take trades <= or >= n
# complete_trades = complete_trades[complete_trades["open_price"] >= 70].copy()
complete_trades['open_price'] = complete_trades['open_price'] / 100
complete_trades['closing_price'] = complete_trades.apply(lambda row: 100 if row['final_spx'] >= row['floor'] and row['final_spx'] <= row['cap'] else 0, axis=1)/100

dollar_trade_amount = 100
starting_balance = 1000

complete_trades['available_contracts'] = round(dollar_trade_amount / (complete_trades['open_price']))
complete_trades['gross_pnl'] = (complete_trades['closing_price'] - complete_trades['open_price']) * complete_trades['available_contracts']

# p(price), c(#of contracts), fee($0.035)
# fee methodology = round up(0.035 x C x P x (1-P))

complete_trades['fees'] = 0.035 * complete_trades['available_contracts'] * complete_trades['open_price'] * (1-complete_trades['open_price'])
complete_trades['fee_percent'] = complete_trades['fees'] / (complete_trades['available_contracts'] * complete_trades['open_price'])
complete_trades['net_pnl'] = complete_trades['gross_pnl'] - complete_trades['fees']

complete_trades["gross_capital"] = starting_balance + complete_trades["gross_pnl"].cumsum()
complete_trades["capital"] = starting_balance + complete_trades["net_pnl"].cumsum()

monthly_sum = complete_trades.resample('M').sum()
monthly_sum_mean = monthly_sum.mean()

win_rate = len(complete_trades[complete_trades["net_pnl"] > 0]) / len(complete_trades)
print(f"Win Rate: {round(win_rate*100,2)}%")
print(f"Monthly Profit: ${round(monthly_sum_mean['net_pnl'],2)}")
print(f"Total Return: {round(((complete_trades['capital'].iloc[-1] - starting_balance)/starting_balance)*100, 2)}% ")

plt.figure(dpi = 300)

plt.xticks(rotation = 45)

plt.plot(complete_trades["gross_capital"])
plt.plot(complete_trades["capital"])

plt.legend(["gross_pnl", "net_pnl (incl. fees)"])

plt.show()