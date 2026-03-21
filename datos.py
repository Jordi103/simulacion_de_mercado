import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import requests


def leer_apikey():
    with open("apikey.txt") as f:
        API = f.read()[:-1]
    return API

def download_stock_data(ticker):
    base = "https://financialmodelingprep.com/stable/"
    endpoint = f"historical-price-eod/full?symbol={ticker}"
    apikey = "&apikey=" + leer_apikey()
    URL = base + endpoint + apikey
    
    data_json = requests.get(URL)
    data = pd.DataFrame(data_json.json())
    data.index = pd.to_datetime(data['date'])
    data.drop(['symbol', 'date'], axis=1, inplace=True) 
    data.sort_index(inplace=True)

    return data


def next_k_bdays(k, start, delta=datetime.timedelta(days=1)):
    bdays = pd.bdate_range(start=start, end=start+k*delta)
    extended_k = k
    while len(bdays) < k+1:
        bdays = pd.bdate_range(start=start, end=start+extended_k*delta)
        extended_k += 1

    return bdays
    
    