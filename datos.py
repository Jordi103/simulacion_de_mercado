import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    return data