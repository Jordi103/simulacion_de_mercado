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
    is_bday = lambda date: True if len(pd.bdate_range(date, date)) == 1 else False
    if not is_bday(start):
        print("WARNING: La fecha de inicio no es laborable.")
    dates_return = [start]
    current_date = start
    while len(dates_return) < k:
        next_date = current_date + delta
        if is_bday(next_date):
            dates_return.append(next_date)
        current_date = next_date
    return dates_return


class OHLC(pd.DataFrame):
    def __init__(self, df=None):
        if df is not None:
            super().__init__(df[['open', 'high', 'low', 'close']])
        else:
            super().__init__()

        
    def plot(self, ax_new=None):
        if ax_new is None:
            fig, ax = plt.subplots(1,1, figsize=(16,6))
        else:
            ax = ax_new
            
        width = 0.5
        width2 = 0.1
        pricesup=self[self.close>=self.open]
        pricesdown=self[self.close<self.open]
        
        ax.bar(pricesup.index,pricesup.close-pricesup.open,width,bottom=pricesup.open,color='g')
        ax.bar(pricesup.index,pricesup.high-pricesup.close,width2,bottom=pricesup.close,color='gray')
        ax.bar(pricesup.index,pricesup.low-pricesup.open,width2,bottom=pricesup.open,color='gray')
        
        ax.bar(pricesdown.index,pricesdown.close-pricesdown.open,width,bottom=pricesdown.open,color='r')
        ax.bar(pricesdown.index,pricesdown.high-pricesdown.open,width2,bottom=pricesdown.open,color='gray')
        ax.bar(pricesdown.index,pricesdown.low-pricesdown.close,width2, bottom=pricesdown.close,color='gray')
        
        ax.grid()

    def desde_serie(self, valores, fechas):
        # asumimos granularidad de las fechas mayor que diaria.
        
        if len(valores) != len(fechas):
            print("ERROR: valores y fechas no tienen la misma longitud.")
            return
        df = pd.DataFrame({'fechas': fechas, 'valores': valores})
        df['dia'] = df['fechas'].dt.day
        
        df_final = df.drop('fechas',axis=1).groupby('dia')
        agg_open = pd.NamedAgg('valores', 'first')
        agg_close = pd.NamedAgg('valores', 'last')
        agg_high = pd.NamedAgg('valores', 'max')
        agg_low = pd.NamedAgg('valores', 'min')
        df_final = df_final.agg(open=agg_open, close=agg_close, high=agg_high, low=agg_low)
    
        return OHLC(df_final)
        
                