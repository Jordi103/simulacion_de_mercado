import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import timedelta

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


def next_k_bdates(k, start, delta=datetime.timedelta(days=1)):
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

        ax.set_xticks(self.index)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


        

    def desde_simulacion(self, serie):
        # asumimos granularidad de las fechas mayor que diaria.

        df = serie.reset_index()
        df['dia'] = df['fechas'].dt.floor("D")
        
        df_final = df.drop('fechas',axis=1).groupby('dia')
        agg_open = pd.NamedAgg('valores', 'first')
        agg_close = pd.NamedAgg('valores', 'last')
        agg_high = pd.NamedAgg('valores', 'max')
        agg_low = pd.NamedAgg('valores', 'min')
        df_final = df_final.agg(open=agg_open, close=agg_close, high=agg_high, low=agg_low)

        return OHLC(df_final)

    
    def desde_volatilidad_relativa(self, serie_con_fechas):

        df = pd.DataFrame()
        # std = serie_con_fechas.std()/np.sqrt(len(serie_con_fechas))
        std = serie_con_fechas.rolling(5).std()/np.sqrt(5)
        
        df['close'] = serie_con_fechas
        df['open'] = serie_con_fechas.shift(-1)
        df['high'] = np.maximum(df['close'], df['open']) + np.abs(np.random.normal(scale=std))
        df['low'] = np.minimum(df['close'], df['open']) - np.abs(np.random.normal(scale=std))
        
        return OHLC(df.dropna())


    def desde_interpolacion_puentes_brownianos(self, serie_con_fechas, n_subpasos=20):
        
        df = pd.DataFrame()
        total_dias = len(pd.bdate_range(start=serie_con_fechas.index[0],
                                        end=serie_con_fechas.index[-1]))
    
        df.index = next_k_bdates(total_dias*n_subpasos, 
                                 start=serie_con_fechas.index[0],
                                 delta=timedelta(minutes=24*60/n_subpasos))

        Deltat_subpasos = 1./n_subpasos
        for i in range(len(serie_con_fechas)-1):
            a = serie_con_fechas.iloc[i]
            b = serie_con_fechas.iloc[i+1]
            T_subpasos = (serie_con_fechas.index[i+1]-serie_con_fechas.index[i]).total_seconds()/(60.*60*24)
            # Deltat_subpasos = T_subpasos/n_subpasos
            n_sp_total = int(T_subpasos/Deltat_subpasos)
            B = np.random.normal(size=n_sp_total-1,
                                 scale=serie_con_fechas.std()*np.sqrt(Deltat_subpasos))\
                .cumsum(axis=0)
            B = np.array([0, *B])
            bridge_00 = B - np.arange(n_sp_total)*Deltat_subpasos/T_subpasos * B[-1]
            bridge_ab = a + np.arange(n_sp_total)*Deltat_subpasos/T_subpasos*(b-a) + bridge_00
            fechas_ab = next_k_bdates(k=n_sp_total,
                                      start=serie_con_fechas.index[i],
                                      delta=timedelta(days=Deltat_subpasos))
                
            df.loc[fechas_ab, 'valores'] = bridge_ab

        df.dropna(inplace=True)

        # serie_con_fechas.reset_index(drop=True).plot()
        # df.reset_index(drop=True).plot()
        return self.desde_simulacion(df['valores'], df.index)






    
                