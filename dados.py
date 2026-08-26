import os
import pandas as pd
import zipfile
from histdata import download_hist_data
def baixar_dados():
        
    # Baixa diretamente passando o par e o timeframe como argumentos
    zip_path = download_hist_data(year='2024', month=None, pair='eurusd', time_frame='M1')


    df = pd.read_csv(
        "DAT_ASCII_EURUSD_M1_2024.csv", 
        sep=';', 
        names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    )

    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d %H%M%S')
    df.set_index('Datetime', inplace=True)

    # Converte para 15 minutos (Resampling)
    df_15m = df.resample('15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    print(df_15m)
    df_15m.to_csv("eurusd_2024_01_15m.csv")
    return df_15m