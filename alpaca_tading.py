from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockLatestBarRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.broker.client import BrokerClient
from alpaca.broker.requests import CreateAccountRequest
from alpaca.trading.requests import OrderRequest
from alpaca.trading.client import TradingClient
from datetime import datetime
import numpy as np
import joblib
import pandas as pd

import alpaca_trade_api as tradeapi
import yfinance as yf
# Replace these with your actual API key and secret key
API_KEY = 'PK6ABVP71J0P7YUS1VJ5'
API_SECRET = 'InbhJR0It5qx1z9kGpPdq0fmgWlNuNvrc6YRxUjh'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading



def get_historical_data(ticker):
    # Authenticate
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    # Fetch account information
    today = datetime.today().date()
    print(today)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        start='2020-01-01',
        timeframe=TimeFrame.Day
    )

    response = client.get_stock_bars(request)
    return response.df



def calculate_new_row(df, new_row):
    # Assuming you have your original DataFrame df
    # and new_row is a dictionary with the new data to be added

    # Append the new row to the DataFrame
    df = df.append(new_row, ignore_index=True)

    # Update the calculations for the new row
    last_index = df.index[-1]

    # Calculate SMA and EMA
    if len(df) >= 20:
        df.at[last_index, 'SMA_20'] = df['Close'].rolling(window=20).mean().iloc[-1]
    df.at[last_index, 'EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]

    # Rate of Change
    df.at[last_index, 'ROC'] = df['Close'].pct_change(periods=1).iloc[-1] * 100

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df.at[last_index, 'RSI'] = 100 - (100 / (1 + rs.iloc[-1]))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df.at[last_index, 'MACD'] = ema_12.iloc[-1] - ema_26.iloc[-1]
    df.at[last_index, 'MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean().iloc[-1]

    # Bollinger Bands
    if len(df) >= 20:
        df.at[last_index, 'BB_Middle'] = df['Close'].rolling(window=20).mean().iloc[-1]
        df.at[last_index, 'BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std().iloc[-1]
        df.at[last_index, 'BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std().iloc[-1]

    # Standard Deviation
    if len(df) >= 20:
        df.at[last_index, 'STD'] = df['Close'].rolling(window=20).std().iloc[-1]

    # Average True Range (ATR)
    if len(df) >= 14:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)
        df.at[last_index, 'ATR'] = tr.rolling(window=14).mean().iloc[-1]

    # On-Balance Volume (OBV)
    df.at[last_index, 'OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum().iloc[-1]

    # Accumulation/Distribution Line
    ad = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df.at[last_index, 'ADL'] = ad.cumsum().iloc[-1]

    # Stochastic Oscillator
    if len(df) >= 14:
        df.at[last_index, 'L14'] = df['Low'].rolling(window=14).min().iloc[-1]
        df.at[last_index, 'H14'] = df['High'].rolling(window=14).max().iloc[-1]
        df.at[last_index, '%K'] = (df['Close'] - df['L14']) * 100 / (df['H14'] - df['L14'])
        df.at[last_index, '%D'] = df['%K'].rolling(window=3).mean().iloc[-1]

    # Chaikin Money Flow (CMF)
    if len(df) >= 20:
        df.at[last_index, 'CMF'] = ad.rolling(window=20).sum().iloc[-1] / df['Volume'].rolling(window=20).sum().iloc[-1]

    # Williams %R
    if len(df) >= 14:
        df.at[last_index, 'Williams %R'] = (df['H14'] - df['Close']) / (df['H14'] - df['L14']) * -100

    # Pivot Points
    df.at[last_index, 'Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df.at[last_index, 'Pivot_R1'] = 2 * df['Pivot'] - df['Low']
    df.at[last_index, 'Pivot_S1'] = 2 * df['Pivot'] - df['High']
    return df
    # Note: For Parabolic SAR, consider using a library like TA-Lib for accurate calculation.





def get_sp100_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    table = pd.read_html(url, header=0)
    df = table[0]
    tickers = df['Symbol'].to_list()
    tickers = [str(ticker) for ticker in tickers]
    return tickers[:50]

def load_csv(csv_path):
    try:
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except FileExistsError:
        data = pd.DataFrame()
    return data


def get_current_data(symbol):
    data = yf.download(tickers=symbol, start='2024-07-10',end=datetime.today())
    return pd.DataFrame(data).iloc[-1]

class Trader:
    def __init__(self, capital) -> None:
        self.tickers = get_sp100_tickers()
        for ticker in self.tickers:
            data = {
                'Capital': capital,
                'Position': 0
            }
            data = pd.DataFrame(data)
            data.to_csv('resources\\'+ticker+'capital.csv')

    def order(self, symbol, qty, action):
        client = TradingClient(API_KEY, API_SECRET)
        req = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=action,
            type='market',
            time_in_force='gtc',
        )
        client.submit_order(req)


    def sell(self, symbol, price):
        symbol_data = load_csv('resources\\'+symbol+'capital.csv')
        position = symbol_data['Position']
        if (position <= 0):
            return
        self.order(symbol=symbol, qty=position, action='sell')
        symbol_data['Capital'] = position*price
        symbol_data['Position'] = 0
        symbol_data.to_csv('resources\\'+symbol+'capital.csv')


    def buy(self, symbol, price):
        symbol_data = load_csv('resources\\'+symbol+'capital.csv')
        capital = symbol_data['Capital']
        if (capital < price):
            return
        position = capital / price
        self.order(symbol=symbol, qty=position, action='buy')
        symbol_data['Capital'] = 0
        symbol_data['Position'] = position
        symbol_data.to_csv('resources\\'+symbol+'capital.csv')

    def trade_symbol(self, symbol):
        model = joblib.load('initial_stock_model_peaks.pkl')
        history_data = load_csv('resources\\'+symbol+'.csv')
        new_data = get_current_data(symbol=symbol)
        history_data = calculate_new_row(history_data, new_data)
        prediction = model.predict(history_data.iloc[-1])
        if (prediction == 1):
            self.sell(symbol=symbol, price=new_data['Close'])
        elif(prediction == -1):
            self.buy(symbol=symbol, price=new_data['Close'])

    def trade_all(self):
        for ticker in self.tickers:
            self.trade_symbol(ticker)
        

def main():
    trader = Trader(1000)
    trader.trade_all()
    
