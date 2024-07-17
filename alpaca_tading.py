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
import json
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



def calculate_new_row(df):
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=1) * 100

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()

    # Standard Deviation
    df['STD'] = df['Close'].rolling(window=20).std()

    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)
    df['ATR'] = tr.rolling(window=14).mean()

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Accumulation/Distribution Line (ADL)
    ad = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['ADL'] = ad.cumsum()

    # Stochastic Oscillator
    df['L14'] = df['Low'].rolling(window=14).min()
    df['H14'] = df['High'].rolling(window=14).max()
    df['%K'] = (df['Close'] - df['L14']) * 100 / (df['H14'] - df['L14'])
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Chaikin Money Flow (CMF)
    df['CMF'] = (ad.rolling(window=20).sum()) / (df['Volume'].rolling(window=20).sum())

    # Williams %R
    df['Williams %R'] = (df['H14'] - df['Close']) / (df['H14'] - df['L14']) * -100

    # Pivot Points
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Pivot_R1'] = 2 * df['Pivot'] - df['Low']
    df['Pivot_S1'] = 2 * df['Pivot'] - df['High']

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
    return pd.DataFrame(data).tail(1)

# Function to load a dictionary from a JSON file
def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to save a dictionary to a JSON file
def save_dict_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

class Trader:
    def __init__(self) -> None:
        self.tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK']
       
    def order(self, symbol, qty, action):
        client = TradingClient(API_KEY, API_SECRET)
        req = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=action,
            type='market',
            time_in_force='day',
        )
        client.submit_order(req)


    def sell(self, symbol, price):
        symbol_data = load_dict_from_json('resources\\'+symbol+'.json')
        position = symbol_data['Position']
        if (position <= 0):
            return
        self.order(symbol=symbol, qty=position, action='sell')
        symbol_data['Capital'] = position*price
        symbol_data['Position'] = 0
        save_dict_to_json(symbol_data, 'resources\\'+symbol+'.json')


    def buy(self, symbol, price):   
        symbol_data = load_dict_from_json('resources\\'+symbol+'.json')
        capital = symbol_data['Capital']
        if (capital == 0):
            return
        position = capital / price
        self.order(symbol=symbol, qty=position, action='buy')
        symbol_data['Capital'] = 0
        symbol_data['Position'] = position
        save_dict_to_json(symbol_data, 'resources\\'+symbol+'.json')


    def trade_symbol(self, symbol):
        model = joblib.load('initial_stock_model_peaks.pkl')
        history_data = load_csv('resources\\'+symbol+'.csv')
        new_data = get_current_data(symbol=symbol)
        history_data = calculate_new_row(pd.concat([history_data, new_data]))
        prediction = model.predict(history_data.tail(1))
        if (prediction == 1):
            self.sell(symbol=symbol, price=new_data['Close'].iloc[-1])
        elif(prediction == -1):
            self.buy(symbol=symbol, price=new_data['Close'].iloc[-1])

    def trade_all(self):
        for ticker in self.tickers:
            self.trade_symbol(ticker)
        

def main():
    trader = Trader()
    trader.trade_all()
    
