import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import math
def download_stock_data(tickers, start_date, end_date):
    all_data = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data['Ticker'] = ticker
        all_data = pd.concat([all_data, data])
        
    # Fill NaN with the mean of each column
    for column in all_data.columns:
        all_data[column].fillna(all_data[column].mean(), inplace=True)
    return all_data

def fill_mean(all_data):
    for column in all_data.columns:
        all_data[column].fillna(all_data[column].mean(), inplace=True)
    return all_data

def add_features(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low_Diff'] = df['High'] - df['Low']
    df['Next_Close'] = df['Close'].shift(-1)
    return df

def train_initial_model(tickers, start_date, end_date, model_path='initial_stock_model.pkl'):
    # Download and process data
    data = download_stock_data(tickers, start_date, end_date)
    data = data.groupby('Ticker').apply(add_features)
    
    # Prepare data for machine learning
    features = data[['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'Volume_Change', 'High_Low_Diff']]
    target = data['Next_Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Initial Model - MSE: {mse}')
    
    # Save the model
    joblib.dump(model, model_path)
    print(f'Model saved as {model_path}')
    
    return model, mse

# Example of training and saving an initial model for multiple stocks
tickers = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "ORCL", "IBM", "INTC", "CSCO",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABT", "AMGN", "BMY", "UNH", "GILD", "LLY", "MDT",
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BK", "SCHW", "PNC",
    # Consumer Discretionary
    "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LULU", "LOW", "BKNG", "CMG",
    # Industrials
]


#start_date = '2015-01-01'
#end_date = '2023-01-01'
#
#initial_model, initial_mse = train_initial_model(tickers, start_date, end_date)

def fine_tune_model(ticker, start_date, end_date, initial_model_path='initial_stock_model.pkl', fine_tuned_model_path='fine_tuned_stock_model.pkl'):
    # Download and process data
    data = yf.download(ticker, start=start_date, end=end_date)
    data = add_features(data)
    data = fill_mean(data)
    # Prepare data for machine learning
    features = data[['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'Volume_Change', 'High_Low_Diff']]
    target = data['Next_Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Load the initial model
    model = joblib.load(initial_model_path)
    
    # Fine-tune the model on the new data
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Fine-Tuned Model for {ticker} - MSE: {mse}')
    
    # Save the fine-tuned model
    joblib.dump(model, fine_tuned_model_path)
    print(f'Fine-tuned model saved as {fine_tuned_model_path}')
    
    return model, mse

## Example of fine-tuning the model for a specific stock
#ticker_to_fine_tune = 'AMZN'  # Amazon stock
#start_date_fine_tune = '2015-01-01'
#end_date_fine_tune = '2023-01-01'
#
#fine_tuned_model, fine_tuned_mse = fine_tune_model(ticker_to_fine_tune, start_date_fine_tune, end_date_fine_tune, initial_model_path='initial_stock_model.pkl')

def find_local_extrema(data):
    """
    This function takes a pandas DataFrame with historical stock prices and returns the dates of local minima and maxima.

    Parameters:
    data (DataFrame): DataFrame with columns 'Date' and 'Close'.

    Returns:
    local_min_dates (list): List of dates where local minima occur.
    local_max_dates (list): List of dates where local maxima occur.
    """
    
    if 'Date' not in data.columns or 'Close' not in data.columns:
        raise ValueError("DataFrame must contain 'Date' and 'Close' columns")

    data['Close'] = data['Close'].astype(float)
    
    # Compute the differences between consecutive elements
    diff = np.diff(data['Close'])
    
    # Find local minima (peaks)
    local_min = np.where((np.diff(np.sign(diff)) > 0))[0] + 1
    local_min_dates = data['Date'].iloc[local_min].tolist()
    
    # Find local maxima (troughs)
    local_max = np.where((np.diff(np.sign(diff)) < 0))[0] + 1
    local_max_dates = data['Date'].iloc[local_max].tolist()

    return local_min_dates, local_max_dates


class Simulation:

    def __init__(self, ticker, startCapital) -> None:
        self.ticker_ = ticker
        self.capital_ = startCapital
        self.model_ = fine_tune_model(ticker, "2015-01-01", "2024-01-01")[0]
        self.buying_price_ = 0
        self.number_of_stocks_ = 0

    def sell(self, price):
        if (self.number_of_stocks_ == 0):
            return
        worth = price * self.number_of_stocks_
        profit = worth - self.buying_price_ * self.number_of_stocks_
        tax = profit / 4
        self.capital_ += (worth - tax)
        self.buying_price_ = 0
        self.number_of_stocks_ = 0

    def buy(self, price):
        if (price > self.capital_):
            print("Cant buy")
            return
        self.buying_price_ = price
        self.number_of_stocks_ =  math.floor(self.capital_ / price)
        self.capital_ -= price * self.number_of_stocks_
        
        
    
    def simulate(self):
         # Download and process data
        data = yf.download([self.ticker_], start="2024-01-01", end="2024-07-01")
        data = add_features(data)
        data = fill_mean(data)

        for date, today in data.iterrows():
            newdf = pd.DataFrame([today[['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'Volume_Change', 'High_Low_Diff']]])
            prediction = self.model_.predict(newdf)
            today["Next_Close"] = prediction

            #Calculating peaks an trouts
            df_for_peaks = pd.DataFrame(data['Close'][:date])
            df_for_peaks = pd.concat([df_for_peaks, pd.DataFrame([prediction], columns=df_for_peaks.columns)])
            df_for_peaks = find_local_peaks(df_for_peaks, 'Close')

            #Checking if the current price is a peak
            if (df_for_peaks['min'][date] == today['Close']):
                self.buy(today['Close'])
                today['Action'] = 'Buying'
            if (df_for_peaks['max'][date] == today['Close']):
                self.sell(today['Close'])
                today['Action'] = 'Selling'
        #Shifting the predicted values back
        data.Next_Close.shift(1)

        #Selling if still left some stocks
        self.sell(data['Close'][date])
        return data

        
def find_local_peaks(df, column):
    # Find local peaks
    df['min'] = df[column][(df[column].shift(1) > df[column]) & (df[column].shift(-1) > df[column])]
    df['max'] = df[column][(df[column].shift(1) < df[column]) & (df[column].shift(-1) < df[column])]
    return df

            
        


sim = Simulation("META", 10000)
data = sim.simulate()

newdf = pd.DataFrame(data)
peaks = find_local_peaks(newdf, "Next_Close")
x = newdf.index
print(sim.capital_)
plt.plot(newdf.index, newdf['Close'], label="Actual")
plt.plot(newdf.index, newdf['Next_Close'], label='Prediction')
plt.scatter(peaks.index, peaks['min'], c="r")
plt.scatter(peaks.index, peaks['max'], c="g")

plt.legend()
plt.show()
