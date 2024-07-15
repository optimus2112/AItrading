

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib as plt

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def calculate_stock_parameters(df):
    # Calculate moving averages
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
   
    # Accumulation/Distribution Line
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
   
    # Parabolic SAR (Using an external library could be easier, but we will keep it simple)
    # Example: Using TA-Lib for this calculation
   
    # Pivot Points
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Pivot_R1'] = 2 * df['Pivot'] - df['Low']
    df['Pivot_S1'] = 2 * df['Pivot'] - df['High']
   
    df = df.fillna(df.mean())
    df['min'] = df['Close'][(df['Close'].shift(1) > df['Close']) & (df['Close'].shift(-1) > df['Close'])]
    df['max'] = df['Close'][(df['Close'].shift(1) < df['Close']) & (df['Close'].shift(-1) < df['Close'])]

    df['IsPeak'] = df.apply(lambda row: 1 if row['Close'] == row['max'] else (-1 if row['Close'] == row['min'] else 0), axis=1)
    return df


def find_local_peaks(df, column):
    # Find local peaks
    df['min'] = df[column][(df[column].shift(1) > df[column]) & (df[column].shift(-1) > df[column])]
    df['max'] = df[column][(df[column].shift(1) < df[column]) & (df[column].shift(-1) < df[column])]
    return df



def plot_stock_data(df): 
    fig, axs = plt.subplots(6, 1, figsize=(15, 25), sharex=True) # Price and Moving Averages 
    axs[0].plot(df['Close'], label='Close Price')
    axs[0].plot(df['SMA_20'], label='SMA 20')
    axs[0].plot(df['EMA_20'], label='EMA 20') 
    axs[0].set_title('Stock Price and Moving Averages') 
    axs[0].legend() 
    # MACD 
    axs[1].plot(df['MACD'], label='MACD') 
    axs[1].plot(df['MACD_Signal'], label='MACD Signal') 
    axs[1].set_title('MACD') 
    axs[1].legend() # Bollinger Bands 
    axs[2].plot(df['Close'], label='Close Price') 
    axs[2].plot(df['BB_Middle'], label='BB Middle') 
    axs[2].plot(df['BB_Upper'], label='BB Upper') 
    axs[2].plot(df['BB_Lower'], label='BB Lower') 
    axs[2].set_title('Bollinger Bands') 
    axs[2].legend() # RSI 
    axs[3].plot(df['RSI'], label='RSI') 
    axs[3].set_title('Relative Strength Index (RSI)') 
    axs[3].legend() # Volume and OBV 
    axs[4].plot(df['Volume'], label='Volume', color='gray') 
    axs[4].plot(df['OBV'], label='On-Balance Volume (OBV)') 
    axs[4].set_title('Volume and OBV') 
    axs[4].legend() # ATR 
    axs[5].plot(df['ATR'], label='Average True Range (ATR)') 
    axs[5].set_title('Average True Range (ATR)') 
    axs[5].legend() 
    plt.tight_layout() 
    plt.show()  



def download_stock_data(tickers, start_date, end_date):
    all_data = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data = calculate_stock_parameters(data)
        all_data = pd.concat([all_data, data])
        
    # Filling NaN cells
    all_data = all_data.fillna(all_data.mean())
    return all_data  

def train_initial_model(tickers, start_date, end_date, model_path='initial_stock_model_peaks.pkl'):
    # Download and process data
    data = download_stock_data(tickers, start_date, end_date)
    x = data.drop(columns=['IsPeak'])
    y = data['IsPeak']

    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the model
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy: ",accuracy_score(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, model_path)
    print(f'Model saved as {model_path}')
    
    return model



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



model = joblib.load('initial_stock_model_peaks.pkl')

# Load historical data for an unseen stock (e.g., Tesla)
unseen_stock_ticker = 'META'
unseen_stock_data = yf.download(unseen_stock_ticker, start='2020-01-01', end='2023-01-01')

# Calculate the parameters for the unseen stock data
unseen_stock_data = calculate_stock_parameters(unseen_stock_data)

# Prepare the data for prediction (excluding the target and any identifier columns)
X_unseen = unseen_stock_data.drop(columns=['IsPeak'], errors='ignore')

# Use the trained model to predict local extremes
predictions = model.predict(X_unseen)
unseen_stock_data['Predictions'] = predictions

# Initial capital
initial_capital = 1000  # Starting with $1,000
capital = initial_capital
position = 0  # No initial stock position
tax_rate = 0.25  # Tax rate is 25%

# Trading simulation
for i in range(1, len(unseen_stock_data)):
    if unseen_stock_data['Predictions'].iloc[i] == -1:  # Local minimum (buy signal)
        if capital > 0:  # Ensure we have cash to buy
            position = capital / unseen_stock_data['Close'].iloc[i]  # Buy as many shares as possible
            capital = 0  # All cash converted to stock
            print(f"Buy at {unseen_stock_data['Close'].iloc[i]} on {unseen_stock_data.index[i]}")
    elif unseen_stock_data['Predictions'].iloc[i] == 1:  # Local maximum (sell signal)
        if position > 0:  # Ensure we have stock to sell
            sell_price = unseen_stock_data['Close'].iloc[i]
            capital = position * sell_price  # Sell all shares
            buy_price = capital / position  # Calculate the effective buy price per share
            profit = capital - (buy_price * position)  # Profit from the sale
            tax = profit * tax_rate  # Calculate tax
            capital -= tax  # Deduct tax from capital
            position = 0  # All stock converted to cash
            print(f"Sell at {sell_price} on {unseen_stock_data.index[i]}, Tax: {tax:.2f}")

# Calculate final value (if holding any stock, convert to cash at the last close price)
if position > 0:
    capital = position * unseen_stock_data['Close'].iloc[-1]

# Calculate profit/loss
profit_loss = capital - initial_capital
print(f"Final capital: ${capital:.2f}")
print(f"Profit/Loss: ${profit_loss:.2f}")

df = pd.DataFrame(unseen_stock_data)

# Filter the DataFrame to get local maxima and minima
local_maxima = df[df['Predictions'] == 1]
local_minima = df[df['Predictions'] == -1]

# Plot the close prices
plt.figure(figsize=(10, 5))

# Highlight local maxima
plt.scatter(local_maxima.index, local_maxima['Close'], color='red', label='Local Maxima')

# Highlight local minima
plt.scatter(local_minima.index, local_minima['Close'], color='green', label='Local Minima')
plt.plot(unseen_stock_data.index, unseen_stock_data['Close'])
# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Close Prices with Local Extremes')
plt.legend()

# Show the plot
plt.show()
