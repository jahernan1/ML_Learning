import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score

def download_data(ticker='AAPL', start='2015-01-01', end=None, save_csv=None):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise ValueError("No data downloaded - check ticker and dates")
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return macd_line, signal_line, hist

def create_features(df):
    df = df.copy()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Return'] = df['Close'].pct_change()
     
    # RSI
    df["RSI14"] = compute_rsi(df["Close"], window=14)

    # MACD
    macd, macd_signal, macd_hist = compute_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist
    
    df['Target'] = df['Close'].shift(-1)
    df['Signal'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

def time_train_test_split(X, y, test_size=0.2):
    split = int((1 - test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

def plot_preds(actual, predicted, title='Prediction vs Actual', savepath=None):
    plt.figure(figsize=(12,6))
    plt.plot(actual, label='Actual', linewidth=2)
    plt.plot(predicted, label='Predicted', linestyle='--')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Price ($)')
    plt.legend()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()

def evaluate_and_print(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    print(f'Mean Absolute Error: {mae:.4f}')
    return mae
