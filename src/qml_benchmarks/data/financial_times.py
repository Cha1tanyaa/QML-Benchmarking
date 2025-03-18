import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def compute_rsi(series, window=14):
    """
    Compute the Relative Strength Index (RSI) for a price series.

    Args:
        series (pandas.Series): Time series of stock closing prices.
        window (int): Window length for calculating RSI.

    Returns:
        pandas.Series: RSI values.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_stock_features_and_labels(ticker="AAPL", start="2010-01-01", end="2024-01-01"):
    """
    Data generation procedure for financial time series data.

    Args:
        ticker (str): Stock ticker symbol.
        start (str): Start date for historical data.
        end (str): End date for historical data.

    Returns:
        X (ndarray): Standardized feature array of shape (n_samples, n_features).
        y (ndarray): Array of labels (-1 corresponds to a price decrease, +1 to a price increase).
    """
    # Download historical stock data
    df = yf.download(ticker, start=start, end=end, interval="1d")

    # Compute technical indicators
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["RSI_14"] = compute_rsi(df["Close"], window=14)

    # Generate binary target: +1 if next day's closing price is higher, -1 otherwise
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, -1)

    # Remove rows with missing values (from rolling computations and shift)
    df.dropna(inplace=True)

    # Select features
    features = ["Close", "SMA_50", "SMA_200", "RSI_14"]
    X = df[features].values
    y = df["Target"].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
