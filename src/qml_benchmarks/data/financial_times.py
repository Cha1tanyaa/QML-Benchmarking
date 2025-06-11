# Copyright 2025 Chaitanya Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    df = yf.download(ticker, start=start, end=end, interval="1d")

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["RSI_14"] = compute_rsi(df["Close"], window=14)

    df["target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)

    df.dropna(inplace=True)

    features = ["Close", "SMA_50", "SMA_200", "RSI_14"]
    X = df[features].values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
