import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# ----------------------
# 1. Download Stock Data
# ----------------------
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Feature engineering
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA20"] = df["Close"].rolling(window=20).mean()
df["Return"] = df["Close"].pct_change()
df["Target"] = df["Close"].shift(-1)

df = df.dropna()


# ----------------------
# 2. Prepare data
# ----------------------
features = ["Close", "MA5", "MA20", "Return", "Volume"]
X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# ----------------------
# 3. Train model
# ----------------------
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)
error = mean_absolute_error(y_test, preds)

print(f"Mean Absolute Error: {round(error, 2)}")


# ----------------------
# 4. Visualize prediction vs actual
# ----------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(preds, label="Predicted", linestyle="--")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.show()
