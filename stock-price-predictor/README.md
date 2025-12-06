## Stock Price Prediction with Machine Learning & LSTM Deep Learning

A complete end-to-end project for learning practical machine learning and deep learning techniques for time-series forecasting using real stock market data.
This project walks through:

- Fetching financial data using yfinance
- Feature engineering (moving averages, returns, volume)
- Training a Random Forest Regressor
- Building an LSTM neural network for next-day price prediction
- Visualizing and evaluating model performance

This repository is designed for students, researchers, and developers who want a hands-on introduction to financial ML modeling.

---

## Project Overview

Stock prices are highly nonlinear and influenced by many factors. This project introduces two modeling approaches:

1. Random Forest Regressor (Traditional ML)
    - A strong baseline model that uses engineered features to predict the next day’s closing price.

2. LSTM Neural Network (Deep Learning)
    - A sequence-based model designed to learn patterns from time-series data using a sliding window of historical prices.

The project demonstrates how to prepare sequential data, scale features, train recurrent neural networks, and compare model performance against traditional ML.

---

## Repository Structure

```
stock-price-predictor/
│── data/
│   └── AAPL.csv
│── notebooks/
│   └── stock_price_predictor.ipynb
│── src/
│   ├── random_forest_model.py
│   ├── lstm_model.py
│   └── utils.py
│── requirements.txt
└── README.md
```
---

## Features

### Data Pipeline

- Downloads historical stock price data
- Cleans and preprocesses missing data
- Generates technical features:
  - 5-day moving average (MA5)
  - 20-day moving average (MA20)
  - Daily returns
  - Volume

### Machine Learning Model

- Random Forest Regressor
- Predicts next-day closing price
   Produces visualization comparing predicted vs actual prices

### Deep Learning Model (LSTM)

- Two-layer LSTM architecture
- Dropout for regularization
- Sliding window sequence construction
- Scaled inputs and inverse-transformed predictions
- Loss curves and prediction plots

## Installation

Clone the repository:
`git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor`

Install dependencies:
`pip install -r requirements.txt`

### How to Run
Jupyter Notebook (recommended)
`jupyter notebook`

Open:
`notebooks/stock_price_predictor.ipynb`

### Run Individual Python Scripts

Random Forest model:
`python src/random_forest_model.py`

LSTM model:
`python src/lstm_model.py`

## Example Results
| Model |	Mean Absolute Error (MAE) |
| :---- | :------------------------ |
| Random Forest |	~1–3 USD |
| LSTM | ~1–2 USD |

Both models output prediction plots comparing true vs predicted closing prices.

## Key Concepts Learned

- Time-series feature engineering
- Normalization for neural networks
- Building sequence windows for LSTMs
- LSTM model design and training
- Time-series validation
- Performance evaluation with MAE
- Comparing ML vs Deep Learning forecasting

## Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- yfinance

## Future Improvements

- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Predict multiple days ahead (7-day, 30-day
- Try Bidirectional LSTM or GRU models
- Experiment with Transformer-based models
- Add hyperparameter tuning
