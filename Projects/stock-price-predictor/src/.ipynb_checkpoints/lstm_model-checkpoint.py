#!/usr/bin/env python3
#\"\"\"LSTM model for next-day stock price prediction.\"\"\"
import argparse, os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from utils import download_data, create_features, time_train_test_split, plot_preds, evaluate_and_print

def create_sequences(X, y, seq_len=20):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def main(args):
    df = download_data(args.ticker, args.start, args.end)
    df = create_features(df)
    features = ['Close','MA5','MA20','Return','Volume']

    X = df[features].values
    y = df['Target'].values.reshape(-1,1)

    # scale
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    seq_len = args.seq_len
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len=seq_len)

    split = int((1-args.test_size) * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = build_model((X_train.shape[1], X_train.shape[2]))
    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint(f'models/lstm_{args.ticker}.keras', save_best_only=True, monitor='val_loss')
    early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                        validation_split=0.1, callbacks=[checkpoint, early], verbose=1)

    # predict and inverse transform
    preds_scaled = model.predict(X_test)
    preds = scaler_y.inverse_transform(preds_scaled)
    actual = scaler_y.inverse_transform(y_test)

    os.makedirs('figs', exist_ok=True)
    plot_preds(actual, preds, title=f'{args.ticker} LSTM Predictions', savepath=f'figs/lstm_{args.ticker}.png')

    mae = evaluate_and_print(actual, preds)
    print(f'LSTM MAE: {mae:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)
