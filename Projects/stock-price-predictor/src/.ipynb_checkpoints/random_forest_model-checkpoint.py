#!/usr/bin/env python3
# \"\"\"Random Forest baseline for next-day stock price prediction.\"\"\"
import argparse, os, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from utils import download_data, create_features, time_train_test_split, plot_preds, evaluate_and_print

def main(args):
    df = download_data(args.ticker, args.start, args.end)
    df = create_features(df)
    features = ['Close','MA5','MA20','Return','Volume']
    X = df[features].values
    y = df['Target'].values

    # simple imputer (if needed)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_size=args.test_size)

    # Grid search small
    param_grid = {'n_estimators':[100], 'max_depth':[5,10,None]}
    rfr = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rfr, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    print(f'Best params: {grid.best_params_}')
    print(f'Random Forest MAE: {mae:.4f}')

    # save model and plots
    os.makedirs('models', exist_ok=True)
    joblib.dump(best, f'models/rf_{args.ticker}.joblib')

    os.makedirs('figs', exist_ok=True)
    plot_preds(y_test, preds, title=f'{args.ticker} Random Forest Predictions', savepath=f'figs/rf_{args.ticker}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
