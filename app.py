#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Fetch S&P 500 data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period='max')

# Plot closing prices
sp500.plot.line(y='Close', use_index=True)
plt.show()

# Clean data by removing unnecessary columns
del sp500['Dividends']
del sp500['Stock Splits']

# Create a column to predict tomorrow's price
sp500['Tomorrow'] = sp500['Close'].shift(-1)

# Create target column
sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)

# Focus on data from 1990 onwards
sp500 = sp500.loc['1990-01-01':].copy()

# Train initial machine learning model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Split data into training and test sets
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Define predictors
predictors = ['Close', 'Volume', 'Open', 'High', 'Low']

# Fit the model
model.fit(train[predictors], train['Target'])

# Generate predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Calculate precision score
print(f'Precision Score: {precision_score(test['Target'], preds)}')

# Combine actual values with predicted values
combined = pd.concat([test['Target'], preds], axis=1)

# Plot the combined data
combined.plot()
plt.show()

# Define predict function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined

# Define backtest function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:i + step].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Perform backtesting
predictions = backtest(sp500, model, predictors)

# Evaluate predictions
print(predictions['Predictions'].value_counts())
print(f'Precision Score: {precision_score(predictions['Target'], predictions['Predictions'])}')
print(predictions['Target'].value_counts() / predictions.shape[0])

# Define new predictors based on rolling averages
horizons = [2, 5, 60, 250, 1000]
new_predictors = []
for horizon in horizons:
    rolling_averages = sp500['Close'].rolling(window=horizon).mean()
    ratio_column = f'Close_Rate_{horizon}'
    sp500[ratio_column] = sp500['Close'] / rolling_averages
    trend_column = f'Trend_{horizon}'
    sp500[trend_column] = sp500['Target'].shift(1).rolling(window=horizon).sum()
    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()

# Update the model and predictors
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Update predict function for probabilistic prediction
def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined

# Perform backtesting with new predictors
predictions = backtest(sp500, model, new_predictors)

# Evaluate new predictions
print(predictions['Predictions'].value_counts())
print(f'Precision Score: {precision_score(predictions['Target'], predictions['Predictions'])}')
