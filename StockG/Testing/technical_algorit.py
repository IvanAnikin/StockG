#https://app.datacamp.com/workspace/w/ef4001e5-2f73-47c6-a2c0-d8a7489e3ca6/edit

import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv

def visualise_charts():
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Price in $')
    stock['Close'].plot(ax=ax1, color='r', lw=2.)
    ax1.plot(signals.loc[signals.positions == 1.0].index,
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(signals.loc[signals.positions == -1.0].index,
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
    portfolio['total'].plot(ax=ax1, lw=2.)
    ax1.plot(portfolio.loc[signals.positions == 1.0].index,
             portfolio.total[signals.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(portfolio.loc[signals.positions == -1.0].index,
             portfolio.total[signals.positions == -1.0],
             'v', markersize=10, color='k')
    plt.show()

source = "E:/VS_Projects/StockG_data/S&P_500/s&p_largest_50.csv" #"E:/VS_Projects/StockG_data/S&P_500/Preprocessed/2021/"
short_window = 40
long_window = 100
initial_capital= float(100000.0)
visualise = False
start=datetime.datetime(2020, 1, 1)
end=datetime.datetime(2021, 1, 1)

with open(source) as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for row in csv_reader:
        try:
            stock_name = row['Symbol']
            stock = pdr.get_data_yahoo(stock_name, start=start, end=end)

            # Calculate signals
            signals = pd.DataFrame(index=stock.index)
            signals['signal'] = 0.0
            signals['short_mavg'] = stock['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
            signals['long_mavg'] = stock['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
            signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                                        > signals['long_mavg'][short_window:], 1.0, 0.0)
            signals['positions'] = signals['signal'].diff()

            # Test signals
            positions = pd.DataFrame(index=signals.index).fillna(0.0)
            positions[stock_name] = 1000 * signals['signal']
            portfolio = positions.multiply(stock['Adj Close'], axis=0)
            pos_diff = positions.diff()
            portfolio['holdings'] = (positions.multiply(stock['Adj Close'], axis=0)).sum(axis=1)
            portfolio['cash'] = initial_capital - (pos_diff.multiply(stock['Adj Close'], axis=0)).sum(axis=1).cumsum()
            portfolio['total'] = portfolio['cash'] + portfolio['holdings']
            portfolio['returns'] = portfolio['total'].pct_change()

            # Visualisation
            if(visualise): visualise_charts()
            Return = portfolio['total'][len(portfolio['total'])-1]-initial_capital
            print("{stock_name} | {start}-{end} | Total return: {Return} = {percentage_return}%".format(stock_name=stock_name, Return = int(Return),
                                start=start.strftime("%m/%d/%Y"), end=end.strftime("%m/%d/%Y"), percentage_return = np.round(Return/initial_capital*100,1)))
        except Exception as e:
            print("Exception: " + str(e))
