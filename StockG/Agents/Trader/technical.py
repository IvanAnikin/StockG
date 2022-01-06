import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv
import io
from PIL import Image

def visualise_charts(stock, signals, portfolio):
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

    buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    signals_img =  buf  #np.array(fig.canvas.renderer._renderer)
    #buf.seek(0)
    #im = Image.open(buf)
    #im.show()
    #buf.close()

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

    buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    portfolio_img =  buf    #np.array(fig.canvas.renderer._renderer)
    #buf.seek(0)
    #im = Image.open(buf)
    #im.show()   #np.array(fig.canvas.renderer._renderer)

    return signals_img, portfolio_img

def get_signals(short_window = 40, long_window = 100, initial_capital = float(100000.0), visualise = True,
                start = datetime.datetime(2015, 1, 1), end = datetime.datetime(2021, 1, 1), stock_name="MSFT"):

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

    Return = portfolio['total'][len(portfolio['total']) - 1] - initial_capital
    log_string = "{stock_name} | {start}-{end} | Total return: {Return} = {percentage_return}%".format(
        stock_name=stock_name,
        Return=int(Return), start=start.strftime("%m/%d/%Y"), end=end.strftime("%m/%d/%Y"),
        percentage_return=np.round(Return / initial_capital * 100, 1))

    # Visualisation
    if (visualise):
        print(log_string)

        signals_img, portfolio_img = visualise_charts(stock, signals, portfolio)

        return log_string, Return, signals_img, portfolio_img

    return log_string, Return #signals_img, portfolio_img, Return

def get_signals_loop(start = datetime.datetime(2015, 1, 1), end = datetime.datetime(2021, 1, 1)):
    best = []
    highest_return = 0
    for short_window in range(20, 100, 5):
        for long_window in range(80, 160, 5):
            log_string, Return = get_signals(short_window=short_window, long_window=long_window,
                                                                         visualise=False, start=start, end=end)
            if Return>highest_return: best=[short_window, long_window]

    log_string, Return, signals_img, portfolio_img = get_signals(short_window=best[0], long_window=best[1],
                                                                 visualise=True, start=start, end=end)
    return best, log_string, int(Return), signals_img, portfolio_img
