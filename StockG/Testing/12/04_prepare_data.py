import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib
from talib import RSI, BBANDS, MACD, ATR
MONTH = 21
YEAR = 12 * MONTH
START = '2010-01-01'
END = '2017-12-31'
sns.set_style('darkgrid')
idx = pd.IndexSlice
percentiles = [.001, .01, .02, .03, .04, .05]
percentiles += [1-p for p in percentiles[::-1]]
T = [1, 5, 10, 21, 42, 63]

DATA_STORE = './12/data/assets.h5'
ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']
df = pd.read_csv("./data/WIKI_PRICES.csv")
#df = dataset.loc[dataset['ticker'] == .ticker]
# no longer needed
# df = pd.concat([df.loc[:, 'code'].str.strip(),
#                 df.loc[:, 'name'].str.split('(', expand=True)[0].str.strip().to_frame('name')], axis=1)
print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/stocks', df)

#for column in df:
#    if column not in ohlcv:
#        df.drop(column, axis=1, inplace=True)

#with pd.HDFStore(DATA_STORE) as store:
#    prices = (store['quandl/wiki/prices']
#              .loc[idx[START:END, :], ohlcv] # select OHLCV columns from 2010 until 2017
#              .rename(columns=lambda x: x.replace('adj_', '')) # simplify column names
#              .swaplevel()
#              .sort_index())
#    metadata = (store['us_equities/stocks'].loc[:, ['marketcap', 'sector']])

prices = df
metadata = pd.read_csv("./data/us_equilities_meta_data.csv")

prices['volume'] /= 1e3 # make vol figures a bit smaller
prices.index.names = ['symbol'] #'date'
metadata.index.name = 'symbol'

min_obs = 7 * YEAR
nobs = prices.groupby(level='symbol').size()
keep = nobs[nobs > min_obs].index
prices = prices.iloc[idx[keep, :], :] #?-[idx[keep, :]]

metadata = metadata[~metadata.index.duplicated() & metadata.sector.notnull()]
metadata.sector = metadata.sector.str.lower().str.replace(' ', '_')
shared = (prices.index.get_level_values('symbol').unique()
          .intersection(metadata.index))
metadata = metadata.loc[shared, :]
prices = prices.loc[idx[shared, :], :]

universe = metadata.marketcap.nlargest(1000).index
prices = prices.loc[idx[universe, :], :]
metadata = metadata.loc[universe]
metadata.sector.value_counts()

prices['dollar_vol'] = prices[['close', 'volume']].prod(1).div(1e3)

# compute dollar volume to determine universe
dollar_vol_ma = (prices
                 .dollar_vol
                 .unstack('symbol')
                 .rolling(window=21, min_periods=1) # 1 trading month
                 .mean())

prices['dollar_vol_rank'] = (dollar_vol_ma
                            .rank(axis=1, ascending=False)
                            .stack('symbol')
                            .swaplevel())
prices.info(show_counts=True)

prices['rsi'] = prices.groupby(level='symbol').close.apply(RSI)
ax = sns.distplot(prices.rsi.dropna())
ax.axvline(30, ls='--', lw=1, c='k')
ax.axvline(70, ls='--', lw=1, c='k')
ax.set_title('RSI Distribution with Signal Threshold')
sns.despine()
plt.tight_layout();

def compute_bb(close):
    high, mid, low = BBANDS(close, timeperiod=20)
    return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)
prices = (prices.join(prices
                      .groupby(level='symbol')
                      .close
                      .apply(compute_bb)))
prices['bb_high'] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
prices['bb_low'] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)
fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
sns.distplot(prices.loc[prices.dollar_vol_rank<100, 'bb_low'].dropna(), ax=axes[0])
sns.distplot(prices.loc[prices.dollar_vol_rank<100, 'bb_high'].dropna(), ax=axes[1])
sns.despine()
plt.tight_layout();

prices['NATR'] = prices.groupby(level='symbol',
                                group_keys=False).apply(lambda x:
                                                        talib.NATR(x.high, x.low, x.close))
def compute_atr(stock_data):
    df = ATR(stock_data.high, stock_data.low,
             stock_data.close, timeperiod=14)
    return df.sub(df.mean()).div(df.std())
prices['ATR'] = (prices.groupby('symbol', group_keys=False)
                 .apply(compute_atr))
prices['PPO'] = prices.groupby(level='symbol').close.apply(talib.PPO)
def compute_macd(close):
    macd = MACD(close)[0]
    return (macd - np.mean(macd))/np.std(macd)
prices['MACD'] = (prices
                  .groupby('symbol', group_keys=False)
                  .close
                  .apply(compute_macd))
metadata.sector = pd.factorize(metadata.sector)[0].astype(int)
prices = prices.join(metadata[['sector']])
by_sym = prices.groupby(level='symbol').close
for t in T:
    prices[f'r{t:02}'] = by_sym.pct_change(t)
for t in T:
    prices[f'r{t:02}dec'] = (prices[f'r{t:02}']
                             .groupby(level='date')
                             .apply(lambda x: pd.qcut(x,
                                                      q=10,
                                                      labels=False,
                                                      duplicates='drop')))
for t in T:
    prices[f'r{t:02}q_sector'] = (prices
                                  .groupby(['date', 'sector'])[f'r{t:02}']
                                  .transform(lambda x: pd.qcut(x,
                                                               q=5,
                                                               labels=False,
                                                               duplicates='drop')))
for t in [1, 5, 21]:
    prices[f'r{t:02}_fwd'] = prices.groupby(level='symbol')[f'r{t:02}'].shift(-t)
prices[[f'r{t:02}' for t in T]].describe()

outliers = prices[prices.r01 > 1].index.get_level_values('symbol').unique()
prices = prices.drop(outliers, level='symbol')
prices['year'] = prices.index.get_level_values('date').year
prices['month'] = prices.index.get_level_values('date').month
prices['weekday'] = prices.index.get_level_values('date').weekday
prices.info(show_counts=True)

prices.drop(['open', 'close', 'low', 'high', 'volume'], axis=1).to_hdf('./12/data.h5', 'model_data')
