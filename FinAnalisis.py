# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
import pandas_datareader as pdr
import datetime

aapl = pdr.get_data_yahoo('AAPL',start=datetime.datetime(2006,10,1), end = datetime.datetime(2012,1,1))


#%%
import quandl
aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01",end_date="2012-01-01")


#%%

aapl.head()

# aapl.tail()

# aapl.describe()


#%%
# tip you can save to csv and read it from csv 

# import pandas as pd
# aapl.to_csv('data/aapl_ohlc.csv')
# df = pd.read_csv('data/aapl_ohlc.csv', header=0, index_col='Date', parse_dates=True)


#%%
aapl.index
aapl.columns
ts = aapl['Close'][-10:]
type(ts)


#%%
# Sample 20 rows
sample = aapl.sample(20)

# # Print `sample`
print(sample)

# # Resample to monthly level 
monthly_aapl = aapl.resample('M').mean()

# # Print `monthly_aapl`
print(monthly_aapl)


#%%
aapl.asfreq("M", method="bfill")


#%%
# Add a column `diff` to `aapl` 
aapl['diff'] = aapl.Open - aapl.Close

# Delete the new `diff` column
del aapl['diff']


#%%
import matplotlib.pyplot as plt 
aapl['Close'].plot(grid=True)

plt.show()


#%%
# Import `numpy` as `np`
import numpy as np

# Assign `Adj Close` to `daily_close`
daily_close = aapl[['Adj. Close']]  # Notice there is a dot for 'adj.'

# Daily returns
daily_pct_change = daily_close.pct_change()

# Replace NA values with 0
daily_pct_change.fillna(0, inplace=True)

# # Inspect daily returns
# print(daily_pct_change)

# # # Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)

# # # Print daily log returns
print(daily_log_returns)


#%%

## monthly or quarterly return

# Resample `aapl` to business months, take last observation as value 
monthly = aapl.resample('BM').apply(lambda x: x[-1])
# print(monthly)

# # Calculate the monthly percentage change
monthly.pct_change()

# # Resample `aapl` to quarters, take the mean as value per quarter
quarter = aapl.resample("4M").mean()

# # Calculate the quarterly percentage change
quarter.pct_change()


#%%
# Using pct_change() is quite the convenience, but it also obscures how exactly the daily percentages are calculated. 
# That’s why you can alternatively make use of Pandas’ shift() function instead of using pct_change(). 
# You then divide the daily_close values by the daily_close.shift(1) -1. By using this function, 
# however, you will be left with NA values at the beginning of the resulting DataFrame.

# Daily returns
daily_pct_change =daily_close/ daily_close.shift(1) - 1

# Print `daily_pct_change`
print(daily_pct_change)


#%%
#For your reference, the calculation of the daily percentage change is based on the following formula: 
#rt=pt/(pt−1)−1, where p is the price, t is the time (a day in this case) and r is the return.

# dairly log return 
daily_log_returns_shift = np.log(daily_close / daily_close.shift(1))

# print(daily_log_returns_shift)


#%%
# plot the distribution of daily_pct_change


# Import matplotlib
import matplotlib.pyplot as plt

# Plot the distribution of `daily_pct_c`
daily_pct_change.hist(bins=50)

# Show the plot
plt.show()

# Pull up summary statistics
print(daily_pct_change.describe())


#%%
# The cumulative daily rate of return

# Calculate the cumulative daily return

cum_daily_return = (1 + daily_pct_change).cumprod()

# Print `cum_daily_return`
print(cum_daily_return)


#%%
# use matplotlib to quickly plot the cum_daily_return

# Import matplotlib
# import matplotlib.pyplot as plt 

# Plot the cumulative daily returns
cum_daily_return.plot(figsize=(12,8))  # this is pandas dataframe plot function

# Show the plot
plt.show()


#%%
# wnat to see the monthly return  
# use resample again

# Resample the cumulative daily return to cumulative monthly return
cum_monthly_return = cum_daily_return.resample('M').mean()

# print the cum_monthly_return 
print(cum_monthly_return)


#%%
# Knowing how to calculate the returns is a valuable skill, but you’ll often see that these numbers
# don’t really say much when you don’t compare them to other stock. 
# That’s why you’ll often see examples where two or more stocks are compared. 
# In the rest of this section, you’ll focus on getting more data from Yahoo! Finance 
# so that you can calculate the daily percentage change and compare the results.

## Compare the stocks

### get more stock need to use a function to do it



#%%
import pandas as pd
from pandas_datareader import data as pdr

def get(tickers, startdate, enddate):
  def data(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map (data, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))

print(all_data)


#%%
# use pandas DataFrame to make some plot 

import matplotlib.pyplot as plt 

# Isolate the 'Adj close' values and transform the DataFrame

daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date','Ticker','Adj Close')

# print(daily_close_px) 


# calculate the daily percentage change for 'daily_close_px'
daily_pct_change = daily_close_px.pct_change()


# print(daily_pct_change)

# plot the distribution of daily_pct_change
daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))

# show the result plot 
plt.show()


#%%
# Import matplotlib
import matplotlib.pyplot as plt
# from pandas.plotting  import scatter_matrix

# Plot the scatter matrix with daily_pct_change
pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,figsize=(12,12))
# pd.scatter_matrix(daily_pct_change, diagonal ='kde', alpha =0.1, figsize=(12,12))

# show the plot
plt.show()


#%%
## Moving windows

# there are a lot of function in pandas to calculate tthe moving winddows such as rolling_mean(), and rolling_std()


#%%
#For example, a rolling mean smoothes out short-term fluctuations and highlight longer-term trends in data.

# isolate the adjusted closeing pricing
adj_close_px = aapl['Adj. Close']  # need to concern the adj. close or adj close

# print(adj_close_px)

moving_avg = adj_close_px.rolling(window=40).mean()

print(moving_avg[-10:])


#%%
# short moving window rolling mean 
aapl['42'] = adj_close_px.rolling(window=40).mean()

# long moving window rolling mean
aapl['252'] = adj_close_px.rolling(window=252).mean()

# plot the adjusted closing price, the short and long windows of rolling means
aapl[['Adj. Close','42','252']].plot()
plt.show()


#%%
# Volatility Calculation


#%%
# the moving historical standard deviation of the log returns—
# i.e. the moving historical volatility—might be more of interest: 
# Also make use of pd.rolling_std(data, window=x) * math.sqrt(window) 
# for the moving historical standard deviation of the log returns (aka the moving historical volatility).


#%%
import matplotlib.pyplot as plt

# define the minumum  of periods to consider
min_periods =75

# Calculate the volatility
vol= daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
# vol_mean = monthly.rolling(min_periods).std() * np.sqrt(min_periods).mean()
# plot the volatility
vol.plot(figsize=(10,8))


# show the plot
plt.show()


#%%
# Ordinary Least-Squares Regression (OLS)


#%%
# Import the `api` model of `statsmodels` under alias `sm`
import statsmodels.api as sm

# Import the `datetools` module from `pandas`
# from pandas.core import datetools
from pandas.core import tools

# Isolate the adjusted closing price
all_adj_close = all_data[['Adj Close']]

# print(all_adj_close)

# Calculate the returns 
all_returns = np.log(all_adj_close / all_adj_close.shift(1))
# print(all_returns)

# Isolate the AAPL returns 
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
aapl_returns.index = aapl_returns.index.droplevel('Ticker')


# Isolate the MSFT returns
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')

# Build up a new DataFrame with AAPL and MSFT returns
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']

# Add a constant 
X = sm.add_constant(return_data['AAPL'])

# Construct the model
model = sm.OLS(return_data['MSFT'],X).fit()



# Print the summarydel.summary())
print(model.summary)


#%%
# # import API model of statsmodels 

# import statsmodels.api as sm

# # import datetools module from pandas
# # from pandas.core import datetools

# # isolate the adjusted closing price
# all_adj_close = all_data[['Adj Close']]

# # print(all_adj_close)

# # Calculate the return 
# all_returns = np.log(all_adj_close/all_adj_close.shift(1))

# # print(all_returns)

# # Isolate the AAPL returns 
# aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
# aapl_returns.index = aapl_returns.index.droplevel('Ticker')

# # Isolate the MSFT returns
# msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
# msft_returns.index = msft_returns.index.droplevel('Ticker')


# # Build up a new dataframe with aapl and msft returns
# return_data = pd.concat([aapl_returns,msft_returns], axis=1)[1:]
# return_data.columns =['AAPL','MSFT']

# # ADD  a constant
# X = sm.add_constant(return_data['AAPL'])

# # construct the model
# model = sm.OLS(return_data['MSFT'],X).fit()

# print(model.summary)


#%%



