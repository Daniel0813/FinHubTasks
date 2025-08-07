import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mcal

# setting up constants, directories and file paths
TICKERS = ['MSFT', 'JNJ', 'WMT', 'KO', 'MCD', 'HD', 'PG', 'INTC', 'V', 'XOM']
START = '2021-12-01'
END = '2025-07-01'
RAW_DIR = 'rawdata'
DATA_DIR = 'data'
FIG_DIR = 'figures'
LOG_PATH = 'logs/run_portfolio.log'
END_DATE_TAG = '20250630'

# create directories if they do not exist
def setup_directories():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

# download yahoo tickers into dataframe 'data'
def download_monthly_prices(tickers, start, end):
    data = yf.download(
        tickers=tickers, # for each of the tickers e.g. MSFT we have a table with columns for Open, High, Low, Close, Volume, and Adj Close
        start=start,
        end=end,
        interval='1mo',
        auto_adjust=True #auto-replaces close prices with adjusted close prices, so 'Close' is already adjusted
    )
    # to use 'Adj Close' we need to set auto_adjust=False instead, then select the 'Adj Close' column
    return data['Close']  # selects the Adjusted Close price column which is adjusted for dividends and stock splits, just in case

# save the raw data 
def save_raw_prices(data, tickers):
    for ticker in tickers:
        # data[ticker] returns a Series (a single column), whereas data[[ticker]] returns a DataFrame (single column but 2D), cleaner to save as CSV
        # remove rows containing NaN values before saving using dropna()
        ticker_data = data[[ticker]].dropna() 
        # save each ticker's data to a CSV file under the rawdata directory e.g. rawdata/MSFT_prices_20250630.csv
        ticker_data.to_csv(f'{RAW_DIR}/{ticker}_prices_{END_DATE_TAG}.csv')

# Simple returns calculation
def compute_monthly_returns(data, tickers):
    # pct_change() computes the percentage change between a row and its previous row, which is the simple return for each month: (P_t / P_{t-1}) - 1
    returns = data.pct_change().dropna() # drop the first row with NaN values for January 2022 because we don't have the previous month price
    # rename columns to 'ret_TICKER' for clarity e.g. ret_MSFT
    returns.columns = [f'ret_{t}' for t in tickers] 
    return returns

# Equal-weighted w_{i,t} = 1/n: rebalance annually in January
def compute_equal_weighted_returns(returns):
    # we only want to include the return columns, even though ew and vw are initially not there, they may ne added later and we don't want to include them in the weights calculation
    ret_tickers = [col for col in returns.columns if col.startswith("ret_") and col not in ['ret_ew', 'ret_vw']] 
    returns_only = returns[ret_tickers]

    # create an empty DataFrame to store equal weights, index are dates(same as returns_only) and columns are the return tickers (ret_tickers)
    ew_weights = pd.DataFrame(index=returns_only.index, columns=ret_tickers)

    # loop through each unique year in the returns_only DataFrame, update the weight every year
    for year in returns_only.index.year.unique(): 
        # get the first date of that year, which is January
        jan_start = returns_only.loc[str(year)].index[0] 
        # set the weights of all months from January onwards to w_{i,t} = 1/n
        ew_weights.loc[returns_only.index >= jan_start] = 1 / len(ret_tickers) #.loc accesses or modifies rows and/or columns by label

    # apply the formula ret_ew = sum(w_{i,t} * ret_{i,t}) across each row (axis=1)
    ret_ew = (returns_only * ew_weights).sum(axis=1)

    # add results to returns DataFrame
    return ret_ew

# Value-weighted: weights based on price at previous month
def compute_value_weighted_returns(returns, data):
    # Filter returns to only include stock returns (exclude portfolio returns if they exist)
    ret_tickers = [col for col in returns.columns if col.startswith("ret_") and col not in ['ret_ew', 'ret_vw']]
    returns_only = returns[ret_tickers]
    
    # create a DataFrame with the lagged prices (previous month prices)
    lagged_prices = data.shift(1)

    # divide each stock price by total of stock prices for that month to get corresponding weights
    # for this dataframe, sum of the rows equals 1
    vw_weights = lagged_prices.div(lagged_prices.sum(axis=1), axis=0).reindex(returns_only.index)

    # calculate vw_return per month: multiplies each stock's return by its weight (row-wise), then sums
    # element wise multiplication of two matrices with dimension (n_months, n_stocks), result is n_months
    ret_vw = (returns_only.values * vw_weights.values).sum(axis=1)
    # convert ret_vw to a pandas Series with the same index as returns
    ret_vw = pd.Series(ret_vw, index=returns_only.index)

    return ret_vw

# compute max drawdown of return series
# key risk metric, worst drop from a peak in the cumulative return series 
def max_drawdown(series):
    # Converts monthly simple returns to cumulative portfolio value, how much 1 dollar invested at the start grows
    # e.g. series = pd.Series([0.05, -0.03, 0.02], index=pd.to_datetime(['2022-01-31', '2022-02-28', '2022-03-31']))
    # calling (1 + series).cumprod() yields [2022-01-31: 1.050000, 2022-02-28: 1.018500, 2022-03-31: 1.039870]
    cumulative = (1 + series).cumprod() 
    # peak is the running highest value reached reached so far
    peak = cumulative.cummax()
    # drawdown is the percentage drop from the peak
    drawdown = (cumulative - peak) / peak
    # most negative/largest drop is the max drawdown
    return drawdown.min()

# Plot cumulative returns for both portfolios
def plot_cumulative_returns(ret_ew, ret_vw):
    # running cumulative returns for each portfolio
    n = 1 # initial investment of $1
    cumret_ew = n * (1 + ret_ew).cumprod()
    cumret_vw = n * (1 + ret_vw).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cumret_ew, label='Equal Weighted')
    plt.plot(cumret_vw, label='Value Weighted')
    plt.title('Cumulative Return of $1 Invested (Jan 2022 - Jun 2025)')
    plt.ylabel('Portfolio Value')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/portfolio_cumulret.svg')

def build_monthly_series_from_daily(tickers, start_date, end_date): #TODO doesn't seem to work, e.g. MSTF March 2024 missing
    # download daily adjusted close prices
    daily_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval='1d', # daily data
        auto_adjust=True
    )['Close'] # same as download monthly prices, auto_adjust=True replaces close prices with adjusted close prices

    # get NYSE calendar and month-end trading days
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    eom_dates = schedule.resample('ME').last().index # 'M' is deprecated and will be removed in a future version, please use 'ME' instead

    # filter daily prices at month-end dates
    monthly_data = daily_data.loc[daily_data.index.isin(eom_dates)]

    return monthly_data

# Main function to execute the portfolio construction
def main(use_custom_monthly=False):
    # setup directories
    setup_directories()
    
    # download monthly prices and save raw data
    # if use_custom_monthly is True, build monthly series from daily data using actual NYSE trading days
    if use_custom_monthly: 
        data = build_monthly_series_from_daily(TICKERS, START, END)
    else:
        data = download_monthly_prices(TICKERS, START, END)
    save_raw_prices(data, TICKERS)

    # compute monthly simple returns, equal-weighted returns, and value-weighted returns
    returns = compute_monthly_returns(data, TICKERS)
    ret_ew = compute_equal_weighted_returns(returns)
    ret_vw = compute_value_weighted_returns(returns, data)

    # combine returns and save
    returns['ret_ew'] = ret_ew
    returns['ret_vw'] = ret_vw
    # date is a index column, convert it to a regular column for saving into .parquet
    returns.reset_index(inplace=True)
    # Format 'date' column as YYYYMMDD string if required
    returns['Date'] = returns['Date'].dt.strftime('%Y%m%d')
    # save the returns DataFrame to a parquet file in /data directory
    returns.to_parquet(f'{DATA_DIR}/portfolio_returns_{END_DATE_TAG}.parquet', index=False)

    # Compute mean, std, and max drawdown for both portfolios
    mean_ew = ret_ew.mean()
    std_ew = ret_ew.std()
    dd_ew = max_drawdown(ret_ew)

    mean_vw = ret_vw.mean()
    std_vw = ret_vw.std()
    dd_vw = max_drawdown(ret_vw)

    # Plot and save
    plot_cumulative_returns(ret_ew, ret_vw)

    # Logging
    with open(LOG_PATH, 'a', encoding='utf-8') as log:
        log.write(f"[{datetime.now()}] Portfolio construction log\n")
        log.write(f"Tickers: {', '.join(TICKERS)}\n")
        log.write(f"Date Range: {START} to {END}\n")
        log.write(f"Monthly Observations: {returns.shape[0]}\n\n")
        log.write(f"Equal-Weighted Portfolio:\n")
        log.write(f"  Mean Monthly Return: {mean_ew:.4f}\n")
        log.write(f"  Std Dev: {std_ew:.4f}\n")
        log.write(f"  Max Drawdown: {dd_ew:.4f}\n\n")
        log.write(f"Value-Weighted Portfolio:\n")
        log.write(f"  Mean Monthly Return: {mean_vw:.4f}\n")
        log.write(f"  Std Dev: {std_vw:.4f}\n")
        log.write(f"  Max Drawdown: {dd_vw:.4f}\n")
        log.write("="*40 + "\n\n") # visual separator 

if __name__ == "__main__":
    main()
