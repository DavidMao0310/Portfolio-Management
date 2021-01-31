import pandas as pd

import test_MPT_functions as fun

full_df = pd.read_csv('fulldata.csv')
full_df['Date'] = pd.to_datetime(full_df['Date'])
full_df.set_index('Date', inplace=True)
# Resample the full dataframe to monthly timeframe
monthly_df = full_df.resample('BMS').first()
# Calculate daily returns of stocks
returns_daily = full_df.pct_change()
# Calculate monthly returns of the stocks
returns_monthly = monthly_df.pct_change().dropna()

covariances = fun.get_cov_d_m(returns_daily, returns_monthly)
portfolio_list_dict = fun.get_portfolio_list_dict(returns_monthly, covariances, 3, 2000)

portfolio_weights = fun.get_portfolio_weights_dict(portfolio_list_dict)
portfolio_volatility = fun.get_portfolio_volatility_dict(portfolio_list_dict)
portfolio_returns = fun.get_portfolio_returns_dict(portfolio_list_dict)
fun.plot_latest_efficient_frontier(covariances, portfolio_returns, portfolio_volatility)
sharpe_ratio = fun.get_sharpe_ratio(portfolio_returns,portfolio_volatility)
max_sharpe_idxs = fun.get_max_sharpe_idx(portfolio_returns, sharpe_ratio)
ewma_daily = returns_daily.ewm(span=30).mean()
# Resample daily returns to first business day of the month with the first day for that month
ewma_monthly = ewma_daily.resample('BMS').first()
# Shift ewma for the month by 1 month forward so we can use it as a feature for future predictions
ewma_monthly = ewma_monthly.shift(1).dropna()

fun.mark_latest_best_sharpe_ratio(covariances, portfolio_returns, portfolio_volatility, max_sharpe_idxs)

targets = fun.get_targets(portfolio_weights, max_sharpe_idxs, ewma_monthly)
features = fun.get_features(ewma_monthly)
