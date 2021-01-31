import numpy as np
from matplotlib import pyplot as plt


def get_cov_d_m(returns_daily, returns_monthly):
    covariances = {}  # saving all cov, it is a dictionary
    rtd_idx = returns_daily.index
    for i in returns_monthly.index:
        # Mask(use for filter) daily returns for each month and year, and calculate covariance
        mask = (rtd_idx.month == i.month) & (rtd_idx.year == i.year)
        # Use the mask to get daily returns for the current month and year of monthy returns index
        covariances[i] = returns_daily[mask].cov()
    return covariances


def get_portfolio_list_dict(returns_monthly, covariances, number_of_assets, n):
    portfolio_returns, portfolio_volatility, portfolio_weights = {}, {}, {}
    a = [portfolio_returns, portfolio_volatility, portfolio_weights]
    # Get portfolio performances at each month
    for date in sorted(covariances.keys()):
        cov = covariances[date]
        for portfolio in range(n):
            weights = np.random.random(number_of_assets)
            weights /= np.sum(weights)  # /= divides weights by their sum to normalize
            returns = np.dot(weights, returns_monthly.loc[date])
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            portfolio_returns.setdefault(date, []).append(returns)
            portfolio_volatility.setdefault(date, []).append(volatility)
            portfolio_weights.setdefault(date, []).append(weights)
    return a


def get_portfolio_returns_dict(portfolio_list_dict):
    portfolio_returns = portfolio_list_dict[0]
    return portfolio_returns


def get_portfolio_volatility_dict(portfolio_list_dict):
    portfolio_volatility = portfolio_list_dict[1]
    return portfolio_volatility


def get_portfolio_weights_dict(portfolio_list_dict):
    portfolio_weights = portfolio_list_dict[2]
    return portfolio_weights


def plot_latest_efficient_frontier(covariances, portfolio_returns, portfolio_volatility):
    # Get latest date of available data
    date = sorted(covariances.keys())[-1]
    # Plot efficient frontier
    plt.scatter(x=portfolio_volatility[date], y=portfolio_returns[date], alpha=0.3)
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.show()


# Get sharpe ratio

def get_sharpe_ratio(portfolio_returns, portfolio_volatility):
    # Empty dictionaries for sharpe ratios and best sharpe indexes by date
    sharpe_ratio = {}
    # Loop through dates and get sharpe ratio for each portfolio
    for date in portfolio_returns.keys():
        for i, ret in enumerate(portfolio_returns[date]):
            # Divide returns by the volatility for the date and index, i
            sharpe_ratio.setdefault(date, []).append(ret / portfolio_volatility[date][i])
    return sharpe_ratio


def get_max_sharpe_idx(portfolio_returns, sharpe_ratio):
    max_sharpe_idxs = {}
    # Loop through dates and get sharpe ratio for each portfolio
    for date in portfolio_returns.keys():
        # Get the index of the best sharpe ratio for each date
        max_sharpe_idxs[date] = np.argmax(sharpe_ratio[date])
    return max_sharpe_idxs


def get_targets(portfolio_weights, max_sharpe_idxs, ewma_monthly):
    targets = []
    # Create features from price history and targets as ideal portfolio
    for date, ewma in ewma_monthly.iterrows():
        # Get the index of the best sharpe ratio
        best_idx = max_sharpe_idxs[date]
        targets.append(portfolio_weights[date][best_idx])
    targets = np.array(targets)
    return targets


def get_features(ewma_monthly):
    features = []
    # Create features from price history and targets as ideal portfolio
    for date, ewma in ewma_monthly.iterrows():
        features.append(ewma)  # add ewma to features
    features = np.array(features)
    return features


def mark_latest_best_sharpe_ratio(covariances, portfolio_returns, portfolio_volatility, max_sharpe_idxs):
    # Get most recent (current) returns and volatility
    date = sorted(covariances.keys())[-1]
    cur_returns = portfolio_returns[date]
    cur_volatility = portfolio_volatility[date]
    # Plot efficient frontier with sharpe as point
    plt.scatter(x=cur_volatility, y=cur_returns, alpha=0.1, color='blue')
    best_idx = max_sharpe_idxs[date]
    # Place an orange "X" on the point with the best Sharpe ratio
    plt.scatter(x=cur_volatility[best_idx], y=cur_returns[best_idx], marker='x', color='orange')
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.show()
