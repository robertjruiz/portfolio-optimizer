
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
from fredapi import Fred
import matplotlib.pyplot as plt


##tickers = ['SPY','BND','GLD','QQQ','VTI']

input_string = input('Enter tickers sperated by a space: \n')
tickers = input_string.split()


end_date = datetime.today()
start_date = end_date - timedelta(days = 2*365)

print(start_date)

adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start= start_date, end= end_date)
    adj_close_df[ticker] = data['Adj Close']


##print(adj_close_df)

log_returns = np.log(adj_close_df / adj_close_df.shift(1))

log_returns = log_returns.dropna()


cov_matrix = log_returns.cov()*252

##print(cov_matrix)

def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)


def expected_return (weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252


def sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    return(expected_return (weights, log_returns) - risk_free_rate) / standard_deviation (weights, cov_matrix)


fred = Fred(api_key='a4aeaf1e9c87ea07e539b05700bfe551')
ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100


risk_free_rate = ten_year_treasury_rate.iloc[-1]

##print(risk_free_rate)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

max_weight = input('Enter max asset weight in decimal format: ')


constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, max_weight) for _ in range(len(tickers))]


inital_weights = np.array([1/len(tickers)]*len(tickers))

##print(inital_weights)


optimized_results = minimize(neg_sharpe_ratio, inital_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)

optimal_weights = optimized_results.x


print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

print()

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")



plt.figure(figsize=(10, 6))
bars = plt.bar(tickers, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

plt.bar_label(bars)


plt.show()