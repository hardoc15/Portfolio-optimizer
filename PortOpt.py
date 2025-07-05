import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

# Step 1: Get Data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
raw_data = yf.download(tickers, start="2019-01-01", end="2024-01-01", group_by='ticker', auto_adjust=True)
prices = pd.concat([raw_data[ticker]['Close'] for ticker in tickers], axis=1)
prices.columns = tickers

# Step 2: Calculate Daily Returns
returns = prices.pct_change().dropna()

# Step 3: Annualized Metrics
mu = returns.mean() * 252
cov = returns.cov() * 252

# Step 4: Portfolio Optimization
n = len(tickers)
weights = cp.Variable(n)
risk = cp.quad_form(weights, cov)
constraints = [cp.sum(weights) == 1, weights >= 0]
problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve()

# Step 5: Results
opt_weights = weights.value
print("Optimal Weights:")
for ticker, w in zip(tickers, opt_weights):
    print(f"{ticker}: {w:.2%}")

# Step 6: Backtest
portfolio_returns = (returns @ opt_weights).cumsum()
portfolio_returns.plot(title="Cumulative Return of Optimized Portfolio")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.show()

def optimize_portfolio(returns, mu, cov, strategy="Minimum Variance", rf_rate=0.02):
    n = len(mu)
    w = cp.Variable(n)
    ret = mu.values @ w
    risk = cp.quad_form(w, cov.values)

    if strategy == "Minimum Variance":
        objective = cp.Minimize(risk)
    elif strategy == "Maximum Sharpe Ratio":
        objective = cp.Maximize((ret - rf_rate) / cp.sqrt(risk))

    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value

