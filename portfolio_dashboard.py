import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from datetime import datetime

st.set_page_config(page_title="Portfolio Optimizer Pro", layout="wide")

# -------------------------
# 🎯 Sidebar: User Inputs
# -------------------------
with st.sidebar:
    st.title("⚙️ Portfolio Settings")
    tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,META").upper().split(',')
    start_date = st.date_input("Start Date", pd.to_datetime("2019-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    strategy = st.radio("Optimization Strategy", ["Minimum Variance", "Maximum Sharpe Ratio"])
    rf_rate = st.slider("Risk-Free Rate (Annualized)", 0.0, 0.10, 0.02, step=0.005)
    rebalance = st.checkbox("Enable Monthly Rebalancing", value=True)

# -------------------------
# 📥 Data Loading
# -------------------------
st.title("📈 Portfolio Optimization Dashboard")

try:
    prices = yf.download(tickers + ["SPY"], start=start_date, end=end_date, auto_adjust=True)["Close"]
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    spy_returns = returns["SPY"]
    asset_returns = returns[tickers]
    mu = asset_returns.mean() * 252
    cov = asset_returns.cov() * 252
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

# -------------------------
# 🔧 Optimizer
# -------------------------
def optimize(mu, cov, strategy, rf_rate):
    n = len(mu)
    w = cp.Variable(n)
    ret = mu.values @ w
    risk = cp.quad_form(w, cov.values)

    if strategy == "Minimum Variance":
        objective = cp.Minimize(risk)
    elif strategy == "Maximum Sharpe Ratio":
        objective = cp.Maximize((ret - rf_rate) / cp.sqrt(risk))

    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return np.round(w.value, 4)

# -------------------------
# 🧠 Benchmarks
# -------------------------
def get_equal_weight(n):
    return np.array([1 / n] * n)

def get_risk_parity_weights(cov):
    inv_vol = 1 / np.sqrt(np.diag(cov))
    return inv_vol / inv_vol.sum()

# -------------------------
# 📅 Rebalancing / Static
# -------------------------
if rebalance:
    weights_over_time = []
    dates = asset_returns.resample('M').first().dropna().index
    for date in dates:
        window_data = asset_returns.loc[:date].tail(252)
        if len(window_data) < 60:
            continue
        mu_rolling = window_data.mean() * 252
        cov_rolling = window_data.cov() * 252
        weights = optimize(mu_rolling, cov_rolling, strategy, rf_rate)
        weights_over_time.append(pd.Series(weights, index=tickers, name=date))
    weights_df = pd.DataFrame(weights_over_time)
    weights_df = weights_df.reindex(asset_returns.index, method='ffill')
    portfolio_returns = (asset_returns * weights_df).sum(axis=1)
else:
    opt_weights = optimize(mu, cov, strategy, rf_rate)
    weights_df = pd.DataFrame({'Ticker': tickers, 'Weight': opt_weights})
    portfolio_returns = (asset_returns @ opt_weights)

# -------------------------
# 📊 Show & Export Weights
# -------------------------
st.subheader("📌 Optimized Portfolio Weights")
latest_weights = weights_df.dropna().iloc[-1] if rebalance else weights_df.set_index("Ticker")["Weight"]
display_df = pd.DataFrame(latest_weights).reset_index()
display_df.columns = ['Ticker', 'Weight']
st.dataframe(display_df.set_index('Ticker').style.highlight_max(axis=0, color='lightgreen'))

# 📤 Export CSV
csv = display_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Export Weights as CSV", csv, "portfolio_weights.csv", "text/csv")

# -------------------------
# 📈 Cumulative Performance
# -------------------------
st.subheader("📈 Cumulative Return Comparison")
cumulative_returns = pd.DataFrame({
    'Optimized': (1 + portfolio_returns).cumprod(),
    'SPY': (1 + spy_returns).cumprod(),
    'Equal Weight': (1 + asset_returns @ get_equal_weight(len(tickers))).cumprod(),
    'Risk Parity': (1 + asset_returns @ get_risk_parity_weights(cov)).cumprod()
})

fig, ax = plt.subplots()
cumulative_returns.plot(ax=ax, linewidth=2)
ax.set_title("Growth of $1 - Portfolio vs Benchmarks")
ax.set_ylabel("Cumulative Return")
ax.grid(True, linestyle="--", alpha=0.6)
st.pyplot(fig)

# -------------------------
# 📐 Metrics
# -------------------------
st.subheader("📊 Portfolio Performance Metrics")
excess_return = portfolio_returns.mean() * 252 - rf_rate
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = excess_return / volatility

col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{portfolio_returns.mean() * 252:.2%}")
col2.metric("Volatility", f"{volatility:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

# -------------------------
# 💾 Session History
# -------------------------
st.subheader("🕒 Saved Strategies")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("💾 Save Current Strategy"):
    snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers": tickers,
        "strategy": strategy,
        "rf_rate": rf_rate,
        "rebalance": rebalance,
        "weights": display_df.to_dict(orient='records')
    }
    st.session_state.history.append(snapshot)

for strat in st.session_state.history[::-1]:
    with st.expander(f"Saved on {strat['timestamp']}"):
        st.write(f"Tickers: {', '.join(strat['tickers'])}")
        st.write(f"Strategy: {strat['strategy']} | Rebalance: {strat['rebalance']}")
        st.dataframe(pd.DataFrame(strat['weights']).set_index('Ticker'))

# -------------------------
# 📄 PDF Export Tip
# -------------------------
st.markdown("---")
st.subheader("📄 Export Dashboard to PDF")
st.markdown("""
Click your browser’s **File > Print > Save as PDF** or press `Cmd+P` / `Ctrl+P` and choose **Save as PDF** to export this dashboard.
""")

# -------------------------
# 🌐 Footer
# -------------------------
st.markdown("---")
st.markdown("Made with ❤️ by [Your Name] • Powered by Streamlit, Yahoo Finance, CVXPY")
