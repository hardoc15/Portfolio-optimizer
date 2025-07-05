# 📈 Portfolio Optimization Dashboard

An interactive and professional-grade portfolio optimization tool built using **Streamlit** and **CVXPY**. This dashboard lets users input stock tickers, optimize asset allocations based on financial strategies, compare performance with benchmarks like SPY, and export results — all without writing any code.

---

## 🚀 Features

- ✅ **Mean-Variance Optimization**
- 📊 **Sharpe Ratio Maximization**
- 🔁 **Monthly Rebalancing Support**
- 📉 **Benchmarks**: Compare vs SPY, Equal-Weight, and Risk Parity
- 💾 **Download Portfolio Weights** (CSV)
- 💡 **Save Strategy Snapshots**
- 🖨️ **PDF Export Instructions**
- ☁️ **Streamlit Cloud Deployable**

---

## 🛠️ Technologies Used

- Python 3.x  
- [Streamlit](https://streamlit.io)  
- [CVXPY](https://www.cvxpy.org/)  
- [yFinance](https://pypi.org/project/yfinance/)  
- Matplotlib, NumPy, Pandas

---

## 🧮 Optimization Strategies

- **Minimum Variance**: Finds the asset allocation with the lowest portfolio volatility.
- **Maximum Sharpe Ratio**: Allocates capital to maximize risk-adjusted return.

---

## 📥 Installation

```bash
git clone https://github.com/hardoc15/Portfolio-optimizer.git
cd portfolio-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run portfolio_dashboard.py
