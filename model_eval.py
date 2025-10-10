import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Load predictions ===
pred = pd.read_csv("./output.csv")

# Combine year/month if needed
if "year_month" not in pred.columns:
    if "year" in pred.columns and "month" in pred.columns:
        pred["year_month"] = pd.to_datetime(pred["year"].astype(str) + "-" + pred["month"].astype(str) + "-01")
    else:
        raise KeyError("Missing 'year' and 'month' columns; cannot construct 'year_month'.")

# Ensure year and month exist
if "year" not in pred.columns or "month" not in pred.columns:
    pred["year"] = pred["year_month"].dt.year
    pred["month"] = pred["year_month"].dt.month

# Ensure we’re using real realized returns
if "stock_ret" not in pred.columns:
    if "ret_next" in pred.columns:
        pred = pred.rename(columns={"ret_next": "stock_ret"})
        print("Using 'ret_next' as true next-month realized return.")
    else:
        raise KeyError("Need 'stock_ret' or 'ret_next' column for real evaluation.")

# === 2. Choose model ===
model = "xgb"  # or "ols", "ridge", "lasso", "xgb"
print(f"Analyzing model: {model}")

# === 3. Rank stocks into deciles (safe for small groups) ===
def safe_decile_rank(s):
    n = min(10, s.nunique())  # avoid errors for small sample months
    return pd.qcut(s.rank(method="first"), n, labels=range(1, n + 1))

pred["decile"] = pred.groupby(["year", "month"])[model].transform(safe_decile_rank)

# === 4. Compute monthly returns per decile ===
monthly = (
    pred.groupby(["year", "month", "decile"])["stock_ret"]
        .mean()
        .unstack("decile")
        .sort_index()
)

# Create long-short portfolio: top decile minus bottom decile
monthly["longshort"] = monthly[10] - monthly[1]
longshort = monthly["longshort"].dropna()

# === 5. Sharpe Ratio (annualized) ===
sharpe = longshort.mean() / longshort.std() * np.sqrt(12)
print(f"Sharpe ratio (annualized): {sharpe:.3f}")

# === 6. Load market data ===
mkt = pd.read_csv("./mkt_ind.csv")  # columns: year, month, ret, rf
if not all(col in mkt.columns for col in ["year", "month", "ret", "rf"]):
    raise KeyError("Market data must include 'year', 'month', 'ret', and 'rf' columns.")

mkt["mkt_rf"] = mkt["ret"] - mkt["rf"]

# Merge monthly portfolio with market data
monthly_port = (
    monthly.assign(
        year=monthly.index.get_level_values("year"),
        month=monthly.index.get_level_values("month")
    )
    .reset_index(drop=True)
    .merge(mkt, on=["year", "month"], how="inner")
)

if len(monthly_port) < len(monthly):
    print("Warning: Some months dropped due to missing market data.")

# === 7. CAPM Regression ===
monthly_port["longshort_excess"] = monthly_port["longshort"] - monthly_port["rf"]

X = sm.add_constant(monthly_port["mkt_rf"])
y = monthly_port["longshort_excess"]
nw_ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

alpha_monthly = nw_ols.params["const"]
beta = nw_ols.params["mkt_rf"]
alpha_annual = alpha_monthly * 12

# Print regression results safely
try:
    print(nw_ols.summary())
except ValueError:
    print("Statsmodels summary F-test skipped (robust covariance).")

print(f"Alpha (monthly): {alpha_monthly:.4%} (t={nw_ols.tvalues['const']:.2f})")
print(f"Beta: {beta:.2f} (t={nw_ols.tvalues['mkt_rf']:.2f})")
print(f"Alpha (annualized): {alpha_annual:.3%}")

# === 8. Risk Metrics ===
max_1m_loss = longshort.min()
cum = (1 + longshort).cumprod()
peak = cum.cummax()
drawdown = (peak - cum) / peak
max_drawdown = drawdown.max()

print(f"Max 1-month loss: {max_1m_loss:.2%}")
print(f"Maximum drawdown: {max_drawdown:.2%}")

# === 9. Plot cumulative return vs. market ===
plt.figure(figsize=(8, 4))
(1 + monthly["longshort"]).cumprod().plot(label="Model long-short")
(1 + monthly_port["mkt_rf"]).cumprod().plot(label="Market excess return")
plt.legend()
plt.title(f"{model.upper()} vs Market (Cumulative Return)")
plt.ylabel("Cumulative Growth")
plt.xlabel("Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()