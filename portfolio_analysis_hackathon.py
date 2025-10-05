import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# === Load predictions ===
pred = pd.read_csv("../output.csv")

# Combine year/month if needed
if "year_month" not in pred.columns:
    pred["year_month"] = pd.to_datetime(pred["year"].astype(str) + "-" + pred["month"].astype(str) + "-01")

# Ensure we’re using real returns
if "stock_ret" not in pred.columns:
    if "ret_next" in pred.columns:
        pred = pred.rename(columns={"ret_next": "stock_ret"})
        print("Using 'ret_next' as true next-month realized return.")
    else:
        raise KeyError("Need 'ret_next' column for real evaluation.")

# === 3. Choose model ===
model = "ridge"   # or "ols", "ridge", "lasso", "xgb"
print(f"Analyzing model: {model}")
# === 4. Rank stocks into deciles ===
pred["decile"] = pred.groupby(["year", "month"])[model] \
                     .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=range(1, 11)))

# === 5. Compute monthly returns per decile ===
monthly = (pred.groupby(["year", "month", "decile"])["stock_ret"]
              .mean()
              .unstack("decile")
              .sort_index())

monthly["longshort"] = monthly[10] - monthly[1]
longshort = monthly["longshort"].dropna()

# === 6. Sharpe Ratio ===
sharpe = longshort.mean() / longshort.std() * np.sqrt(12)
print(f"Sharpe ratio (annualized): {sharpe:.3f}")

# === 7. CAPM Alpha/Beta ===
mkt = pd.read_csv("../mkt_ind.csv")  # columns: year, month, ret, rf
mkt["mkt_rf"] = mkt["ret"] - mkt["rf"]

monthly_port = (monthly
    .assign(year=monthly.index.get_level_values("year"),
            month=monthly.index.get_level_values("month"))
    .reset_index(drop=True)
    .merge(mkt, on=["year", "month"], how="inner")
)

X = sm.add_constant(monthly_port["mkt_rf"])
y = monthly_port["longshort"]
nw_ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

alpha_monthly = nw_ols.params["const"]
beta = nw_ols.params["mkt_rf"]
alpha_annual = alpha_monthly * 12

try:
    print(nw_ols.summary())
except ValueError:
    print("Statsmodels summary F-test skipped (robust covariance).")
    print(f"Alpha (monthly): {alpha_monthly:.4f}, Beta: {beta:.2f}")
    print(f"t(alpha): {nw_ols.tvalues['const']:.2f}, t(beta): {nw_ols.tvalues['mkt_rf']:.2f}")

print(f"Alpha (annualized): {alpha_annual:.3%}, Beta: {beta:.2f}")


# === 8. Risk Metrics ===
max_1m_loss = longshort.min()
cum = (1 + longshort).cumprod()
peak = cum.cummax()
drawdown = (peak - cum) / peak
max_drawdown = drawdown.max()

print(f"Max 1-month loss: {max_1m_loss:.2%}")
print(f"Maximum drawdown: {max_drawdown:.2%}")

# === 9. Plot cumulative return ===
plt.figure(figsize=(8,4))
(1 + monthly["longshort"]).cumprod().plot(label="Model long-short")
(1 + monthly_port["mkt_rf"]).cumprod().plot(label="Market excess return")
plt.legend()
plt.title(f"{model.upper()} vs Market (Cumulative)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


