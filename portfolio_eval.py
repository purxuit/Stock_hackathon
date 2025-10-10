import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

pred = pd.read_csv("./output.csv")
model = "xgb"

# ensure true returns
if "ret_next" in pred.columns:
    pred = pred.rename(columns={"ret_next": "stock_ret"})

# storage for portfolio records
records = []

# === monthly rebalancing ===
for (year, month), df in pred.groupby(["year", "month"]):
    df = df.sort_values(model, ascending=False)
    if len(df) < 200:  # need enough stocks
        continue

    longs = df.head(100).copy()
    shorts = df.tail(100).copy()

    # equal weights
    longs["weight"] =  1/100
    shorts["weight"] = -1/100
    portfolio = pd.concat([longs, shorts])

    # portfolio return = sum(weight * realized return)
    port_ret = (portfolio["weight"] * portfolio["stock_ret"]).sum()

    records.append({
        "year": year,
        "month": month,
        "long_ids": longs["id"].tolist(),
        "short_ids": shorts["id"].tolist(),
        "long_ret": longs["stock_ret"].mean(),
        "short_ret": shorts["stock_ret"].mean(),
        "port_ret": port_ret
    })

monthly_port = pd.DataFrame(records).sort_values(["year","month"])
monthly_port["date"] = pd.to_datetime(monthly_port["year"].astype(str) + "-" + monthly_port["month"].astype(str) + "-01")

# === evaluation ===
r = monthly_port["port_ret"]
sharpe = r.mean()/r.std()*np.sqrt(12)
print(f"📈 Sharpe ratio (annualized): {sharpe:.3f}")

# CAPM
mkt = pd.read_csv("./mkt_ind.csv")  # must have year, month, ret, rf
mkt["mkt_rf"] = mkt["ret"] - mkt["rf"]
df = monthly_port.merge(mkt, on=["year","month"], how="inner")

X = sm.add_constant(df["mkt_rf"])
y = df["port_ret"]
ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags":3})
alpha = ols.params["const"]; beta = ols.params["mkt_rf"]
print(f"✅ Alpha (annualized): {alpha*12:.3%}, Beta: {beta:.2f}")
print(f"t(alpha): {ols.tvalues['const']:.2f}, t(beta): {ols.tvalues['mkt_rf']:.2f}")

# drawdown plot
cum = (1+r).cumprod()
peak = cum.cummax()
drawdown = (peak-cum)/peak
print(f"⚠️ Max 1-month loss: {r.min():.2%}, 📉 Max drawdown: {drawdown.max():.2%}")

# === Portfolio vs Market Performance Comparison ===

# Compute cumulative returns
df["port_cum"] = (1 + df["port_ret"]).cumprod()
df["mkt_cum"]  = (1 + df["ret"]).cumprod()
df["excess_cum"] = (1 + df["port_ret"] - df["ret"]).cumprod()  # optional: relative outperformance

# Plot portfolio vs market
plt.figure(figsize=(9,5))
plt.plot(df["date"], df["port_cum"], label="Model Portfolio (100×100)")
plt.plot(df["date"], df["mkt_cum"], label="Market Index", linestyle="--", color="gray")
plt.title(f"{model.upper()} Portfolio vs Market (Cumulative Return)")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth (×)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Compute relative performance metrics ---
port_ann_ret = (df["port_cum"].iloc[-1]**(12/len(df)) - 1)
mkt_ann_ret  = (df["mkt_cum"].iloc[-1]**(12/len(df)) - 1)
port_vol     = df["port_ret"].std() * np.sqrt(12)
mkt_vol      = df["ret"].std() * np.sqrt(12)
tracking_err = np.std(df["port_ret"] - df["ret"]) * np.sqrt(12)
info_ratio   = (port_ann_ret - mkt_ann_ret) / tracking_err if tracking_err > 0 else np.nan

print(f"🏁 Model annualized return: {port_ann_ret:.2%}")
print(f"📊 Market annualized return: {mkt_ann_ret:.2%}")
print(f"⚙️  Portfolio volatility: {port_vol:.2%}, Market volatility: {mkt_vol:.2%}")
print(f"📈 Tracking error: {tracking_err:.2%}, Information ratio: {info_ratio:.2f}")


# === stock display for latest month ===
latest = monthly_port.sort_values(["year","month"]).iloc[-1]
print("\n📅 Latest Rebalance:", latest["year"], latest["month"])
print("\nTop 10 LONGS:")
print(latest["long_ids"][:10])
print("\nTop 10 SHORTS:")
print(latest["short_ids"][:10])

print("\nAll 100 LONGS:")
print(latest["long_ids"])
print("\nAll 100 SHORTS:")
print(latest["short_ids"])
