import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from pandas.tseries.offsets import MonthBegin

# Read predicted values (adjust path to your predicted values)
pred_path = "output.csv"
pred = pd.read_csv(pred_path, parse_dates=[["year","month"]])
pred["year"] = pred["year_month"].dt.year
pred["month"] = pred["year_month"].dt.month
# Select the model you want to evaluate
model = "ridge"

# Rank stocks into deciles based on predicted returns
predicted = pred.groupby(["year", "month"])[model]

pred["rank"] = np.floor(
    predicted.transform(lambda s: s.rank()) * 10 / predicted.transform(lambda s: len(s) + 1)
)

# Sort stocks based on ranks
pred = pred.sort_values(["year", "month", "rank", "id"])

# Calculate realized returns for each portfolio (decile)
monthly_port = pred.groupby(["year", "month", "rank"]).apply(
    lambda df: pd.Series(np.average(df["stock_ret"], axis=0))
)
monthly_port = monthly_port.unstack().dropna().reset_index()
monthly_port.columns = ["year", "month"] + ["port_" + str(x) for x in range(1, 11)]
monthly_port["port_11"] = monthly_port["port_10"] - monthly_port["port_1"]

# Calculate the Sharpe Ratio for the long-short portfolio
sharpe = monthly_port["port_11"].mean() / monthly_port["port_11"].std() * np.sqrt(12)
print("Sharpe Ratio:", sharpe)

# Calculate the CAPM Alpha for the long-short portfolio
mkt_path = "..\\mkt_ind.csv"
mkt = pd.read_csv(mkt_path)
mkt["mkt_rf"] = mkt["ret"] - mkt["rf"]
monthly_port = monthly_port.merge(mkt, how="inner", on=["year", "month"])

# Newey-West regression for heteroskedasticity and autocorrelation robust standard errors
nw_ols = sm.ols(formula="port_11 ~ mkt_rf", data=monthly_port).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
print(nw_ols.summary())

# Max one-month loss
max_1m_loss = monthly_port["port_11"].min()
print("Max 1-Month Loss:", max_1m_loss)

# Calculate Maximum Drawdown
monthly_port["log_port_11"] = np.log(monthly_port["port_11"] + 1)
monthly_port["cumsum_log_port_11"] = monthly_port["log_port_11"].cumsum()
rolling_peak = monthly_port["cumsum_log_port_11"].cummax()
drawdowns = rolling_peak - monthly_port["cumsum_log_port_11"]
max_drawdown = drawdowns.max()
print("Maximum Drawdown:", max_drawdown)
