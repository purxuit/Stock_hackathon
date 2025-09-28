import os, datetime
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

print(datetime.datetime.now())

# === Settings ===
work_dir = r"C:\Users\Naifu\Desktop\Hackathon"
ret_var = "stock_ret"
# Use *python* date objects, not polars expressions, for cutoffs
start_date = datetime.date(2005, 1, 1)
end_date   = datetime.date(2026, 1, 1)

# === Load predictors list ===
fac_path = os.path.join(work_dir, "factor_char_list.csv")
stock_vars = pl.read_csv(fac_path)["variable"].to_list()

# === Load raw data with Polars (set schema!) ===
csv_path = os.path.join(work_dir, "ret_sample.csv")
raw = pl.read_csv(
    csv_path,
    try_parse_dates=True,
    # IMPORTANT: pass schema_overrides here (not after)
    schema_overrides={
        **{v: pl.Float32 for v in stock_vars},   # predictors as float32
        "gvkey": pl.Utf8, "iid": pl.Utf8, "id": pl.Utf8,  # identifiers as strings
        # leave year/month as whatever the file has; we don't transform them
    },
    low_memory=True,
    rechunk=True,
)

# Add predictor date column (char_date = YYYYMMDD as int/str)
raw = raw.with_columns(
    pl.col("char_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date")
)

# Keep only rows with realized (t+1) return
raw = raw.filter(pl.col(ret_var).is_not_null())

# === Cross-sectional rank transform per month ===
# Use map_groups (not apply), and make sure we always write floats.
def rank_to_unit(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for var in stock_vars:
        # fill NAs with cross-sectional median for the month
        median = out[var].median()
        out = out.with_columns(
            pl.when(pl.col(var).is_null()).then(pl.lit(median, dtype=pl.Float32)).otherwise(pl.col(var)).alias(var)
        )
        # dense rank (1..k) -> 0..(k-1) as float
        out = out.with_columns((pl.col(var).rank(method="dense") - 1).cast(pl.Float32).alias(var))
        # scale to [-1, 1]; guard groups where everything is null/identical
        maxv = out[var].max()
        if maxv is None or maxv == 0:
            out = out.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(var))
        else:
            out = out.with_columns(((pl.col(var) / pl.lit(maxv, dtype=pl.Float32)) * 2.0 - 1.0).alias(var))
    return out

# Group by predictor month and rank-transform within each month
data = raw.group_by("date", maintain_order=True).map_groups(rank_to_unit)

# (Optional) ensure date is python-date compatible for slicing
# Polars Date -> python date via .dt.date() if needed; filtering works with polars Date vs python date
# data schema is already fine since "date" is pl.Date

# === Expanding window backtest ===
pred_frames = []
counter = 0

# Stop automatically before the last year of data, based on what's actually loaded
max_data_date = data.select(pl.col("date").max()).item()
safe_end = min(end_date, (max_data_date - datetime.timedelta(days=365)))  # leave at least 1y for test

while (start_date + datetime.timedelta(days=365*(11 + counter))) <= safe_end:
    cutoff = [
        start_date,
        start_date + datetime.timedelta(days=365*(8  + counter)),  # train end
        start_date + datetime.timedelta(days=365*(10 + counter)),  # val end
        start_date + datetime.timedelta(days=365*(11 + counter)),  # test end
    ]
    print("Window:", cutoff)

    # Slice in Polars
    train = data.filter((pl.col("date") >= cutoff[0]) & (pl.col("date") < cutoff[1]))
    val   = data.filter((pl.col("date") >= cutoff[1]) & (pl.col("date") < cutoff[2]))
    test  = data.filter((pl.col("date") >= cutoff[2]) & (pl.col("date") < cutoff[3]))

    if train.height == 0 or val.height == 0 or test.height == 0:
        print("Empty window, skipping.")
        counter += 1
        continue

    # --- to numpy for sklearn ---
    X_train = train.select(stock_vars).to_numpy()
    Y_train = train.select(ret_var).to_numpy().ravel()
    X_val   = val.select(stock_vars).to_numpy()
    Y_val   = val.select(ret_var).to_numpy().ravel()
    X_test  = test.select(stock_vars).to_numpy()
    Y_test  = test.select(ret_var).to_numpy().ravel()

    # standardize on TRAIN only
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # de-mean Y for fit_intercept=False
    Y_mean = Y_train.mean()
    Y_dm   = Y_train - Y_mean

    # base frame to collect predictions + realized returns
    reg_pred = test.select(["year", "month", "ret_eom", "id", ret_var]).to_pandas()

    # --- OLS ---
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, Y_dm)
    reg_pred["ols"] = reg.predict(X_test) + Y_mean

    # --- Lasso ---
    lambdas = np.arange(-4, 4.1, 0.1)
    val_mse = []
    for p in lambdas:
        m = Lasso(alpha=10**p, max_iter=1_000_000, fit_intercept=False)
        m.fit(X_train, Y_dm)
        val_mse.append(mean_squared_error(Y_val, m.predict(X_val) + Y_mean))
    best = lambdas[int(np.argmin(val_mse))]
    m = Lasso(alpha=10**best, max_iter=1_000_000, fit_intercept=False)
    m.fit(X_train, Y_dm)
    reg_pred["lasso"] = m.predict(X_test) + Y_mean

    # --- Ridge ---
    lambdas = np.arange(-1, 8.1, 0.1)
    val_mse = []
    for p in lambdas:
        m = Ridge(alpha=(10**p)*0.5, fit_intercept=False)
        m.fit(X_train, Y_dm)
        val_mse.append(mean_squared_error(Y_val, m.predict(X_val) + Y_mean))
    best = lambdas[int(np.argmin(val_mse))]
    m = Ridge(alpha=(10**best)*0.5, fit_intercept=False)
    m.fit(X_train, Y_dm)
    reg_pred["ridge"] = m.predict(X_test) + Y_mean

    # --- Elastic Net ---
    lambdas = np.arange(-4, 4.1, 0.1)
    val_mse = []
    for p in lambdas:
        m = ElasticNet(alpha=10**p, max_iter=1_000_000, fit_intercept=False)
        m.fit(X_train, Y_dm)
        val_mse.append(mean_squared_error(Y_val, m.predict(X_val) + Y_mean))
    best = lambdas[int(np.argmin(val_mse))]
    m = ElasticNet(alpha=10**best, max_iter=1_000_000, fit_intercept=False)
    m.fit(X_train, Y_dm)
    reg_pred["en"] = m.predict(X_test) + Y_mean

    pred_frames.append(reg_pred)
    counter += 1

# === Collect & save ===
import pandas as pd
pred_out_pd = pd.concat(pred_frames, ignore_index=True)
pred_out = pl.from_pandas(pred_out_pd)
pred_out.write_csv(os.path.join(work_dir, "output.csv"))

# === Evaluate OOS R^2 (vs zero baseline, per your code) ===
yreal = pred_out.select(ret_var).to_numpy().ravel()
for name in ["ols", "lasso", "ridge", "en"]:
    ypred = pred_out.select(name).to_numpy().ravel()
    r2 = 1 - np.sum((yreal - ypred)**2) / np.sum(yreal**2)
    print(name, r2)

print(datetime.datetime.now())
