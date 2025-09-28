import datetime
import os
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

print(datetime.datetime.now())

# === Settings ===
work_dir = "C:\\Users\\Naifu\\Desktop\\Hackathon"
ret_var = "stock_ret"
start_date = pl.date(2005, 1, 1)
end_date = pl.date(2026, 1, 1)

# === Load predictors list ===
file_path = os.path.join(work_dir, "factor_char_list.csv")
stock_vars = pl.read_csv(file_path)["variable"].to_list()

# === Load raw data with Polars ===
file_path = os.path.join(work_dir, "ret_sample.csv")
raw = pl.read_csv(
    file_path,
    try_parse_dates=True,
    low_memory=True,
    rechunk=True
)
schema_overrides = {var: pl.Float32 for var in stock_vars}
# treat identifiers as Utf8 (string) so they won’t cause dtype mismatches
schema_overrides.update({"gvkey": pl.Utf8, "iid": pl.Utf8, "id": pl.Utf8})

# Add predictor date column (from char_date)
raw = raw.with_columns([
    pl.col("char_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date")
])

# Filter out rows with missing returns
raw = raw.filter(pl.col(ret_var).is_not_null())

# === Cross-sectional rank transform per month ===
def rank_to_unit(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for var in stock_vars:
        median = out[var].median()
        out = out.with_columns(
            pl.when(pl.col(var).is_null()).then(median).otherwise(pl.col(var)).alias(var)
        )
        out = out.with_columns((pl.col(var).rank("dense") - 1).alias(var))
        
        maxv = out[var].max()
        if maxv is None or maxv == 0:
            out = out.with_columns(pl.lit(0).alias(var))
        else:
            out = out.with_columns(((pl.col(var) / maxv) * 2 - 1).alias(var))
    return out


data = raw.group_by("date", maintain_order=True).map_groups(rank_to_unit)

# === Expanding window backtest ===
pred_out = []

counter = 0
while (start_date + datetime.timedelta(days=365*11 + counter*365)) <= end_date:
    cutoff = [
        start_date,
        start_date + datetime.timedelta(days=365*(8 + counter)),   # training end
        start_date + datetime.timedelta(days=365*(10 + counter)),  # validation end
        start_date + datetime.timedelta(days=365*(11 + counter))   # test end
    ]

    # Slice data (still Polars)
    train = data.filter((pl.col("date") >= cutoff[0]) & (pl.col("date") < cutoff[1]))
    validate = data.filter((pl.col("date") >= cutoff[1]) & (pl.col("date") < cutoff[2]))
    test = data.filter((pl.col("date") >= cutoff[2]) & (pl.col("date") < cutoff[3]))

    if train.height == 0 or validate.height == 0 or test.height == 0:
        print("Empty window:", cutoff)
        counter += 1
        continue

    # Convert predictors/target to NumPy
    X_train = train.select(stock_vars).to_numpy()
    Y_train = train[ret_var].to_numpy()
    X_val = validate.select(stock_vars).to_numpy()
    Y_val = validate[ret_var].to_numpy()
    X_test = test.select(stock_vars).to_numpy()
    Y_test = test[ret_var].to_numpy()

    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Demean returns
    Y_mean = Y_train.mean()
    Y_train_dm = Y_train - Y_mean

    # === Models ===
    reg_pred = test.select(["year", "month", "ret_eom", "id", ret_var]).to_pandas()

    # Linear Regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, Y_train_dm)
    reg_pred["ols"] = reg.predict(X_test) + Y_mean

    # Lasso
    lambdas = np.arange(-4, 4.1, 0.1)
    val_mse = []
    for i in lambdas:
        model = Lasso(alpha=10**i, max_iter=1_000_000, fit_intercept=False)
        model.fit(X_train, Y_train_dm)
        val_mse.append(mean_squared_error(Y_val, model.predict(X_val) + Y_mean))
    best_lambda = lambdas[np.argmin(val_mse)]
    reg = Lasso(alpha=10**best_lambda, max_iter=1_000_000, fit_intercept=False)
    reg.fit(X_train, Y_train_dm)
    reg_pred["lasso"] = reg.predict(X_test) + Y_mean

    # Ridge
    lambdas = np.arange(-1, 8.1, 0.1)
    val_mse = []
    for i in lambdas:
        model = Ridge(alpha=(10**i)*0.5, fit_intercept=False)
        model.fit(X_train, Y_train_dm)
        val_mse.append(mean_squared_error(Y_val, model.predict(X_val) + Y_mean))
    best_lambda = lambdas[np.argmin(val_mse)]
    reg = Ridge(alpha=(10**best_lambda)*0.5, fit_intercept=False)
    reg.fit(X_train, Y_train_dm)
    reg_pred["ridge"] = reg.predict(X_test) + Y_mean

    # ElasticNet
    lambdas = np.arange(-4, 4.1, 0.1)
    val_mse = []
    for i in lambdas:
        model = ElasticNet(alpha=10**i, max_iter=1_000_000, fit_intercept=False)
        model.fit(X_train, Y_train_dm)
        val_mse.append(mean_squared_error(Y_val, model.predict(X_val) + Y_mean))
    best_lambda = lambdas[np.argmin(val_mse)]
    reg = ElasticNet(alpha=10**best_lambda, max_iter=1_000_000, fit_intercept=False)
    reg.fit(X_train, Y_train_dm)
    reg_pred["en"] = reg.predict(X_test) + Y_mean

    # Append results
    pred_out.append(reg_pred)
    counter += 1

# === Collect results ===
pred_out = pl.from_pandas(np.concatenate([df.to_numpy() for df in pred_out]))
pred_out.write_csv(os.path.join(work_dir, "output.csv"))

# === Evaluate R2 ===
yreal = pred_out[ret_var].to_numpy()
for model_name in ["ols", "lasso", "ridge", "en"]:
    ypred = pred_out[model_name].to_numpy()
    r2 = 1 - np.sum((yreal - ypred)**2) / np.sum(yreal**2)
    print(model_name, r2)

print(datetime.datetime.now())
