"""
Finance Hackathon 2025 – Model Training Pipeline
------------------------------------------------
Expanding-window predictive regression for next-month stock returns.
Models: OLS, Lasso, Ridge, ElasticNet, XGBoost
Evaluates out-of-sample rank correlation (Spearman) and R².
"""
print("🚀 Starting model training pipeline...")
# === Imports ===
import os, datetime, warnings
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# === Config ===
warnings.filterwarnings("ignore")
sns.set()
pd.set_option('display.max_columns', None)

work_dir = r"C:\Users\Naifu\Desktop\Finance Hackathon 2025"
ret_var = "stock_ret"
start_date = datetime.date(2005, 1, 1)
end_date   = datetime.date(2026, 1, 1)

print("✅ Imports loaded at", datetime.datetime.now())

# === Define predictors ===
long_vars  = ["f_score","gp_at","op_at","ebitda_mev","qmj_prof","ni_inc8q","ocf_at",
              "fcf_me","taccruals_at","noa_at","inv_gr1","capx_gr1","ret_12_1",
              "ret_6_1","qmj_safety","ni_be","sale_bev","rd_me"]
short_vars = ["ni_ivol","earnings_variability","taccruals_at","inv_gr1","noa_gr1a",
              "capx_gr3","o_score","z_score","betadown_252d","ivol_capm_21d",
              "rmax1_21d","rmax5_21d","ret_1_0","zero_trades_252d","bidaskhl_21d"]

stock_vars = sorted(set(long_vars + short_vars))
cols_keep  = ["char_date","date","gvkey","iid","id","ret_eom","year","month",ret_var] + stock_vars
print(f"Loaded {len(stock_vars)} predictors.")

# === Load & clean data ===
csv_path = r"./final_data.csv"
raw = pl.read_csv(
    csv_path,
    try_parse_dates=True,
    schema_overrides={**{v: pl.Float32 for v in stock_vars},
                      "gvkey": pl.Utf8, "iid": pl.Utf8, "id": pl.Utf8},
    low_memory=True,
)

# Convert char_date to Date type
raw = raw.with_columns(pl.col("char_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date"))
raw = raw.filter(pl.col(ret_var).is_not_null())
cleaned = raw.select([c for c in cols_keep if c in raw.columns]).drop_nulls()
print("✅ Cleaned data shape:", cleaned.shape)

# Align returns to next month (prevent look-ahead bias)
data = cleaned.sort(["id","date"]).with_columns(
    pl.col("stock_ret").shift(-1).over("id").alias("ret_next")
).filter(pl.col("ret_next").is_not_null())

# Fill missing values globally with medians
for v in stock_vars:
    med = data.select(pl.col(v).median()).item()
    data = data.with_columns(
        pl.when(pl.col(v).is_null()).then(pl.lit(med, dtype=pl.Float32)).otherwise(pl.col(v)).alias(v)
    )

# === Expanding-window training ===
pred_frames = []
corr_tracker = {"window": [], "ols": [], "lasso": [], "ridge": [], "en": [], "xgb": []}
counter = 0

max_date = pd.Timestamp(data.select(pl.col("date").max()).item())
safe_end = min(pd.Timestamp(end_date), max_date - pd.Timedelta(days=365))
start = pd.Timestamp(start_date)

while (start + pd.DateOffset(years=11 + counter)) <= pd.Timestamp(end_date):
    cutoff = [
        start,
        start + pd.DateOffset(years=8 + counter),
        start + pd.DateOffset(years=10 + counter),
        start + pd.DateOffset(years=11 + counter),
    ]
    print(f"\n🪟 Window {counter}: Train {cutoff[0].year}-{cutoff[1].year}, "
          f"Val {cutoff[1].year}-{cutoff[2].year}, Test {cutoff[2].year}-{cutoff[3].year}")

    # --- Slice data ---
    train = data.filter((pl.col("date") >= cutoff[0]) & (pl.col("date") < cutoff[1]))
    val   = data.filter((pl.col("date") >= cutoff[1]) & (pl.col("date") < cutoff[2]))
    test  = data.filter((pl.col("date") >= cutoff[2]) & (pl.col("date") < cutoff[3]))

    if min(train.height, val.height, test.height) == 0:
        print("⚠️ Empty window, skipping.")
        counter += 1
        continue

    # --- Convert to numpy ---
    X_train, Y_train = train.select(stock_vars).to_numpy(), train.select(ret_var).to_numpy().ravel()
    X_val,   Y_val   = val.select(stock_vars).to_numpy(),   val.select(ret_var).to_numpy().ravel()
    X_test,  Y_test  = test.select(stock_vars).to_numpy(),  test.select(ret_var).to_numpy().ravel()

    # --- Standardize predictors ---
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    # --- Prepare output frame ---
    reg_pred = test.select(["year","month","ret_eom","id","ret_next",ret_var]).to_pandas()

    # ========== LINEAR MODELS ==========
    ols = LinearRegression(fit_intercept=False).fit(X_train, Y_train)
    reg_pred["ols"] = ols.predict(X_test)

    # Lasso (λ grid search)
    lambdas = np.arange(-4, 4.1, 0.1)
    val_mse = [mean_squared_error(Y_val,
                Lasso(alpha=10**p, max_iter=1_000_000, fit_intercept=False, random_state=42)
                .fit(X_train, Y_train).predict(X_val)) for p in lambdas]
    best = lambdas[np.argmin(val_mse)]
    reg_pred["lasso"] = Lasso(alpha=10**best, max_iter=1_000_000, fit_intercept=False, random_state=42)\
                        .fit(X_train, Y_train).predict(X_test)

    # Ridge
    lambdas = np.arange(-1, 8.1, 0.1)
    val_mse = [mean_squared_error(Y_val,
                Ridge(alpha=(10**p)*0.5, fit_intercept=False, random_state=42)
                .fit(X_train, Y_train).predict(X_val)) for p in lambdas]
    best = lambdas[np.argmin(val_mse)]
    reg_pred["ridge"] = Ridge(alpha=(10**best)*0.5, fit_intercept=False, random_state=42)\
                        .fit(X_train, Y_train).predict(X_test)

    # ElasticNet
    lambdas = np.arange(-4, 4.1, 0.1)
    val_mse = [mean_squared_error(Y_val,
                ElasticNet(alpha=10**p, max_iter=1_000_000, fit_intercept=False, random_state=42)
                .fit(X_train, Y_train).predict(X_val)) for p in lambdas]
    best = lambdas[np.argmin(val_mse)]
    reg_pred["en"] = ElasticNet(alpha=10**best, max_iter=1_000_000, fit_intercept=False, random_state=42)\
                     .fit(X_train, Y_train).predict(X_test)

    # ========== XGBOOST ==========
    xgb = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, Y_train)
    reg_pred["xgb"] = xgb.predict(X_test)

    # ========== EVALUATION ==========
    corr_tracker["window"].append(counter)
    for model_name in ["ols","lasso","ridge","en","xgb"]:
        corr, _ = spearmanr(Y_test, reg_pred[model_name])
        corr_tracker[model_name].append(corr if not np.isnan(corr) else 0)
    print("Spearman rank corr:", {m: round(corr_tracker[m][-1],4) for m in ["ols","lasso","ridge","en","xgb"]})

    pred_frames.append(reg_pred)
    counter += 1

# === Summary of performance ===
plt.figure(figsize=(8,5))
for m in ["ols","lasso","ridge","en","xgb"]:
    plt.plot(corr_tracker["window"], corr_tracker[m], label=m)
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Window #")
plt.ylabel("Out-of-sample Spearman corr")
plt.title("Model Predictive Rank Correlation Over Time")
plt.legend(); plt.tight_layout(); plt.show()

print(f"✅ Finished {counter} expanding windows.")

# === Save all predictions ===
pred_out = pd.concat(pred_frames, ignore_index=True)
pred_out.to_csv(r"./output.csv", index=False)
print("💾 Saved predictions to output.csv")


# === Final R² summary ===
y_real = pred_out[ret_var].to_numpy()
for name in ["ols","lasso","ridge","en","xgb"]:
    y_pred = pred_out[name].to_numpy()
    r2 = r2_score(y_real, y_pred)
    print(f"{name}: R² = {r2:.4f}")
