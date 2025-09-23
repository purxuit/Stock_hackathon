import datetime
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Display current time (for timing purposes)
    print(datetime.datetime.now())

    # Set working directory (adjust as necessary)
    work_dir = "C:\\Users\\Naifu\\Desktop\\Hackathon"

    # Read the sample data
    file_path = os.path.join(work_dir, "ret_sample.csv")
    chunk_iter = pd.read_csv(file_path, 
                         parse_dates=["ret_eom"], 
                         low_memory=False, 
                         chunksize=500000, # adjust chunksize
                         nrows=2_000_000) # only look at first 2M rows

    # Example: just look at first chunk
    first_chunk = next(chunk_iter)

    # Or concatenate processed chunks
    data_list = []
    for chunk in chunk_iter:
        # do some preprocessing here (filtering, dropping cols)
        data_list.append(chunk)
    raw = pd.concat(data_list, ignore_index=True)

    # Read the list of stock variables (predictors)
    file_path = os.path.join(work_dir, "factor_char_list.csv")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    # Define the return variable
    ret_var = "stock_ret"
    new_set = raw[raw[ret_var].notna()].copy()  # Filter out missing returns

    # Rank-transform each stock variable monthly
    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(var_median)  # Fill missing values with median
            group[var] = group[var].rank(method="dense") - 1
            group_max = group[var].max()
            if group_max > 0:
                group[var] = (group[var] / group_max) * 2 - 1
            else:
                group[var] = 0  # Handle all missing values
                print(f"Warning: {date} {var} set to zero.")

        # Append the adjusted data
        data = pd.concat([data, group], ignore_index=True)

    # Set initial training start date
    starting = pd.to_datetime("2005-01-01")
    counter = 0
    pred_out = pd.DataFrame()

    # Expanding window backtest loop
    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("2026-01-01"):
        cutoff = [
            starting,
            starting + pd.DateOffset(years=8 + counter),  # 8 years for training
            starting + pd.DateOffset(years=10 + counter),  # 2 years for validation
            starting + pd.DateOffset(years=11 + counter),  # 1 year for testing
        ]

        # Split the dataset into training, validation, and test sets
        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        # Standardize the data
        scaler = StandardScaler().fit(train[stock_vars])
        train[stock_vars] = scaler.transform(train[stock_vars])
        validate[stock_vars] = scaler.transform(validate[stock_vars])
        test[stock_vars] = scaler.transform(test[stock_vars])

        # Prepare training, validation, and test sets
        X_train = train[stock_vars].values
        Y_train = train[ret_var].values
        X_val = validate[stock_vars].values
        Y_val = validate[ret_var].values
        X_test = test[stock_vars].values
        Y_test = test[ret_var].values

        # Demean the returns
        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean

        # Linear regression prediction
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred = test[["year", "month", "ret_eom", "id", ret_var]]
        reg_pred["ols"] = x_pred

        # Lasso Regression
        lambdas = np.arange(-4, 4.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = Lasso(alpha=(10**i), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = Lasso(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["lasso"] = x_pred

        # Ridge Regression
        lambdas = np.arange(-1, 8.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = Ridge(alpha=(10**i * 0.5), fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = Ridge(alpha=(10**best_lambda * 0.5), fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["ridge"] = x_pred

        # ElasticNet Regression
        lambdas = np.arange(-4, 4.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = ElasticNet(alpha=(10**i), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = ElasticNet(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["en"] = x_pred

        # Append predictions
        pred_out = pd.concat([pred_out, reg_pred], ignore_index=True)

        # Move to the next year
        counter += 1

    # Output the predicted values
    pred_out.to_csv("output.csv", index=False)

    # Print OOS R2
    yreal = pred_out[ret_var].values
    for model_name in ["ols", "lasso", "ridge", "en"]:
        ypred = pred_out[model_name].values
        r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
        print(model_name, r2)

    print(datetime.datetime.now())
