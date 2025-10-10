"""
Shift accounting ratios forward by 4 calendar months and merge them
back into the main dataset for predictive modeling (lead fundamentals).
"""

import polars as pl

# --- File paths ---
#replace with ur own paths, this name is different because they apparently updated the data but using just ret_sample.csv should be fine
data_path = r"./ret_sample_update.csv"
ratio_path = r"./acc_ratios.csv"
output_path = r"./final_data.csv"

# --- Load main dataset ---
data = pl.read_csv(
    data_path,
    try_parse_dates=True,
    low_memory=True,
    rechunk=True,
)

# Parse known date columns if needed
for col in ["date", "ret_eom", "char_date", "char_eom"]:
    if col in data.columns and data[col].dtype == pl.Utf8:
        data = data.with_columns(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False))

# --- Load accounting ratio variable list ---
ratio_list = pl.read_csv(ratio_path)["Variable"].to_list()

# --- Keep only identifiers and ratios ---
ratio_cols = ["id", "char_eom"] + ratio_list
ratios = data.select([c for c in ratio_cols if c in data.columns])

# --- Ensure char_eom is parsed as a proper Date first ---
# If it's an integer like 20210331, convert to string then to Date
if ratios["char_eom"].dtype == pl.Int64:
    ratios = ratios.with_columns(
        pl.col("char_eom")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
    )

# Now safely shift by 4 calendar months
ratios = ratios.with_columns(
    pl.col("char_eom").dt.offset_by("4mo").alias("ret_eom")
)


# --- Rename ratio columns to indicate lead4 ---
lead_cols = {col: f"{col}_lead4" for col in ratio_list}
ratios = ratios.rename(lead_cols)

# --- Keep relevant columns for merging ---
ratios = ratios.select(["id", "ret_eom"] + list(lead_cols.values()))

# --- Ensure both join keys ('ret_eom') are Date type ---
if data["ret_eom"].dtype == pl.Int64:
    data = data.with_columns(
        pl.col("ret_eom").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
    )

if ratios["ret_eom"].dtype != data["ret_eom"].dtype:
    ratios = ratios.with_columns(pl.col("ret_eom").cast(data["ret_eom"].dtype))

# --- Merge back with main data (left join ensures no loss of returns) ---
final_data = data.join(ratios, on=["id", "ret_eom"], how="left")

# --- Save output ---
final_data.write_csv(output_path)

print(f"✅ Saved merged dataset with 4-month lead ratios → {output_path}")
print("Final dataset shape:", final_data.shape)