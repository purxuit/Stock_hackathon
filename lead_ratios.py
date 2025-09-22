"""
This code takes a 4-month lead of accounting ratios and adds them back to the original sample.
These lead ratios can be used as predictors for the out-of-sample testing.
"""

import pandas as pd

# Read the original sample dataset
data = pd.read_csv(
    "ret_sample.csv",
    parse_dates=["date", "ret_eom", "char_date", "char_eom"],
    low_memory=False,
)

# Read the list of accounting ratios
ratio_list = pd.read_csv("acc_ratios.csv")
ratio_list = ratio_list["Variable"].tolist()

# Keep the necessary identifiers and ratios
ratios = data[["id", "char_eom"] + ratio_list]

# Move the ratios 4 months ahead for predictive modeling
ratios["char_eom"] = ratios["char_eom"] + pd.DateOffset(months=4)
ratios = ratios.rename(columns={col: col + "_lead4" for col in ratio_list})
ratios = ratios.rename(columns={"char_eom": "ret_eom"})  # Align with the correct date for merging

# Merge the lead ratios with the original data
final_data = pd.merge(data, ratios, on=["id", "ret_eom"], how="left")

# Save the final dataset
final_data.to_csv("final_data.csv", index=False)
