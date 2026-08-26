"""
Day 4: Open the Lending Club raw dataset and confirm its structure.

Reads the first 200,000 rows (file is too large to load in full),
then inspects dtypes/missingness, key numeric summary stats, grade/
loan_status distributions, and the issue_d date format.
"""

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

RAW_PATH = "data/raw/accepted_2007_to_2018Q4.csv"

df = pd.read_csv(RAW_PATH, nrows=200000, low_memory=False)

print("=== shape ===")
print(df.shape)

print()
print("=== .info() ===")
df.info(verbose=True, show_counts=True)

print()
print("=== .describe() on key business columns ===")
key_cols = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "fico_range_high",
]
print(df[key_cols].describe())

print()
print("=== grade value counts ===")
print(df["grade"].value_counts().sort_index())

print()
print("=== loan_status value counts ===")
print(df["loan_status"].value_counts())

print()
print("=== issue_d format check ===")
print(df["issue_d"].dropna().head(10).tolist())
print("dtype:", df["issue_d"].dtype)
