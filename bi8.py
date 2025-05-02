import pandas as pd
from datetime import datetime

# -------------------------------
# ✅ E - Extract Data from CSV Files
# -------------------------------

# Read the first CSV file
df1 = pd.read_csv('etl1.csv')

# Read the second CSV file
df2 = pd.read_csv('etl2.csv')

# -------------------------------
# ✅ T - Transform Data
# -------------------------------

# Standardize Date Format in both datasets

# For df1: Convert to datetime (already in YYYY-MM-DD, but let's ensure it's the right format)
df1['Date'] = pd.to_datetime(df1['Date'], format='%Y-%m-%d')

# For df2: Convert to datetime (change from DD/MM/YYYY to YYYY-MM-DD)
df2['Date'] = pd.to_datetime(df2['Date'], format='%d/%m/%Y')

# Select only the common columns: Product, Sales, Region, and the standardized Date
df1_transformed = df1[['Date', 'Product', 'Sales', 'Region']]
df2_transformed = df2[['Date', 'Product', 'Sales', 'Region']]

# -------------------------------
# ✅ L - Load Data
# -------------------------------

# Merge the two datasets (on Date, Product, and Region) to combine the data
merged_df = pd.concat([df1_transformed, df2_transformed], ignore_index=True)

# Display the result (or load it to a CSV, database, etc.)
merged_df.to_csv('merged_sales_data.csv', index=False)
print("Data saved to merged_sales_data.csv")

# Optionally, display the merged DataFrame
print(merged_df.head())
