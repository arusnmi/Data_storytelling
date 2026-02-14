import pandas as pd

df = pd.read_csv('AB_US_2023.csv')

# Drop rows with missing values in specific columns
nalessdf = df.dropna(subset=['host_name', 'name'], inplace=False)

dupelessdf = nalessdf.drop_duplicates(subset=['id', 'host_id'], inplace=False)

# after dropping nulls & duplicates, explore numeric columns
numeric = dupelessdf.select_dtypes(include='number')

# compute bounds based on percentiles and filter out extreme values
bounds = {}
for col in numeric.columns:
    low = numeric[col].quantile(0.01)
    high = numeric[col].quantile(0.99)
    bounds[col] = (low, high)

mask = pd.Series(True, index=numeric.index)
for col, (low, high) in bounds.items():
    mask &= numeric[col].between(low, high)

filtered_df = dupelessdf[mask].copy()

# recompute numeric subset after filtering
numeric = filtered_df.select_dtypes(include='number')

# descriptive statistics on filtered data
stats = numeric.agg(['mean', 'median', 'min', 'max', 'std'])
print("Numeric column statistics (filtered):")
print(stats)

# optionally save the filtered dataset
filtered_df.to_csv('AB_US_2023_filtered.csv', index=False)
print("Filtered dataset saved as AB_US_2023_filtered.csv")
print("Visualizations saved. End of script.")




