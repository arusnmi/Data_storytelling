import pandas as pd

df = pd.read_csv('AB_US_2023.csv')

# Drop rows with missing values in specific columns
nalessdf = df.dropna(subset=['host_name', 'name'], inplace=False)

dupelessdf = nalessdf.drop_duplicates(subset=['id', 'host_id'], inplace=False)

# numeric columns to inspect
numeric_cols = ['latitude', 'longitude', 'price', 'minimum_nights',
                'number_of_reviews', 'reviews_per_month',
                'calculated_host_listings_count', 'availability_365',
                'number_of_reviews_ltm']

# drop rows with obviously incorrect values (e.g. negative prices or lat/lon outside valid range)
condition = (
    (dupelessdf['price'] >= 0) &
    (dupelessdf['latitude'].between(-90, 90)) &
    (dupelessdf['longitude'].between(-180, 180))
)
cleaned = dupelessdf[condition].copy()

# compute medians and normalize numeric columns
medians = cleaned[numeric_cols].median()

# store medians for later use if needed
print("Column medians:\n", medians)

# normalize by subtracting median then dividing by IQR to reduce effect of outliers
iqr = cleaned[numeric_cols].quantile(0.75) - cleaned[numeric_cols].quantile(0.25)
normalized = (cleaned[numeric_cols] - medians) / iqr.replace(0, 1)

# attach normalized columns back to dataframe with suffix
for col in numeric_cols:
    cleaned[f"{col}_norm"] = normalized[col]

# identify and drop extreme outliers based on normalized values
# threshold = 10% of the maximum normalized value for each column
max_norm = normalized.abs().max()
thresholds = 0.1 * max_norm

# build boolean mask for rows that are within thresholds for all cols
mask = pd.Series(True, index=cleaned.index)
for col in numeric_cols:
    mask &= cleaned[f"{col}_norm"].abs() <= thresholds[col]

filtered = cleaned[mask].copy()

# final dataset
final_df = filtered

# save cleaned data to a new CSV file
output_path = 'AB_US_2023_cleaned.csv'
final_df.to_csv(output_path, index=False)
print(f"Cleaned data written to {output_path} (rows: {len(final_df)})")




