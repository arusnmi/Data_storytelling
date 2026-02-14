import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# load cleaned data file (produced by DATA CLEANING.py)
INPUT = "AB_US_2023_cleaned.csv"

def main():
    df = pd.read_csv(INPUT)

    # 1. descriptive insights for selected columns
    cols = ["price", "number_of_reviews", "availability_365"]
    desc = df[cols].agg(["mean", "median", "min", "max"]).T
    print("\nDescriptive statistics:")
    print(desc)

    # 2. Explore distributions
    plt.figure(figsize=(10, 4))
    sns.histplot(df["price"], bins=50, kde=False)
    plt.title("Price distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("hist_price.png")
    plt.clf()

    plt.figure(figsize=(10, 4))
    sns.histplot(df["reviews_per_month"], bins=50, kde=False)
    plt.title("Reviews per Month distribution")
    plt.xlabel("Reviews per month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("hist_reviews_per_month.png")
    plt.clf()

    # 3. identify neighbourhood patterns
    loc_col = "neighbourhood_group" if "neighbourhood_group" in df.columns else "city"
    counts = df[loc_col].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Listings per {loc_col}")
    plt.ylabel("Number of listings")
    plt.tight_layout()
    plt.savefig("bar_listings_location.png")
    plt.clf()

    # average price by location to spot unusual cases
    avg_price = df.groupby(loc_col)["price"].mean().sort_values(ascending=False)
    print("\nAverage price by", loc_col)
    print(avg_price.head(10))

    # comparison by room type
    if "room_type" in df.columns:
        room_counts = df["room_type"].value_counts()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=room_counts.index, y=room_counts.values)
        plt.title("Listings by room type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("bar_room_type.png")
        plt.clf()

        # proportion of entire home vs others
        print("\nRoom type distribution:\n", room_counts)

    print("\nEDA complete. Plots saved to current directory.")


if __name__ == "__main__":
    main()
