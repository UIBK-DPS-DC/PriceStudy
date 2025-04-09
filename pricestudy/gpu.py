import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import re

from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import LinearSegmentedColormap

csv_filename = "data/ec2_prices.csv"


def parse_price(price_str):
    if pd.isna(price_str):
        return 0.0

    s = price_str.strip().lower()

    if "unavailable" in s:
        return 0.0

    # e.g. "$0.0042 hourly"
    if s.startswith("$"):
        s = s[1:]

    parts = s.split()
    try:
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


df = pd.read_csv(csv_filename)
df["Price_Dollar"] = df["On Demand"].apply(parse_price)
df["GPUs_Num"] = df["GPUs"]

df = df[(df["GPUs_Num"] > 0) & (df["Price_Dollar"] > 0)]

df = df[
    df["Name"].str.startswith(
        (
            "M",  # General purpose
            "T",  # General purpose
            "C",  # Compute-optimized
            "R",  # Memory-optimized
            "X",  # Memory-optimized
            "U",  # Memory-optimized
            "z",  # Memory-optimized
            "P",  # Accelerated computing
            "G",  # Accelerated computing
        )
    )
    & ~df["Name"].str.startswith("Mac")
]

print(f"Number of rows: {df.shape[0]}")


# Outlier removal
def remove_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data[column] >= lower) & (data[column] <= upper)]


df = remove_outliers(df, "Price_Dollar")

average_price = df["Price_Dollar"].mean()
print(f"Average Price: ${average_price:.4f} ($/h)")

pearson_corr, p_pearson = pearsonr(df["Price_Dollar"], df["GPUs_Num"])
spearman_corr, _ = spearmanr(df["Price_Dollar"], df["GPUs_Num"])

print(f"Pearson r = {pearson_corr:.3f} (p = {p_pearson:.2e})")
print(f"Spearman r = {spearman_corr:.3f}")

X = df["GPUs_Num"]
y = df["Price_Dollar"]
model = sm.OLS(y, X).fit()

print(model.summary())

sns.set_theme(style="whitegrid")

# Define palette colors for lines and hist
dark_green = "#5ba300"
light_green = "#89ce00"
blue = "#0073e6"
pink = "#e6308a"
dark_pink = "#b51963"

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# 1) Correlation matrix
corr_matrix = df[["GPUs_Num", "Price_Dollar"]].corr()

# Define the label mapping for both rows and columns
label_mapping = {"Price_Dollar": "Price ($/h)", "GPUs_Num": "GPUs"}

# Rename the index and columns in the correlation matrix
corr_matrix = corr_matrix.rename(index=label_mapping, columns=label_mapping)

# 2) Regression plot
sns.regplot(
    x="GPUs_Num",
    y="Price_Dollar",
    data=df,
    ax=axes[0],
    scatter_kws={"color": blue},
    line_kws={"color": pink},
)
axes[0].set_xlabel("GPUs")
axes[0].set_ylabel("Price ($/h)")

# Pearson correlation
pearson_sto, _ = pearsonr(df["Price_Dollar"], df["GPUs_Num"])
axes[0].text(
    0.95,
    0.05,
    f"r={pearson_sto:.3f}",
    ha="right",
    va="bottom",
    transform=axes[0].transAxes,
    color=dark_pink,
)

# 3) Residuals histogram
sns.histplot(
    model.resid,
    color=blue,
    kde=False,
    stat="density",
    ax=axes[1],
)
sns.kdeplot(
    model.resid,
    color=pink,
    ax=axes[1],
)
axes[1].set_xlabel("Residuals")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
