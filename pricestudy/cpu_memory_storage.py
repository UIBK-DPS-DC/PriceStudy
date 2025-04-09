from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm

import re

# Load the data
csv_filename = "data/ec2_prices.csv"

df = pd.read_csv(csv_filename)


def parse_vcpus(vcpu_str):
    if pd.isna(vcpu_str):
        return np.nan

    # e.g. "1 vCPUs                                                         for a                   1h 12m                   burst"
    vcpu_str = vcpu_str.strip().lower()
    match = re.search(r"(\d+(\.\d+)?)\s*vcpus?", vcpu_str)
    if match:
        return float(match.group(1))

    return np.nan


def parse_memory(mem_str):
    if pd.isna(mem_str):
        return np.nan

    # e.g. "0.613 GiB"
    mem_str = mem_str.strip().lower()
    parts = mem_str.split()
    if len(parts) >= 2 and parts[1].startswith("gib"):
        return float(parts[0])

    return np.nan


def parse_storage(storage_str):
    if pd.isna(storage_str):
        return np.nan

    storage_str = storage_str.strip().lower()
    if "nvme ssd" not in storage_str:
        return 0.0

    # e.g. "59 GB                 NVMe SSD"
    match = re.search(r"(\d+(\.\d+)?)\s*gb", storage_str)
    if match:
        return float(match.group(1))
    else:
        return np.nan


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


# Parse the values
df["vCPUs_Num"] = df["vCPUs"].apply(parse_vcpus)
df["GPUs_Num"] = df["GPUs"]
df["Memory_GiB"] = df["Instance Memory"].apply(parse_memory)
df["Storage_GB"] = df["Instance Storage"].apply(parse_storage)
df["Price_Dollar"] = df["On Demand"].apply(parse_price)

df = df[(df["GPUs_Num"] == 0) & (df["Storage_GB"] > 0) & (df["Price_Dollar"] > 0)]

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
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


df = remove_outliers(df, "Storage_GB")
df = remove_outliers(df, "Memory_GiB")
df = remove_outliers(df, "vCPUs_Num")
df = remove_outliers(df, "Price_Dollar")

average_price = df["Price_Dollar"].mean()
print(f"Average Price: ${average_price:.4f} ($/h)")

# Correlation analysis
price_col = "Price_Dollar"

# Make sure columns exist
required_cols = [price_col, "vCPUs_Num", "Memory_GiB", "Storage_GB"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataframe!")

# Basic correlation matrix (Pearson)
corr_matrix = df[required_cols].corr()
print("Correlation matrix (Pearson):")
print(corr_matrix, "\n")

# Perform linear regression to derive per-unit pricing
X = df[["vCPUs_Num", "Memory_GiB", "Storage_GB"]]
y = df[price_col]  # target

# Fit ordinary least squares regression
model = sm.OLS(y, X).fit()

print("OLS Regression Results:")
print(model.summary())

# Use a clean style
sns.set_theme(style="whitegrid")

# Define palette colors for lines and hist
dark_green = "#5ba300"
light_green = "#89ce00"
blue = "#0073e6"
pink = "#e6308a"
dark_pink = "#b51963"

# For the correlation heatmap, create a custom colormap that goes from white to blue
heatmap_cmap = LinearSegmentedColormap.from_list(
    "heatmap_cmap", ["#ffffff", blue], N=256
)

fig, axes = plt.subplots(1, 4, figsize=(25, 5))

# Define the label mapping for both rows and columns
label_mapping = {
    "Price_Dollar": "Price ($/h)",
    "vCPUs_Num": "vCPUs",
    "Memory_GiB": "Memory (GiB)",
    "Storage_GB": "Storage (GB)",
}

# Rename the index and columns in the correlation matrix
corr_matrix = corr_matrix.rename(index=label_mapping, columns=label_mapping)

# 2) Regression plot (vCPUs vs Price)
sns.regplot(
    x=df["vCPUs_Num"],
    y=df[price_col],
    scatter_kws={"color": blue},  # scatter points
    line_kws={"color": pink},  # regression line
    ax=axes[0],
)
axes[0].set_xlabel("vCPUs")
axes[0].set_ylabel("Price ($/h)")

# Pearson correlation
pearson_cpu, _ = pearsonr(df[price_col], df["vCPUs_Num"])
axes[0].text(
    0.95,
    0.05,
    f"r={pearson_cpu:.3f}",
    ha="right",
    va="bottom",
    transform=axes[0].transAxes,
    color=dark_pink,
)

# 3) Regression plot (Memory vs Price)
sns.regplot(
    x=df["Memory_GiB"],
    y=df[price_col],
    scatter_kws={"color": blue},
    line_kws={"color": pink},
    ax=axes[1],
)
axes[1].set_xlabel("Memory (GiB)")
axes[1].set_ylabel("Price ($/h)")

# Pearson correlation
pearson_mem, _ = pearsonr(df[price_col], df["Memory_GiB"])
axes[1].text(
    0.95,
    0.05,
    f"r={pearson_mem:.3f}",
    ha="right",
    va="bottom",
    transform=axes[1].transAxes,
    color=dark_pink,
)

# 4) Regression plot (Storage vs Price)
sns.regplot(
    x=df["Storage_GB"],
    y=df[price_col],
    scatter_kws={"color": blue},
    line_kws={"color": pink},
    ax=axes[2],
)
axes[2].set_xlabel("Storage (GB)")
axes[2].set_ylabel("Price ($/h)")

# Pearson correlation
pearson_sto, _ = pearsonr(df[price_col], df["Storage_GB"])
axes[2].text(
    0.95,
    0.05,
    f"r={pearson_sto:.3f}",
    ha="right",
    va="bottom",
    transform=axes[2].transAxes,
    color=dark_pink,
)

# 5) Residuals Histogram
sns.histplot(
    model.resid,
    color=blue,
    kde=False,
    stat="density",
    ax=axes[3],
)
sns.kdeplot(
    model.resid,
    color=pink,
    ax=axes[3],
)
axes[3].set_xlabel("Residuals")
axes[3].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
