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
df["Memory_GiB"] = df["Instance Memory"].apply(parse_memory)
df["Storage_GB"] = df["Instance Storage"].apply(parse_storage)
df["Price_Dollar"] = df["On Demand"].apply(parse_price)

df_filtered = df[(df["Storage_GB"] > 0) & (df["Price_Dollar"] > 0)]

df_filtered = df_filtered[
    df_filtered["Name"]
    .str.lower()
    .str.startswith(
        (
            "t",  # General purpose
            "m",  # General purpose
            "c",  # Compute-optimized
            "r",  # Memory-optimized
            "x",  # Memory-optimized
            "u",  # Memory-optimized
            "z",  # Memory-optimized
        )
    )
    & ~df["Name"].str.lower().str.startswith("trn")
]

name_list = df_filtered["Name"].tolist()
print(name_list)


# Outlier removal
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


df_filtered = remove_outliers(df_filtered, "Storage_GB")
df_filtered = remove_outliers(df_filtered, "Memory_GiB")
df_filtered = remove_outliers(df_filtered, "vCPUs_Num")

# Correlation analysis
price_col = "Price_Dollar"

# Make sure columns exist
required_cols = [price_col, "vCPUs_Num", "Memory_GiB", "Storage_GB"]
for col in required_cols:
    if col not in df_filtered.columns:
        raise ValueError(f"Column '{col}' not found in dataframe!")

# Basic correlation matrix (Pearson)
corr_matrix = df_filtered[required_cols].corr()
print("Correlation matrix (Pearson):")
print(corr_matrix, "\n")

# Pairwise Pearson correlations
pearson_cpu, p_cpu = pearsonr(df_filtered[price_col], df_filtered["vCPUs_Num"])
pearson_mem, p_mem = pearsonr(df_filtered[price_col], df_filtered["Memory_GiB"])
pearson_sto, p_sto = pearsonr(df_filtered[price_col], df_filtered["Storage_GB"])

print("Pearson correlation coefficients (OnDemandPrice vs ...):")
print(f"  vCPUs:     r={pearson_cpu:.3f}, p={p_cpu:.3e}")
print(f"  MemoryGiB: r={pearson_mem:.3f}, p={p_mem:.3e}")
print(f"  StorageGB: r={pearson_sto:.3f}, p={p_sto:.3e}")

# Pairwise Spearman correlations
spearman_cpu, _ = spearmanr(df_filtered[price_col], df_filtered["vCPUs_Num"])
spearman_mem, _ = spearmanr(df_filtered[price_col], df_filtered["Memory_GiB"])
spearman_sto, _ = spearmanr(df_filtered[price_col], df_filtered["Storage_GB"])

print("Spearman correlation coefficients (OnDemandPrice vs ...):")
print(f"  vCPUs:     r={spearman_cpu:.3f}")
print(f"  MemoryGiB: r={spearman_mem:.3f}")
print(f"  StorageGB: r={spearman_sto:.3f}")

# Perform linear regression to derive per-unit pricing
X = df_filtered[["vCPUs_Num", "Memory_GiB", "Storage_GB"]]
y = df_filtered[price_col]  # target

# Fit ordinary least squares regression
model = sm.OLS(y, X).fit()

print("OLS Regression Results:")
print(model.summary())

# Use a clean style
sns.set_theme(style="whitegrid")

# Define your 5 palette colors for lines and hist
dark_green = "#5ba300"
light_green = "#89ce00"
blue = "#0073e6"
pink = "#e6308a"
dark_pink = "#b51963"

# For the correlation heatmap, create a custom colormap that goes from white to blue
heatmap_cmap = LinearSegmentedColormap.from_list(
    "heatmap_cmap", ["#ffffff", blue], N=256
)

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# Define the label mapping for both rows and columns
label_mapping = {
    "Price_Dollar": "Price ($)",
    "vCPUs_Num": "vCPUs",
    "Memory_GiB": "Memory (GiB)",
    "Storage_GB": "Storage (GB)",
}

# Rename the index and columns in the correlation matrix
corr_matrix = corr_matrix.rename(index=label_mapping, columns=label_mapping)

# 1) Correlation Matrix Heatmap
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap=heatmap_cmap,  # single gradient colormap
    fmt=".2f",
    linewidths=0.5,
    ax=axes[0],
    cbar_kws={"shrink": 0.8},  # smaller colorbar
)
axes[0].set_title("Correlation Matrix (Pearson)")

# 2) Regression plot (vCPUs vs Price)
sns.regplot(
    x=df_filtered["vCPUs_Num"],
    y=df_filtered[price_col],
    scatter_kws={"color": blue},  # scatter points
    line_kws={"color": pink},  # regression line
    ax=axes[1],
)
axes[1].set_title("Regression: vCPUs vs Price")
axes[1].set_xlabel("vCPUs")
axes[1].set_ylabel("Price ($)")

# 3) Regression plot (Memory vs Price)
sns.regplot(
    x=df_filtered["Memory_GiB"],
    y=df_filtered[price_col],
    scatter_kws={"color": blue},
    line_kws={"color": pink},
    ax=axes[2],
)
axes[2].set_title("Regression: Memory vs Price")
axes[2].set_xlabel("Memory (GiB)")
axes[2].set_ylabel("Price ($)")

# 4) Regression plot (Storage vs Price)
sns.regplot(
    x=df_filtered["Storage_GB"],
    y=df_filtered[price_col],
    scatter_kws={"color": blue},
    line_kws={"color": pink},
    ax=axes[3],
)
axes[3].set_title("Regression: Storage vs Price")
axes[3].set_xlabel("Storage (GB)")
axes[3].set_ylabel("Price ($)")

# 5) Residuals Histogram
sns.histplot(
    model.resid,
    color=blue,
    kde=False,
    stat="density",
    ax=axes[4],
)
sns.kdeplot(
    model.resid,
    color=pink,
    ax=axes[4],
)
axes[4].set_title("Residuals from OLS Regression")
axes[4].set_xlabel("Residuals")
axes[4].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
