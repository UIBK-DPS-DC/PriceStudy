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
    if "ebs only" in storage_str:
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


def parse_network(network_str):
    if pd.isna(network_str):
        return np.nan

    # e.g. "Up to 5 Gigabit"
    network_str = network_str.strip().lower()
    match = re.search(r"(\d+(\.\d+)?)\s*gigabit", network_str)

    if match:
        return float(match.group(1))

    return np.nan


# Parse the values
df["vCPUs_Num"] = df["vCPUs"].apply(parse_vcpus)
df["Memory_GiB"] = df["Instance Memory"].apply(parse_memory)
df["Storage_GB"] = df["Instance Storage"].apply(parse_storage)
df["Price_Dollar"] = df["On Demand"].apply(parse_price)
df["Network_Gbit"] = df["Network Performance"].apply(parse_network)

df_filtered = df[(df["Storage_GB"] > 0) & (df["Price_Dollar"] > 0)]


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
df_filtered = remove_outliers(df_filtered, "Network_Gbit")

# Correlation analysis
price_col = "Price_Dollar"

# Make sure columns exist
required_cols = [price_col, "vCPUs_Num", "Memory_GiB", "Storage_GB", "Network_Gbit"]
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
pearson_net, p_net = pearsonr(df_filtered[price_col], df_filtered["Network_Gbit"])

print("Pearson correlation coefficients (OnDemandPrice vs ...):")
print(f"  vCPUs:     r={pearson_cpu:.3f}, p={p_cpu:.3e}")
print(f"  MemoryGiB: r={pearson_mem:.3f}, p={p_mem:.3e}")
print(f"  StorageGB: r={pearson_sto:.3f}, p={p_sto:.3e}")
print(f"  NetworkGbit: r={pearson_net:.3f}, p={p_net:.3e}\n")

# Pairwise Spearman correlations
spearman_cpu, _ = spearmanr(df_filtered[price_col], df_filtered["vCPUs_Num"])
spearman_mem, _ = spearmanr(df_filtered[price_col], df_filtered["Memory_GiB"])
spearman_sto, _ = spearmanr(df_filtered[price_col], df_filtered["Storage_GB"])
spearman_net, _ = spearmanr(df_filtered[price_col], df_filtered["Network_Gbit"])

print("Spearman correlation coefficients (OnDemandPrice vs ...):")
print(f"  vCPUs:     r={spearman_cpu:.3f}")
print(f"  MemoryGiB: r={spearman_mem:.3f}")
print(f"  StorageGB: r={spearman_sto:.3f}")
print(f"  NetworkGbit: r={spearman_net:.3f}\n")

# Perform linear regression to derive per-unit pricing
X = df_filtered[["vCPUs_Num", "Memory_GiB", "Storage_GB", "Network_Gbit"]]
y = df_filtered[price_col]  # target

# Add a constant to get an intercept in the model
X = sm.add_constant(X)

# Fit ordinary least squares regression
model = sm.OLS(y, X).fit()

print("OLS Regression Results:")
print(model.summary())

# Set up a grid layout (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Correlation Matrix Heatmap
sns.heatmap(
    corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=axes[0, 0]
)
axes[0, 0].set_title("Correlation Matrix (Pearson)")

# Regression plot (vCPUs vs Price)
sns.regplot(
    x=df_filtered["vCPUs_Num"],
    y=df_filtered[price_col],
    scatter=True,
    line_kws={"color": "red"},
    ax=axes[0, 1],
)
axes[0, 1].set_title("Regression: vCPUs vs Price")
axes[0, 1].set_xlabel("vCPUs")
axes[0, 1].set_ylabel("Price ($)")

# Regression plot (Memory vs Price)
sns.regplot(
    x=df_filtered["Memory_GiB"],
    y=df_filtered[price_col],
    scatter=True,
    line_kws={"color": "red"},
    ax=axes[0, 2],
)
axes[0, 2].set_title("Regression: Memory vs Price")
axes[0, 2].set_xlabel("Memory (GiB)")
axes[0, 2].set_ylabel("Price ($)")

# Regression plot (Storage vs Price)
sns.regplot(
    x=df_filtered["Storage_GB"],
    y=df_filtered[price_col],
    scatter=True,
    line_kws={"color": "red"},
    ax=axes[1, 0],
)
axes[1, 0].set_title("Regression: Storage vs Price")
axes[1, 0].set_xlabel("Storage (GB)")
axes[1, 0].set_ylabel("Price ($)")

# Regression plot (Network vs Price)
sns.regplot(
    x=df_filtered["Network_Gbit"],
    y=df_filtered[price_col],
    scatter=True,
    line_kws={"color": "red"},
    ax=axes[1, 1],
)
axes[1, 1].set_title("Regression: Network vs Price")
axes[1, 1].set_xlabel("Network (Gbit)")
axes[1, 1].set_ylabel("Price ($)")

# Residuals Histogram
sns.histplot(model.resid, kde=True, color="blue", ax=axes[1, 2])
axes[1, 2].set_title("Residuals from OLS Regression")
axes[1, 2].set_xlabel("Residuals")
axes[1, 2].set_ylabel("Frequency")

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
