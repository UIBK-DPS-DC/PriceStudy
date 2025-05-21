import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline


# Read input data
csv_filename = "data/ec2_prices.csv"
df = pd.read_csv(csv_filename)


# Parse data
def parse_vcpus(vcpu_str):
    if pd.isna(vcpu_str):
        return np.nan
    vcpu_str = vcpu_str.strip().lower()
    match = re.search(r"(\d+(\.\d+)?)\s*vcpus?", vcpu_str)
    return float(match.group(1)) if match else np.nan


def parse_memory(mem_str):
    if pd.isna(mem_str):
        return np.nan
    mem_str = mem_str.strip().lower()
    parts = mem_str.split()
    return float(parts[0]) if len(parts) >= 2 and parts[1].startswith("gib") else np.nan


def parse_storage(storage_str):
    if pd.isna(storage_str):
        return np.nan
    storage_str = storage_str.strip().lower()
    match = re.search(r"^(\d+)\sgb(?=.*nvme ssd)", storage_str)
    return float(match.group(1)) if match else np.nan


def parse_price(price_str):
    if pd.isna(price_str):
        return 0.0
    s = price_str.strip().lower()
    if "unavailable" in s:
        return 0.0
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
    network_str = network_str.strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)\sgigabit$", network_str)
    return float(match.group(1)) if match else np.nan


df["vCPUs_Num"] = df["vCPUs"].apply(parse_vcpus)
df["GPUs_Num"] = df["GPUs"]
df["Memory_GiB"] = df["Instance Memory"].apply(parse_memory)
df["Storage_GB"] = df["Instance Storage"].apply(parse_storage)
df["Price_Dollar"] = df["On Demand"].apply(parse_price)
df["Network_Gb"] = df["Network Performance"].apply(parse_network)

df = df[(df["Storage_GB"] > 0) & (df["Price_Dollar"] > 0) & (df["Network_Gb"] > 0)]
df = df[
    df["Name"].str.startswith(
        (
            "M",  # General purpose
            "T",  # General purpose
            "C",  # Compute optimized
            "R",  # Memory optimized
            "X",  # Memory optimized
            "U",  # Memory optimized
            "z",  # Memory optimized
            "P",  # Accelerated computing
            "G",  # Accelerated computing
        )
    )
    & ~df["Name"].str.startswith("Mac")
]


# Remove outliers
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


for col in ["Storage_GB", "Memory_GiB", "vCPUs_Num", "Price_Dollar", "Network_Gb"]:
    df = remove_outliers(df, col)

print(f"Number of rows after filtering: {df.shape[0]}")

# Split the data into its features and the target
features = ["GPUs_Num", "vCPUs_Num", "Memory_GiB", "Storage_GB", "Network_Gb"]
X = df[features]
y = df["Price_Dollar"]

# Create a training split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a random forest regressor
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Export the regressor to a PMML pipeline
pipeline = PMMLPipeline([("regressor", rf)])
sklearn2pmml(pipeline, "model/rf_ec2_price.pmml", with_repr=True)

# Perform price prediction on the dataset
y_pred = rf.predict(X)

print(rf.predict(np.array([[0, 64, 256, 5250, 0]])))

df_test = X.copy()
df_test["Price_Dollar"] = y
df_test["RF_Predicted_Price"] = y_pred
df_test["Residual"] = df_test["Price_Dollar"] - df_test["RF_Predicted_Price"]

# Save the dataset
df_test.to_csv("output/data.csv")

# Diplay statistics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"Random Forest Regression Performance:")
print(f"MAE: {mae:.5f}  RMSE: {rmse:.5f}  R^2: {r2:.5f}")

# Plot
sns.set_theme(style="whitegrid")
sns.set_palette(sns.color_palette("tab10"))
palette = sns.color_palette("tab10")
line_color = palette[1]

plt.figure(figsize=(7, 7))
sns.scatterplot(x="Price_Dollar", y="RF_Predicted_Price", data=df_test)
sns.lineplot(
    x=[df_test["Price_Dollar"].min(), df_test["Price_Dollar"].max()],
    y=[df_test["Price_Dollar"].min(), df_test["Price_Dollar"].max()],
    color=line_color,
    linestyle="--",
)
plt.xlabel("Actual Price ($/h)")
plt.ylabel("Predicted Price ($/h)")
plt.grid(True)
plt.savefig(
    "figure/actual_vs_predicted.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.savefig(
    "figure/actual_vs_predicted.png",
    bbox_inches="tight",
    pad_inches=0,
)
plt.tight_layout()

plt.figure(figsize=(7, 5))
sns.histplot(
    data=df_test,
    x="Residual",
    bins=20,
    kde=True,
    color=line_color,
    stat="density",
)
plt.xlabel("Price Error ($/h)")
plt.ylabel(None)
plt.gca().set_yticklabels([])
plt.tight_layout()
plt.savefig(
    "figure/residuals_hist.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.savefig(
    "figure/residuals_hist.png",
    bbox_inches="tight",
    pad_inches=0,
)
