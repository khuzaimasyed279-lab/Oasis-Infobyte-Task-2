# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


sns.set_theme(style="whitegrid")


# STEP 2: LOAD DATASET

df = pd.read_csv("data.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# STEP 3: DATA CLEANING

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Select numerical columns
numeric_cols = [
    "Time on App",
    "Time on Website",
    "Length of Membership",
    "Yearly Amount Spent"
]

# Convert columns into numeric format
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values using median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nData Cleaning Completed!")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# STEP 4: DESCRIPTIVE STATISTICS

print("\nDescriptive Statistics:")
print(df[numeric_cols].describe())

print("\nAverage Yearly Amount Spent:",
      round(df["Yearly Amount Spent"].mean(), 2))

print("Average Time on App:",
      round(df["Time on App"].mean(), 2))

print("Average Time on Website:",
      round(df["Time on Website"].mean(), 2))

print("Average Membership Length:",
      round(df["Length of Membership"].mean(), 2))


# STEP 5: FEATURE SELECTION

# Select three behavioural features

features = [
    "Time on App",
    "Time on Website",
    "Length of Membership"
]

X = df[features]

print("\nSelected Features:")
print(X.head())


# STEP 6: DATA STANDARDIZATION

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData Standardization Completed!")


# STEP 7: ELBOW METHOD

inertia = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


# Plot Elbow Method

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))

plt.show()


# STEP 8: SILHOUETTE SCORE

print("\nSilhouette Scores:")

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print("K =", k, "Score =", round(score, 3))


# STEP 9: APPLY K-MEANS CLUSTERING

# Select K based on the Elbow Method.
# Here we use K = 4 as our initial choice.

optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Clustering Completed!")


# STEP 10: DISPLAY CUSTOMER CLUSTERS

print("\nNumber of Customers in Each Cluster:")

print(df["Cluster"].value_counts().sort_index())


# STEP 11: VISUALIZATION 1
# Time on App vs Time on Website

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Time on App",
    y="Time on Website",
    hue="Cluster",
    palette="tab10",
    s=70
)

plt.title("Customer Segmentation: App vs Website Time")
plt.xlabel("Time on App")
plt.ylabel("Time on Website")

plt.show()


# STEP 12: VISUALIZATION 2
# Membership Length vs Time on App

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Length of Membership",
    y="Time on App",
    hue="Cluster",
    palette="tab10",
    s=70
)

plt.title("Customer Segmentation: Membership vs App Time")
plt.xlabel("Length of Membership")
plt.ylabel("Time on App")

plt.show()


# STEP 13: VISUALIZATION 3
# Membership Length vs Time on Website

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Length of Membership",
    y="Time on Website",
    hue="Cluster",
    palette="tab10",
    s=70
)

plt.title("Customer Segmentation: Membership vs Website Time")
plt.xlabel("Length of Membership")
plt.ylabel("Time on Website")

plt.show()


# STEP 14: CUSTOMER CLUSTER PROFILING

profile = df.groupby("Cluster").agg({

    "Time on App": "mean",

    "Time on Website": "mean",

    "Length of Membership": "mean",

    "Yearly Amount Spent": "mean"

}).round(2)

profile["Number of Customers"] = (
    df.groupby("Cluster").size()
)

profile["Customer Percentage"] = (
    profile["Number of Customers"] / len(df) * 100
).round(2)

print("\nCUSTOMER CLUSTER PROFILE:")

print(profile)


# STEP 15: BAR CHART
# Number of Customers in Each Cluster

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Cluster",
    hue="Cluster",
    palette="viridis",
    legend=False
)

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")

plt.show()


# STEP 16: YEARLY SPENDING ANALYSIS

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Cluster",
    y="Yearly Amount Spent",
    hue="Cluster",
    palette="viridis",
    errorbar=None,
    legend=False
)

plt.title("Average Yearly Spending by Cluster")
plt.xlabel("Customer Cluster")
plt.ylabel("Average Yearly Amount Spent")

plt.show()


# STEP 17: MARKETING INSIGHTS

overall_spending = df["Yearly Amount Spent"].mean()

overall_app = df["Time on App"].mean()

overall_web = df["Time on Website"].mean()

overall_membership = df["Length of Membership"].mean()

print("\n========== MARKETING INSIGHTS ==========")

for cluster in profile.index:

    p = profile.loc[cluster]

    print("\nCluster:", cluster)

    if p["Yearly Amount Spent"] > overall_spending:
        print("Spending: Above average")

        print("Marketing Action: Offer loyalty rewards,")
        print("exclusive discounts and referral benefits.")

    else:
        print("Spending: Below or equal to average")

        print("Marketing Action: Offer personalised")
        print("recommendations and promotional discounts.")

    if p["Time on App"] > overall_app:
        print("App Engagement: High")

    else:
        print("App Engagement: Lower")

    if p["Time on Website"] > overall_web:
        print("Website Engagement: High")

    else:
        print("Website Engagement: Lower")

    if p["Length of Membership"] > overall_membership:
        print("Membership: Longer than average")

    else:
        print("Membership: Shorter than or equal to average")


# STEP 18: SAVE FINAL RESULTS

df.to_csv(
    "customer_segments_output.csv",
    index=False
)

profile.to_csv(
    "cluster_profile_summary.csv"
)

print("\n===================================")
print("CUSTOMER SEGMENTATION PROJECT COMPLETED!")
print("===================================")

print("\nFiles Saved Successfully:")
print("1. customer_segments_output.csv")
print("2. cluster_profile_summary.csv")