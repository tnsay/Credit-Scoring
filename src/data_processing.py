import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
from datetime import timedelta
from sklearn.cluster import KMeans

#data-> model ready format
#add is_high_risk column using RFM, CustomerId

# Step 1: Aggregate Customer-Level Features


def aggregate_customer_features(df):
    """
    Aggregate transaction data per CustomerId.
    Returns a customer-level dataframe.
    """
    agg_df = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        std_amount=("Amount", "std"),
        transaction_count=("Amount", "count")
    ).reset_index()
    return agg_df

# Step 2: Extract Time-Based Features

def extract_time_features(df):
    """
    Extracts hour, day, month, and year from TransactionStartTime.
    """
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year
    return df

# Step 3: Build Feature Transformation Pipeline

def build_pipeline(numeric_features, categorical_features):
    """
    Creates a preprocessing pipeline for numeric and categorical columns.
    """
    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),  #missing values
        ("scaler", StandardScaler())                  #scales all values to similar ranges, StandardScaler ->standardize to mean = 0 and standard deviation = 1
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),   #missing values
        ("encoder", OneHotEncoder(handle_unknown="ignore"))     #new column for each category
    ])

    # Combine
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

# Step 4: Optional - WoE Transformation

def apply_woe(df, target_col):
    """
    Applies Weight of Evidence transformation using xverse.
    Returns dataframe with transformed features.
    """
    woe = WOE()
    woe.fit(df, df[target_col])
    df_woe = woe.transform(df)
    return df_woe

# Step 5: Full Preprocessing Function

def prepare_features(raw_df):
    """
    Full pipeline to generate model-ready features from raw transaction data.
    """
    df = extract_time_features(raw_df)
    agg_df = aggregate_customer_features(df)
    
    # Merge time features back into customer-level aggregation
    time_cols = ["CustomerId", "transaction_hour", "transaction_day", "transaction_month", "transaction_year"]
    time_df = df[time_cols].drop_duplicates("CustomerId")
    final_df = pd.merge(agg_df, time_df, on="CustomerId", how="left")

    return final_df


#task4 Proxy Target Variable Engineering
def create_proxy_target(df, snapshot_date=None):
    """
    Creates proxy credit risk target based on RFM clustering.
    Returns a DataFrame with CustomerId and is_high_risk (0 or 1).
    """
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    # Scale the RFM features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # KMeans clustering (3 segments)
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (lowest Frequency + Monetary, highest Recency)
    cluster_summary = rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = cluster_summary.sort_values(by=["Frequency", "Monetary", "Recency"], ascending=[True, True, False]).index[0]

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]

#train ready data
def prepare_features(raw_df):
    """
    Full pipeline to generate model-ready features from raw transaction data,
    including is_high_risk label.
    """
    df = extract_time_features(raw_df)
    agg_df = aggregate_customer_features(df)

    # Merge time features
    time_cols = ["CustomerId", "transaction_hour", "transaction_day", "transaction_month", "transaction_year"]
    time_df = df[time_cols].drop_duplicates("CustomerId")
    final_df = pd.merge(agg_df, time_df, on="CustomerId", how="left")

    # Add is_high_risk label
    risk_df = create_proxy_target(df)
    final_df = pd.merge(final_df, risk_df, on="CustomerId", how="left")

    return final_df

if __name__ == "__main__":
    raw = pd.read_csv("data/raw/data.csv")  
    processed = prepare_features(raw)
    processed.to_csv("data/processed/train_ready.csv", index=False)
    print("✅ Processed data saved to data/processed/train_ready.csv")
