import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE

#data-> model ready format

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