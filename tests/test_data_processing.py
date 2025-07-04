# pytest
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
from data_processing import (
    prepare_features,
    build_pipeline,
    aggregate_customer_features,
    extract_time_features
)

# Load sample raw data for testing
@pytest.fixture(scope="module")
def raw_data():
    return pd.DataFrame([
        {
            "TransactionId": "TransactionId_76871",
            "CustomerId": "CustomerId_4406",
            "Amount": 1000,
            "TransactionStartTime": "2018-11-15T02:18:49Z"
        },
        {
            "TransactionId": "TransactionId_380",
            "CustomerId": "CustomerId_988",
            "Amount": 20000,
            "TransactionStartTime": "2018-11-15T02:19:08Z"
        },
        {
            "TransactionId": "TransactionId_26203",
            "CustomerId": "CustomerId_4683",
            "Amount": 500,
            "TransactionStartTime": "2018-11-15T02:44:21Z"
        },
    ])
def test_aggregate_customer_features(raw_data):
    agg_df = aggregate_customer_features(raw_data)
    assert "total_amount" in agg_df.columns
    assert agg_df.shape[0] > 0
    assert agg_df.isnull().sum().sum() < agg_df.shape[0]  # Not all NaNs

def test_extract_time_features(raw_data):
    df_with_time = extract_time_features(raw_data.copy())
    expected_cols = ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]
    for col in expected_cols:
        assert col in df_with_time.columns
        assert df_with_time[col].notnull().all()

def test_prepare_features(raw_data):
    final_df = prepare_features(raw_data)
    expected_cols = ["CustomerId", "total_amount", "avg_amount", "transaction_month"]
    for col in expected_cols:
        assert col in final_df.columns
    assert final_df.isnull().sum().sum() < final_df.shape[0]  # Not fully null

def test_build_pipeline():
    numeric_cols = ["total_amount", "avg_amount", "std_amount", "transaction_count"]
    categorical_cols = ["transaction_month"]
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    assert pipeline is not None

