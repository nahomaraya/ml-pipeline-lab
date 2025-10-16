# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 15:22:50 2025

Refactored to expose a reusable preprocessing function for library-style usage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

__all__ = [
    "preprocess_data",
]


def preprocess_data(csv_path: str,
                    target_column: str = "MAX_SEV",
                    positive_class: str = "injury",
                    verbose: bool = False):
    """
    Load and preprocess the accidents dataset, returning train/val/test splits.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    accidents_data = pd.read_csv(csv_path)

    if verbose:
        print(f"Number of rows: {accidents_data.shape[0]}, Columns: {accidents_data.shape[1]}")
        print("\nData types of each column:")
        print(accidents_data.dtypes)

    # Clean column names
    accidents_data.columns = accidents_data.columns.str.strip()
    accidents_data.columns = accidents_data.columns.str.replace(r"\s+", " ", regex=True)

    # Handle missing values
    accidents_data.fillna(accidents_data.mean(numeric_only=True), inplace=True)

    # Handle outliers for known numeric columns if they exist
    candidate_outlier_cols = [
        "WRK_ZONE", "WKDY", "INT_HWY", "LEVEL", "SUR_COND_dry", "WEATHER_adverse",
    ]
    for col in candidate_outlier_cols:
        if col in accidents_data.columns:
            accidents_data[col] = accidents_data[col].clip(
                lower=accidents_data[col].quantile(0.05),
                upper=accidents_data[col].quantile(0.95),
            )

    # Binary encode target
    if target_column not in accidents_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")
    mapping = {positive_class: 1}
    accidents_data["MAX_SEV_binary"] = accidents_data[target_column].map(lambda v: mapping.get(v, 0))

    # One-hot encode categorical features (excluding target)
    categorical_cols = [
        col for col in accidents_data.select_dtypes(include="object").columns
        if col != target_column
    ]
    accidents_data = pd.get_dummies(
        accidents_data,
        columns=categorical_cols,
        drop_first=True,
        dtype=int,
    )

    # Split
    X = accidents_data.drop(columns=[target_column, "MAX_SEV_binary"])
    y = accidents_data["MAX_SEV_binary"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Remove constant and highly correlated columns (based on training set only)
    constant_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        X_val = X_val.drop(columns=[c for c in constant_cols if c in X_val.columns])
        X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

    corr_matrix = X_train.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    if to_drop:
        X_train = X_train.drop(columns=to_drop)
        X_val = X_val.drop(columns=[c for c in to_drop if c in X_val.columns])
        X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])

    if verbose:
        print("Preprocessing complete. Data ready for modeling.")

    return X_train, X_val, X_test, y_train, y_val, y_test
