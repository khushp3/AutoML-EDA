import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def ml_readiness_profile(df: pd.DataFrame, target: str = None):
    dataset_stats = {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "feature_types": df.dtypes.value_counts().to_dict(),
        "missing_cells": df.isnull().sum().sum(),
        "missing_cells_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_rows_pct": (df.duplicated().sum() / len(df)) * 100,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 ** 2)
    }

    # ---------- 2. Per-column stats ----------
    column_stats = {}
    for col in df.columns:
        col_data = df[col]
        col_type = col_data.dtype
        stats = {
            "dtype": str(col_type),
            "distinct": col_data.nunique(),
            "distinct_pct": (col_data.nunique() / len(df)) * 100,
            "missing": col_data.isnull().sum(),
            "missing_pct": (col_data.isnull().sum() / len(df)) * 100,
            "memory_size_kb": col_data.memory_usage(deep=True) / 1024
        }

        if np.issubdtype(col_type, np.number):  # Numeric features
            stats.update({
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "zeros_pct": (col_data == 0).sum() / len(df) * 100,
                "negatives_pct": (col_data < 0).sum() / len(df) * 100,
                "skewness": skew(col_data.dropna()) if col_data.dropna().nunique() > 1 else 0,
                "kurtosis": kurtosis(col_data.dropna()) if col_data.dropna().nunique() > 1 else 0
            })

        elif np.issubdtype(col_type, np.datetime64):  # Datetime features
            stats.update({
                "min_date": col_data.min(),
                "max_date": col_data.max(),
                "range_days": (col_data.max() - col_data.min()).days if col_data.notnull().all() else None
            })

        elif col_data.dtype == object or col_data.dtype.name == "category":  # Categorical/Text features
            stats.update({
                "top_category": col_data.value_counts().index[0] if not col_data.value_counts().empty else None,
                "top_category_freq": col_data.value_counts().iloc[0] if not col_data.value_counts().empty else None,
                "rare_categories_pct": (sum(col_data.value_counts(normalize=True) < 0.01) / col_data.nunique() * 100) if col_data.nunique() > 0 else None
            })

        column_stats[col] = stats

    # ---------- 3. Correlation stats ----------
    correlation_stats = {}
    if target and target in df.columns and np.issubdtype(df[target].dtype, np.number):
        corr_with_target = df.corr(numeric_only=True)[target].sort_values(ascending=False)
        correlation_stats["target_correlations"] = corr_with_target.to_dict()

    high_corr_pairs = []
    corr_matrix = df.corr(numeric_only=True).abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > 0.85:  # Threshold for "high" correlation
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    correlation_stats["highly_correlated_pairs"] = high_corr_pairs

    report = {
        "dataset_stats": dataset_stats,
        "column_stats": column_stats,
        "correlation_stats": correlation_stats
    }

    return report

