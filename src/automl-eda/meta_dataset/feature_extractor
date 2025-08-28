import pandas as pd
import numpy as np

def get_meta_features(df: pd.DataFrame, target_name: str, dataset_name: str):
    rows = []

    for col in df.columns:
        if col == target_name:
            continue

        series = df[col]
        row = {
            "dataset": dataset_name,
            "column": col,
            "data_type": str(series.dtype),
            "rows_missing": series.isna().sum(),
            "missing_percentage": series.isna().mean(),
            "rows_unique": series.nunique(),
            "unique_percentage": series.nunique() / len(series),
            "mean": series.mean() if pd.api.types.is_numeric_dtype(series) else np.nan,
            "std": series.std() if pd.api.types.is_numeric_dtype(series) else np.nan,
            "skewness": series.skew() if pd.api.types.is_numeric_dtype(series) else np.nan,
            "is_categorical": pd.api.types.is_categorical_dtype(series) or series.dtype == "object",
        }

        row["mean_norm"] = (row["mean"] - series.min()) / (series.max() - series.min()) if pd.api.types.is_numeric_dtype(series) else np.nan
        row["std_norm"] = row["std"] / (series.max() - series.min()) if pd.api.types.is_numeric_dtype(series) else np.nan

        row["best_strategy"] = None  
        row["task_type"] = "classification"  
        row["target"] = target_name
        rows.append(row)

    return pd.DataFrame(rows)
