# Features to include: Type Detection, Memory Usage, Execution Time
# Spearman and Pearson Correlation Coefficients

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import spearmanr, pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
class DataProfiler(BaseEstimator, TransformerMixin):
    def __init__(self, target: Optional[str] = None):
        self.target = target
        self.feature_types: Dict[str, str] = {}
        self.memory_usage: Dict[str, float] = {}
        self.execution_time: Dict[str, float] = {}
        self.correlation_coefficients: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProfiler':
        self._detect_feature_types(X)
        self._calculate_memory_usage(X)
        if y is not None:
            self._calculate_correlation_coefficients(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def _detect_feature_types(self, X: pd.DataFrame):
        for column in X.columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                self.feature_types[column] = 'numeric'
            elif pd.api.types.is_categorical_dtype(X[column]) or X[column].dtype == 'object':
                self.feature_types[column] = 'categorical'
            else:
                self.feature_types[column] = 'unknown'

    def _calculate_memory_usage(self, X: pd.DataFrame):
        for column in X.columns:
            self.memory_usage[column] = X[column].memory_usage(deep=True)

    def _calculate_correlation_coefficients(self, X: pd.DataFrame, y: pd.Series):
        for column in X.columns:
            if self.feature_types[column] == 'numeric':
                spearman_corr, _ = spearmanr(X[column], y)
                pearson_corr, _ = pearsonr(X[column], y)
                self.correlation_coefficients[column] = {
                    'spearman': spearman_corr,
                    'pearson': pearson_corr
                }
            elif self.feature_types[column] == 'categorical':
                le = LabelEncoder()
                encoded_y = le.fit_transform(y)
                mi_score = mutual_info_regression(X[[column]], encoded_y) if y.dtype in ['int64', 'float64'] else mutual_info_classif(X[[column]], encoded_y)
                self.correlation_coefficients[column] = {'mutual_info': mi_score[0]}

# profiler.py
import pandas as pd

from dataset_summary import dataset_summary
from categorical_describe import describe_categorical
from numerical_describe import describe_numerical
from boolean_describe import describe_boolean
from datetime_describe import describe_datetime
from correlations import calculate_correlations


def profile_dataset(df: pd.DataFrame):
    """
    Main entry point for profiling a dataset.
    Produces dataset-level summary, column-level stats, and correlations.
    """

    report = {}

    # 1. Dataset-level summary
    report["dataset_overview"] = dataset_summary(df)

    # 2. Column-level stats
    report["columns"] = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)

        if dtype.startswith("object") or dtype == "category":
            report["columns"].append(describe_categorical(series))

        elif pd.api.types.is_numeric_dtype(series):
            report["columns"].append(describe_numerical(series))

        elif pd.api.types.is_bool_dtype(series):
            report["columns"].append(describe_boolean(series))

        elif pd.api.types.is_datetime64_any_dtype(series):
            report["columns"].append(describe_datetime(series))

        else:
            report["columns"].append({
                "feature_name": col,
                "feature_type": "unknown",
                "stats": {},
                "checks": ["Unsupported dtype detected"]
            })

    # 3. Correlations (only numeric columns)
    report["correlations"] = calculate_correlations(df)

    return report


if __name__ == "__main__":
    # Example usage
    data = {
        "age": [23, 45, 31, None, 52],
        "gender": ["M", "F", "F", "M", None],
        "signup_date": pd.to_datetime(["2021-01-01", "2021-02-01", None, "2021-04-01", "2021-05-01"]),
        "is_active": [True, False, True, True, True],
    }
    df = pd.DataFrame(data)

    report = profile_dataset(df)

    # Pretty-print results
    import pprint
    pprint.pprint(report, width=100)
