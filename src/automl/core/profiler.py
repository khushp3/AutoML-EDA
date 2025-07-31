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