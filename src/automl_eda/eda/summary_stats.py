'''
This module computes summary statistics for a dataset.
Currently supports only pandas DataFrames.
'''
import pandas as pd
from typing import Dict, List, Optional
from collections import Counter

def dataset_overview(df: pd.DataFrame) -> dict:
    features = df.shape[1]
    instances = df.shape[0]
    missing_instances = df.isnull().sum().sum()
    missing_instances_percentage = missing_instances / (features * instances) * 100
    duplicate_instances = df.duplicated().sum()
    duplicate_instances_percentage = duplicate_instances / instances * 100
    memory_size = df.memory_usage(deep=True).sum()
    types_count = df.dtypes.value_counts().to_dict()

    return {
        "num_variables": features,
        "num_observations": instances,
        "missing_cells": missing_instances,
        "missing_cells_pct": missing_instances_percentage,
        "duplicate_rows": duplicate_instances,
        "duplicate_rows_pct": duplicate_instances_percentage,
        "total_memory_kb": round(memory_size / 1024, 2),
        "avg_record_size_b": round(memory_size / instances, 2),
        "variable_types": {str(k): v for k, v in types_count.items()}
    }

    return overview 

