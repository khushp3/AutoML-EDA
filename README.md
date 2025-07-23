# AutoML-EDA

## Description

AutoML Data Profiler is a Python library for automated exploratory data analysis (EDA) designed for machine learning workflows. It performs backend data profiling tasks such as statistical summaries, missing value detection, outlier analysis, and ML-specific quality checks. This tool is ideal for integrating into AutoML or preprocessing pipelines.

## Installation

```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Features

- Summary statistics (mean, median, std, etc.)

- Missing value analysis

- Outlier detection (Z-score, IQR)

- Correlation matrix

- Distribution analysis (categorical/numerical)

- ML-relevant metrics (feature variance, class imbalance detection)
