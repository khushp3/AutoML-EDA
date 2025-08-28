import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def compute_correlations(df):
    results = {"numeric": {}, "categorical": {}}
    
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 1:
        results["numeric"] = df[num_cols].corr(method="pearson").to_dict()
    
    cat_cols = df.select_dtypes(include="object").columns
    for i, col1 in enumerate(cat_cols):
        for col2 in cat_cols[i+1:]:
            results["categorical"][f"{col1}-{col2}"] = cramers_v(df[col1], df[col2])
    
    return results
