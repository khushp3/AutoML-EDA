import pandas as pd
from automl.core.profiler import profile_dataframe

def test_summary_statistics():
    df = pd.DataFrame({
        'age': [22, 30, 24, None, 28],
        'gender': ['male', 'female', 'female', 'male', 'male']
    })
    result = profile_dataframe(df)
    assert 'age' in result
    assert isinstance(result['age']['mean'], float)
