import pandas as pd
from automl.core.profiler import profile_dataframe
from automl.core.recommender import Recommender

df = pd.read_csv('examples/titanic.csv')
summary = profile_dataframe(df)

recommender = Recommender(df, target='Survived')
recommendations = recommender.generate_recommendations()

print(summary)
print(recommendations)
