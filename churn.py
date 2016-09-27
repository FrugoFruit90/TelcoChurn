import pandas as pd

df = pd.read_csv('churn.csv', ';')
print(df.describe())
