import pandas as pd

from tools.eda_tools import analyze_numerical_columns
from tools.eda_tools import analyze_categorical_columns

df = pd.read_csv("tools/data.csv")

numerical_results = analyze_numerical_columns(df)
categorical_results = analyze_categorical_columns(df)
print("Numerical Columns\n")
print(numerical_results)
print("\nCategorical Analysis:")
print(categorical_results)
