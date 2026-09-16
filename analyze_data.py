import pandas as pd

df=pd.read_csv("backend/uploads/data.csv")
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.isnull())
print(df.isnull().sum())
print(df.duplicated(keep='first'))
print(df.nunique())
# 1. Identify Numerical Columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

# 2. Identify Categorical Columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)

print("Last Column",df.iloc[:,-1])