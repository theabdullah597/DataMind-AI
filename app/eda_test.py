import pandas as pd

from tools.eda_tools import analyze_numerical_columns
from tools.eda_tools import analyze_categorical_columns
from tools.eda_tools import analyze_target_column
from tools.eda_tools import analyze_correlations
from tools.eda_tools import detect_outliers
df = pd.read_csv("tools/data.csv")

numerical_results = analyze_numerical_columns(df)
categorical_results = analyze_categorical_columns(df)
print("Numerical Columns\n")
print(numerical_results)
print("\nCategorical Analysis:")
print(categorical_results)
target_results = analyze_target_column(
    df,
    "SalePrice"
)

print("\nTarget Analysis:")
print(target_results)
correlation_results = analyze_correlations(df)

print("\nCorrelation Analysis:")
print(correlation_results)

outlier_results = detect_outliers(df)

print("\nOutlier Analysis:")
print(outlier_results)