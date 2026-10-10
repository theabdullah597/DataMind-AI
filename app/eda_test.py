# import pandas as pd
#
# from tools.eda_tools import analyze_numerical_columns
# from tools.eda_tools import analyze_categorical_columns
# from tools.eda_tools import analyze_target_column
# from tools.eda_tools import analyze_correlations
# from tools.eda_tools import detect_outliers
# from tools.eda_tools import generate_eda_insights
# df = pd.read_csv("tools/data.csv")
#
# numerical_results = analyze_numerical_columns(df)
# categorical_results = analyze_categorical_columns(df)
# print("Numerical Columns\n")
# print(numerical_results)
# print("\nCategorical Analysis:")
# print(categorical_results)
# target_results = analyze_target_column(
#     df,
#     "SalePrice"
# )
#
# print("\nTarget Analysis:")
# print(target_results)
# correlation_results = analyze_correlations(df)
#
# print("\nCorrelation Analysis:")
# print(correlation_results)
#
# outlier_results = detect_outliers(df)
#
# print("\nOutlier Analysis:")
# print(outlier_results)
#
# insights = generate_eda_insights(
#     numerical_results,
#     categorical_results,
#     target_results,
#     correlation_results,
#     outlier_results
# )
#
# print("\nAutomatic EDA Insights:")
#
# for insight in insights:
#     print("-", insight)
#



import pandas as pd
from pathlib import Path

from app.tools.ml_tools import validate_target_column

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Build the absolute path to data.csv
DATA_PATH = PROJECT_ROOT / "app/data.csv"

print("Looking for dataset at:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

target_column = "Street"

result = validate_target_column(df, target_column)

print("Target Validation Results:")
print(result)

from app.tools.ml_tools import prepare_ml_data
prepared = prepare_ml_data(df, target_column)

print("\nPreprocessing Results:")
print("Training rows:", prepared["train_rows"])
print("Testing rows:", prepared["test_rows"])
print("Numerical features:", prepared["numerical_columns"])
print("Categorical features:", prepared["categorical_columns"])
print("Training matrix shape:", prepared["X_train"].shape)
print("Testing matrix shape:", prepared["X_test"].shape)