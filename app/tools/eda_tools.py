import pandas as pd
from pandas import value_counts


def analyze_numerical_columns(df:pd.DataFrame)->dict:
    numerical_columns=(df.select_dtypes(include=["number"]).columns.tolist())
    result={}
    for column in numerical_columns:
        series=df[column].dropna()
        result[column]={
            "count":int(series.count()),
            "mean":round(float(series.mean()),2),
            "median":round(float(series.median()),2),
            "std":round(float(series.std()),2),
            "min":round(float(series.min()),2),
            "max":round(float(series.max()),2),
            "q1":round(float(series.quantile(0.25)),2),
            "q3":round(float(series.quantile(0.75)),1),
            "skewness":round(float(series.skew()),2)
        }

    return result
def analyze_categorical_columns(df: pd.DataFrame) -> dict:
    """
    Analyze categorical columns in a dataset.
    """

    categorical_columns = (
        df.select_dtypes(include=["object", "category"])
        .columns
        .tolist()
    )

    results = {}

    for column in categorical_columns:

        series = df[column].dropna()

        value_counts = series.value_counts()

        total_values = len(series)

        frequencies = {
            str(category): int(count)
            for category, count in value_counts.items()
        }

        percentages = {
            str(category): round((count / total_values) * 100, 2)
            for category, count in value_counts.items()
        }

        results[column] = {
            "unique_categories": int(series.nunique()),
            "frequencies": frequencies,
            "percentages": percentages,
            "most_common": str(value_counts.index[0])
            if not value_counts.empty else None,
            "most_common_count": int(value_counts.iloc[0])
            if not value_counts.empty else 0
        }

    return results

def analyze_target_column(df:pd.DataFrame,target_column:str)->dict:
    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column} is not present in the dataset"
        )
    series=df[target_column].dropna()
    unique_values=series.nunique()
    if (series.dtype=='object' or str(series.dtype)=="category" or unique_values<=10):
        problem_type="classification"
        value_counts=series.value_counts()
        distribution={
            str(category):int(count)
            for category,count in value_counts.items()
        }
        percentages={
            str(category):round((count/len(series))*100,2)
                                for category,count in value_counts.items()
        }
        return {
            "target_column":target_column,
            "problem_type":problem_type,
            "unique_values":int(unique_values),
            "distribution":distribution,
            "Percentages":percentages,

        }
    else:
        problem_type="regression"
        return {
            "target_column":target_column,
            "problem_type":problem_type,
            "unique_values":int(unique_values),
            "mean":round(float(series.mean()),2),
            "median":round(float(series.median()),2),
            "min":round(float(series.min()),2),
            "max":round(float(series.min()),2),
            "std":round(float(series.std()),2),

        }