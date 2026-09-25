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

def analyze_correlations(df: pd.DataFrame) -> dict:
    """
    Analyze correlations between numerical columns.
    """

    numerical_columns = (
        df.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    if len(numerical_columns) < 2:
        return {
            "correlation_matrix": {},
            "strong_correlations": []
        }

    correlation_matrix = (
        df[numerical_columns]
        .corr()
        .round(2)
    )

    strong_correlations = []

    for i in range(len(numerical_columns)):

        for j in range(i + 1, len(numerical_columns)):

            column_1 = numerical_columns[i]
            column_2 = numerical_columns[j]

            correlation = correlation_matrix.loc[
                column_1,
                column_2
            ]

            if abs(correlation) >= 0.7:

                strong_correlations.append({
                    "column_1": column_1,
                    "column_2": column_2,
                    "correlation": float(correlation)
                })

    return {
        "correlation_matrix": correlation_matrix.to_dict(),
        "strong_correlations": strong_correlations
    }

def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Detect outliers in numerical columns using the IQR method.
    """

    numerical_columns = (
        df.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    results = {}

    for column in numerical_columns:

        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        results[column] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "outlier_count": int(len(outliers)),
            "outlier_percentage": round(
                (len(outliers) / len(series)) * 100,
                2
            )
        }

    return results