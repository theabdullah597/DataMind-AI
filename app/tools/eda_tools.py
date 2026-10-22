import numpy as np
import pandas as pd


def analyze_numerical_columns(df: pd.DataFrame) -> dict:
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    result = {}
    for column in numerical_columns:
        series = df[column].dropna()
        if series.empty:
            continue

        c_count = int(series.count())
        c_mean = float(series.mean())
        c_median = float(series.median())
        c_std = float(series.std()) if len(series) > 1 else 0.0
        c_min = float(series.min())
        c_max = float(series.max())
        c_q1 = float(series.quantile(0.25))
        c_q3 = float(series.quantile(0.75))
        c_skew = float(series.skew()) if len(series) > 2 else 0.0
        iqr = c_q3 - c_q1

        # Classify distribution shape in plain language
        if abs(c_skew) < 0.5:
            shape = "Normal (Bell-shaped)"
            shape_desc = "Values are symmetrically spread around the center. Ideal for standard statistical analysis."
        elif c_skew >= 0.5:
            shape = "Right-skewed (Tail on Highs)"
            shape_desc = "Most data points cluster at lower numbers, with a tail of higher values stretching out."
        else:
            shape = "Left-skewed (Tail on Lows)"
            shape_desc = "Most data points cluster at higher numbers, with a tail of lower values stretching down."

        # Compute histogram bins for visual charts
        bins_data = []
        try:
            if c_min == c_max:
                bins_data = [{
                    "bin": f"{round(c_min, 1)}",
                    "range": f"[{round(c_min, 1)}]",
                    "count": c_count,
                    "min": round(c_min, 2),
                    "max": round(c_max, 2)
                }]
            else:
                num_bins = min(10, max(5, series.nunique()))
                counts, bin_edges = np.histogram(series, bins=num_bins)
                for i in range(len(counts)):
                    b_start = round(float(bin_edges[i]), 2)
                    b_end = round(float(bin_edges[i + 1]), 2)
                    bins_data.append({
                        "bin": f"{b_start} - {b_end}",
                        "range": f"{b_start} to {b_end}",
                        "count": int(counts[i]),
                        "percentage": round((int(counts[i]) / (c_count or 1)) * 100, 1),
                        "start": b_start,
                        "end": b_end
                    })
        except Exception:
            bins_data = []

        # Plain-English summary for everyday users
        simple_summary = (
            f"Typical values center around {round(c_median, 2)} (average is {round(c_mean, 2)}), "
            f"spanning from a minimum of {round(c_min, 2)} to a maximum of {round(c_max, 2)}."
        )

        result[column] = {
            "count": c_count,
            "mean": round(c_mean, 2),
            "median": round(c_median, 2),
            "std": round(c_std, 2),
            "min": round(c_min, 2),
            "max": round(c_max, 2),
            "q1": round(c_q1, 2),
            "q3": round(c_q3, 2),
            "iqr": round(iqr, 2),
            "skewness": round(c_skew, 2),
            "shape": shape,
            "shape_desc": shape_desc,
            "simple_summary": simple_summary,
            "histogram": bins_data
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
        if series.empty:
            continue

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
            "most_common": str(value_counts.index[0]) if not value_counts.empty else None,
            "most_common_count": int(value_counts.iloc[0]) if not value_counts.empty else 0
        }

    return results


def analyze_target_column(df: pd.DataFrame, target_column: str, task_type: str = "auto") -> dict:
    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column} is not present in the dataset"
        )
    series = df[target_column].dropna()
    unique_values = series.nunique()

    is_classification = False
    if task_type == "classification":
        is_classification = True
    elif task_type == "regression":
        is_classification = False
    else:
        is_classification = (series.dtype == 'object' or str(series.dtype) == "category" or unique_values <= 10)

    if is_classification:
        problem_type = "classification"
        value_counts = series.value_counts()
        distribution = {
            str(category): int(count)
            for category, count in value_counts.items()
        }
        percentages = {
            str(category): round((count / len(series)) * 100, 2)
            for category, count in value_counts.items()
        }
        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "unique_values": int(unique_values),
            "distribution": distribution,
            "Percentages": percentages,
        }
    else:
        problem_type = "regression"
        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "unique_values": int(unique_values),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),  # Fixed bug: was series.min()
            "std": round(float(series.std()), 2),
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
            correlation = correlation_matrix.loc[column_1, column_2]

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


def generate_eda_insights(
    numerical_results: dict,
    categorical_results: dict,
    target_results: dict,
    correlation_results: dict,
    outlier_results: dict
) -> list:
    """
    Generate actionable, plain-English insights from EDA results that anyone can understand.
    Returns structured insight items with title, plain-English description, recommendation, and category.
    """
    insights = []

    # -------------------------
    # Target insights first
    # -------------------------
    if target_results and "error" not in target_results:
        problem_type = target_results.get("problem_type")
        target_col = target_results.get("target_column", "Target")

        if problem_type == "classification":
            percentages = target_results.get("percentages") or target_results.get("Percentages") or {}
            if percentages:
                largest_class_key = max(percentages, key=percentages.get)
                largest_class_pct = percentages[largest_class_key]
                if largest_class_pct >= 70:
                    insights.append({
                        "category": "Target Variable",
                        "badge": "Warning",
                        "title": f"Target Imbalance: '{largest_class_key}' dominates '{target_col}'",
                        "description": (
                            f"Over {largest_class_pct}% of the data belongs to class '{largest_class_key}'. "
                            f"An imbalanced target can mislead models into guessing the dominant class every time."
                        ),
                        "recommendation": "Use stratified sampling, class-weighting, or evaluate with F1-score / ROC-AUC rather than raw accuracy.",
                        "technical_detail": f"Class '{largest_class_key}' comprises {largest_class_pct}% of target values."
                    })
                else:
                    insights.append({
                        "category": "Target Variable",
                        "badge": "Info",
                        "title": f"Balanced Classification Target: '{target_col}'",
                        "description": f"The target categories are reasonably well distributed across classes.",
                        "recommendation": "Standard accuracy and cross-validation will be reliable evaluation metrics.",
                        "technical_detail": f"Target has {target_results.get('unique_values')} unique classes."
                    })
        elif problem_type == "regression":
            mean_val = target_results.get("mean", 0)
            median_val = target_results.get("median", 0)
            insights.append({
                "category": "Target Variable",
                "badge": "Info",
                "title": f"Continuous Target for Regression: '{target_col}'",
                "description": f"Values span continuously with an average of {mean_val} and median of {median_val}.",
                "recommendation": "Standard regression algorithms (Linear Regression, Random Forest Regressor) can be trained to predict this quantity.",
                "technical_detail": f"Min: {target_results.get('min')}, Max: {target_results.get('max')}, Std: {target_results.get('std')}"
            })

    # -------------------------
    # Outlier insights
    # -------------------------
    for column, data in outlier_results.items():
        percentage = data.get("outlier_percentage", 0)
        count = data.get("outlier_count", 0)
        if percentage >= 5:
            insights.append({
                "category": "Data Quality",
                "badge": "Warning" if percentage >= 10 else "Alert",
                "title": f"High Potential Outliers in '{column}'",
                "description": (
                    f"'{column}' has {count} data points ({percentage}%) that sit far outside the normal range "
                    f"(below {data.get('lower_bound')} or above {data.get('upper_bound')})."
                ),
                "recommendation": "Check whether these are extreme but legitimate real-world cases or data entry errors before training linear models.",
                "technical_detail": f"IQR: {data.get('iqr')}, Outlier count: {count} ({percentage}%)"
            })

    # -------------------------
    # Correlation insights
    # -------------------------
    strong_correlations = correlation_results.get("strong_correlations", [])
    for correlation in strong_correlations:
        col_1 = correlation["column_1"]
        col_2 = correlation["column_2"]
        val = correlation["correlation"]
        direction = "positive" if val > 0 else "negative"
        relationship_text = (
            f"as '{col_1}' rises, '{col_2}' also tends to rise significantly"
            if val > 0 else
            f"as '{col_1}' rises, '{col_2}' tends to drop significantly"
        )
        insights.append({
            "category": "Correlation",
            "badge": "Tip",
            "title": f"Strong {direction.capitalize()} Relationship: '{col_1}' & '{col_2}'",
            "description": f"Correlation is {val}: {relationship_text}.",
            "recommendation": "Because these two columns carry overlapping information (multicollinearity), you may consider keeping just one of them.",
            "technical_detail": f"Pearson Correlation: {val}"
        })

    # -------------------------
    # Numerical distribution insights
    # -------------------------
    for column, data in numerical_results.items():
        skewness = data.get("skewness", 0)
        mean_val = data.get("mean", 0)
        median_val = data.get("median", 0)

        if skewness > 1.2:
            insights.append({
                "category": "Distribution",
                "badge": "Info",
                "title": f"Right-Skewed Distribution in '{column}'",
                "description": (
                    f"Most records have lower values (median around {median_val}), "
                    f"but a few unusually high values pull the average up to {mean_val}."
                ),
                "recommendation": "Tree-based models like Random Forest handle this naturally. For linear models, a log transformation may boost accuracy.",
                "technical_detail": f"Skewness score: {skewness}"
            })
        elif skewness < -1.2:
            insights.append({
                "category": "Distribution",
                "badge": "Info",
                "title": f"Left-Skewed Distribution in '{column}'",
                "description": (
                    f"Most records have higher values (median around {median_val}), "
                    f"with a tail of lower values stretching down."
                ),
                "recommendation": "Inspect low-end extreme values to see if they represent missing value codes (like 0 or -1).",
                "technical_detail": f"Skewness score: {skewness}"
            })

    # -------------------------
    # Categorical insights
    # -------------------------
    for column, data in categorical_results.items():
        percentages = data.get("percentages", {})
        if percentages:
            largest_cat = max(percentages, key=percentages.get)
            largest_pct = percentages[largest_cat]
            if largest_pct >= 80:
                insights.append({
                    "category": "Distribution",
                    "badge": "Alert",
                    "title": f"Heavy Category Concentration in '{column}'",
                    "description": (
                        f"'{largest_cat}' accounts for {largest_pct}% of all entries in '{column}'. "
                        f"Very little variety exists in this feature."
                    ),
                    "recommendation": "Since nearly all records share this same value, this column might not provide much predictive power to ML models.",
                    "technical_detail": f"Dominant category '{largest_cat}' has {largest_pct}% share."
                })

    # Add text property to each insight for backward compatibility with string display
    for item in insights:
        item["text"] = f"{item['title']}: {item['description']}"

    return insights