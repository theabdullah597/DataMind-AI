import pandas as pd


def inspect_dataset(file_path: str) -> dict:


    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    else:
        raise ValueError("Only CSV and Excel files are supported.")

    rows, columns = df.shape

    column_names = df.columns.tolist()

    data_types = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }
    missing_counts=df.isnull().sum()
    missing_values = {
        column: int(value)
        for column, value in df.isnull().sum().items()
    }
    missing_percentage={
        column:round(value/rows*100,2)
        for column,value in missing_counts.items()
    }

    # Columns that actually contain missing values
    columns_with_missing = {
        column: value
        for column, value in missing_values.items()
        if value > 0
    }

    duplicate_rows = int(df.duplicated().sum())
    duplicate_percentage={round(duplicate_rows/rows*100,2)
                         if duplicate_rows>0 else 0

    }
    unique_values = {
        column: int(value)
        for column, value in df.nunique().items()
    }

    numerical_cols = (
        df.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_cols = (
        df.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numerical_statistics = {}

    if numerical_cols:
        numerical_statistics = (
            df[numerical_cols]
            .describe()
            .round(2)
            .to_dict()
        )
    empty_col=[column for column in column_names
               if df[column].isnull().all()]
    constant_columns = [
        column
        for column in column_names
        if df[column].nunique(dropna=False) <= 1
    ]

    possible_id_columns=[]
    if rows>0:
        for column in column_names:
            unique_ratio=(df[column].nunique(dropna=True)/rows)
            column_name=column.lower()
            if(unique_ratio>=0.95 and ("id" in column_name or "uuid" in column_name or "code" in column_name)):
                possible_id_columns.append(column)

    high_cardinality_columns=[]
    if rows>0:
        for column in categorical_cols:
            unique_ratio=(df[column].nunique(dropna=True)/rows)
            if unique_ratio>0.50:
                high_cardinality_columns.append(column)

    high_missing_columns=[column for column,percentage in missing_percentage.items()
                          if percentage>50]
    possible_target_column=[]
    for column in column_names:
        unique_count=df.nunique(dropna=True)
        if unique_count is not None ==2:
            possible_target_column.append(column)

    warnings=[]
    if duplicate_rows>0:
        warnings.append(f"Dataset contain {duplicate_rows} duplicated rows")

    if empty_col:
        warnings.append(f"Empty column detected: {empty_col} ")
    if constant_columns:
        warnings.append(f"Constant column detected: {constant_columns} ")
    if high_missing_columns:
        warnings.append(f"High missing column detected: {high_missing_columns} ")
    if high_cardinality_columns:
        warnings.append(f"High cardinality categorical column detected: {high_cardinality_columns} ")

    return {

        "dataset": {
            "rows": rows,
            "columns": columns,
            "column_names": column_names
        },

        "data_types": data_types,

        "missing_values": {
            "counts": missing_values,
            "percentages": missing_percentage,
            "columns_with_missing": columns_with_missing
        },

        "duplicates": {
            "count": duplicate_rows,
            "percentage": duplicate_percentage
        },

        "unique_values": unique_values,

        "column_types": {
            "numerical": numerical_cols,
            "categorical": categorical_cols
        },

        "numerical_statistics": numerical_statistics,

        "data_quality": {
            "empty_columns": empty_col,
            "constant_columns": constant_columns,
            "possible_id_columns": possible_id_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "high_missing_columns": high_missing_columns,
            "warnings": warnings
        },

        "possible_target_columns": possible_target_column
    }