import pandas as pd

def inspect_dataset(file_path: str)->dict:
    if file_path.endswith('csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only csv or xlsx files are supported")

    rows,columns=df.shape
    column_names=df.columns.tolist()
    data_types={
        column:str(dtype)
    for column,dtype in df.dtypes.items()
    }
    missing_values={
        column:int(value)
        for column,value in df.isnull().sum().items()
    }
    columns_with_missing = {
        column: value
        for column, value in missing_values.items()
        if value > 0
    }
    duplicate_rows=int(df.duplicated().sum())
    unique_values={
        column:int(value)
        for column, value in   df.nunique().items()
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
    numerical_statistics={}
    if numerical_cols:
        numerical_statistics = (
            df[numerical_cols]
            .describe()
            .round(2)
            .to_dict()
        )
    return {
        "rows": rows,
        "columns": columns,
        "column_names": column_names,
        "data_types": data_types,
        "missing_values": missing_values,
        "columns_with_missing": columns_with_missing,
        "duplicate_rows": duplicate_rows,
        "unique_values": unique_values,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "numerical_statistics": numerical_statistics,
    }

result = inspect_dataset("data.csv")

print(result)
