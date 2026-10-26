import pandas as pd
import numpy as np

def remove_duplicates(df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicated rows from the DataFrame.
    keep: 'first', 'last', or False
    """
    return df.drop_duplicates(keep=keep)


def handle_missing_values(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    """
    Handle missing values based on a column-specific strategy dictionary.
    Strategy format: {"column_name": "mean" | "median" | "mode" | "drop" | constant_value}
    """
    df_clean = df.copy()
    
    for col, strat in strategy.items():
        if col not in df_clean.columns:
            continue
            
        if strat == "drop":
            df_clean = df_clean.dropna(subset=[col])
        elif strat == "mean":
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strat == "median":
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strat == "mode":
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
        else:
            # Assume constant value fill
            df_clean[col] = df_clean[col].fillna(strat)
            
    return df_clean


def correct_data_types(df: pd.DataFrame, type_dict: dict) -> pd.DataFrame:
    """
    Correct data types based on a dictionary mapping.
    Format: {"column_name": "int" | "float" | "str" | "bool" | "datetime"}
    """
    df_clean = df.copy()
    
    for col, dtype in type_dict.items():
        if col not in df_clean.columns:
            continue
            
        try:
            if dtype == "datetime":
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            elif dtype == "int":
                # Int64 allows for NaN values in pandas
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')
            elif dtype == "float":
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('float64')
            elif dtype == "str":
                df_clean[col] = df_clean[col].astype(str)
            elif dtype == "bool":
                df_clean[col] = df_clean[col].astype(bool)
        except Exception:
            pass # Skip if conversion fails
            
    return df_clean


def extract_datetime_features(df: pd.DataFrame, datetime_columns: list) -> pd.DataFrame:
    """
    Extract year, month, day, day_of_week from datetime columns.
    Original columns will be dropped.
    """
    df_clean = df.copy()
    
    for col in datetime_columns:
        if col in df_clean.columns:
            # ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                
            df_clean[f"{col}_year"] = df_clean[col].dt.year
            df_clean[f"{col}_month"] = df_clean[col].dt.month
            df_clean[f"{col}_day"] = df_clean[col].dt.day
            df_clean[f"{col}_dayofweek"] = df_clean[col].dt.dayofweek
            
            df_clean = df_clean.drop(columns=[col])
            
    return df_clean
