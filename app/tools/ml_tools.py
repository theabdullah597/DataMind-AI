
import pandas as pd


def validate_target_column(
    df: pd.DataFrame,
    target_column: str
) -> dict:
    """
    Validate the target column before ML training.
    """

    if df.empty:
        raise ValueError("The dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    target = df[target_column]

    missing_count = int(target.isna().sum())
    unique_count = int(target.nunique())

    if target.isna().all():
        raise ValueError(
            "The target column contains only missing values."
        )

    if unique_count < 2:
        raise ValueError(
            "The target column must contain at least "
            "two distinct values for supervised learning."
        )

    return {
        "target_column": target_column,
        "total_rows": int(len(df)),
        "missing_values": missing_count,
        "available_values": int(target.notna().sum()),
        "unique_values": unique_count,
        "valid": True
    }



from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_ml_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Validate, split, and preprocess a dataset
    for supervised machine learning.
    """

    validate_target_column(df, target_column)

    # Remove rows with missing target values.
    df = df.dropna(subset=[target_column]).copy()

    # Separate features and target.
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.shape[1] == 0:
        raise ValueError("The dataset has no feature columns.")

    # Identify numerical and categorical features.
    numerical_columns = (
        X.select_dtypes(include=["number"])
        .columns.tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    )

    unsupported_columns = [
        col for col in X.columns
        if col not in numerical_columns
        and col not in categorical_columns
    ]

    if unsupported_columns:
        raise ValueError(
            "Unsupported feature types detected: "
            f"{unsupported_columns}. "
            "Convert or remove these columns first."
        )

    # Create preprocessing pipelines.
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numerical_columns:
        transformers.append((
            "numerical",
            numerical_pipeline,
            numerical_columns
        ))

    if categorical_columns:
        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Use stratification for classification-like targets.
    # This is a preliminary heuristic; the task should
    # eventually be explicitly selected by the user/agent.
    unique_count = y.nunique()
    is_classification_candidate = (
        not pd.api.types.is_numeric_dtype(y)
        or unique_count <= 10
    )

    stratify = None

    if is_classification_candidate:
        class_counts = y.value_counts()

        if (
            len(class_counts) >= 2
            and class_counts.min() >= 2
        ):
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Fit preprocessing ONLY on training features.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the already-fitted preprocessing to test features.
    X_test_processed = preprocessor.transform(X_test)

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    }
