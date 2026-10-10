
import pandas as pd


def validate_target_column(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto"
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

    if task_type not in ["auto", "classification", "regression"]:
        raise ValueError("task_type must be one of 'auto', 'classification', 'regression'.")
        
    if task_type == "regression" and not pd.api.types.is_numeric_dtype(target):
        raise ValueError("Regression requires a numerical target column.")

    return {
        "target_column": target_column,
        "total_rows": int(len(df)),
        "missing_values": missing_count,
        "available_values": int(target.notna().sum()),
        "unique_values": unique_count,
        "valid": True
    }



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_ml_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Validate, split, and preprocess a dataset
    for supervised machine learning.
    """
    
    if not (0 < test_size < 1):
        raise ValueError("test_size must be strictly between 0 and 1.")

    validate_target_column(df, target_column, task_type)

    # Remove rows with missing target values.
    df = df.dropna(subset=[target_column]).copy()

    # Separate features and target.
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.shape[1] == 0:
        raise ValueError("The dataset has no feature columns.")

    # Drop constant features
    constant_features = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)

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
    
    # Drop high-cardinality categorical features
    high_cardinality = [col for col in categorical_columns if X[col].nunique() > 100]
    if high_cardinality:
        X = X.drop(columns=high_cardinality)
        categorical_columns = [col for col in categorical_columns if col not in high_cardinality]

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

    # Determine if task is classification
    is_classification = False
    if task_type == "classification":
        is_classification = True
    elif task_type == "regression":
        is_classification = False
    else:
        unique_count = y.nunique()
        is_classification = (not pd.api.types.is_numeric_dtype(y) or unique_count <= 10)

    stratify = None

    if is_classification:
        class_counts = y.value_counts()
        
        # Ensure stratify is only set if all classes have at least 2 samples
        if len(class_counts) >= 2 and class_counts.min() >= 2:
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



def train_regression_models(prepared: dict, primary_metric: str = "rmse") -> dict:
    """
    Train and compare several regression models.
    """

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    # Regression requires numerical target values.
    if not (
        np.issubdtype(y_train.dtype, np.number)
        and np.issubdtype(y_test.dtype, np.number)
    ):
        raise ValueError(
            "Regression requires a numerical target column."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain at least two distinct target values."
        )

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Sample points for scatter plot visualization
        max_points = 200
        n_points = len(y_test)
        indices = np.random.choice(n_points, min(max_points, n_points), replace=False)
        y_test_array = np.array(y_test)
        
        plot_data = []
        for idx in indices:
            actual = float(y_test_array[idx])
            pred = float(predictions[idx])
            plot_data.append({
                "actual": round(actual, 4), 
                "predicted": round(pred, 4), 
                "residual": round(actual - pred, 4)
            })

        results[name] = {
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "r2_score": round(float(r2), 4),
            "plot_data": plot_data
        }

        trained_models[name] = model

    best_model_name = None
    best_score = float('inf') if primary_metric in ["mae", "mse", "rmse"] else float('-inf')
    
    for name, res in results.items():
        score = res.get(primary_metric, res["rmse"])
        if primary_metric in ["mae", "mse", "rmse"]:
            if score < best_score:
                best_score = score
                best_model_name = name
        else:
            if score > best_score:
                best_score = score
                best_model_name = name
                
    recommendation = {
        "model": best_model_name,
        "reason": f"Selected based on best {primary_metric} score of {best_score}"
    }

    return {
        "results": results,
        "trained_models": trained_models,
        "recommendation": recommendation
    }


def train_classification_models(prepared: dict, primary_metric: str = "accuracy") -> dict:
    """
    Train and compare several classification models.
    """

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain at least two distinct target values for classification."
        )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees Classifier": ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    trained_models = {}
    
    # Determine averaging method for metrics
    avg_method = 'weighted'

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average=avg_method, zero_division=0)
        rec = recall_score(y_test, predictions, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, predictions, average=avg_method, zero_division=0)

        cm = confusion_matrix(y_test, predictions)
        
        roc_auc = None
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_test)
                if y_train.nunique() == 2:
                    roc_auc = roc_auc_score(y_test, probas[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, probas, multi_class="ovr")
            except Exception:
                pass

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist()
        }
        
        if roc_auc is not None:
            results[name]["roc_auc"] = round(float(roc_auc), 4)

        trained_models[name] = model

    best_model_name = None
    best_score = float('-inf')
    
    for name, res in results.items():
        score = res.get(primary_metric, res["accuracy"])
        if score > best_score:
            best_score = score
            best_model_name = name

    recommendation = {
        "model": best_model_name,
        "reason": f"Selected based on best {primary_metric} score of {best_score}"
    }

    return {
        "results": results,
        "trained_models": trained_models,
        "recommendation": recommendation
    }
