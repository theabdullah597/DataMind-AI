import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    is_numeric = pd.api.types.is_numeric_dtype(target)

    if task_type == "regression" and not is_numeric:
        raise ValueError("Regression requires a numerical target column.")

    # Guard: Detect high-cardinality non-numeric targets (e.g., Description, Title, ID fields)
    is_classification_task = (
        task_type == "classification" or
        (task_type == "auto" and (not is_numeric or unique_count <= 10))
    )
    if is_classification_task and not is_numeric and unique_count > 50:
        candidates = [
            c for c in df.columns
            if c != target_column and (df[c].nunique() <= 30 or (pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() >= 2))
        ]
        suggestion = f" (Suggested candidate target columns: {', '.join(candidates[:3])})" if candidates else ""
        raise ValueError(
            f"Target column '{target_column}' has {unique_count} distinct text values. "
            f"This looks like an ID, description, or free-text column, not a classification target. "
            f"Supervised classification requires a target with categories (typically <= 50 distinct classes). "
            f"Please select a valid categorical target column.{suggestion}"
        )

    return {
        "target_column": target_column,
        "total_rows": int(len(df)),
        "missing_values": missing_count,
        "available_values": int(target.notna().sum()),
        "unique_values": unique_count,
        "valid": True
    }


def prepare_ml_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Validate, split, and preprocess a dataset for supervised machine learning,
    with automatic TF-IDF extraction for natural text columns.
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

    # Drop all-NaN features
    all_nan = [col for col in X.columns if X[col].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)

    # Drop constant features
    constant_features = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)

    # Convert datetime columns into numerical components (year, month, day)
    datetime_columns = X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    for col in datetime_columns:
        try:
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
        except Exception:
            pass
        X = X.drop(columns=[col])

    # Identify numerical and categorical features.
    numerical_columns = (
        X.select_dtypes(include=["number"])
        .columns.tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist()
    )

    # Convert any remaining columns to string category
    remaining_cols = [c for c in X.columns if c not in numerical_columns and c not in categorical_columns]
    for c in remaining_cols:
        X[c] = X[c].astype(str)
        categorical_columns.append(c)

    # Separate high-cardinality text columns from discrete categories and IDs
    text_columns = []
    discrete_categorical = []

    for col in categorical_columns:
        X[col] = X[col].fillna("").astype(str)
        n_unique = int(X[col].nunique())
        avg_len = float(X[col].str.len().mean())
        has_spaces = float((X[col].str.count(" ") >= 1).mean()) > 0.3

        if n_unique > 50 and (avg_len > 10 or has_spaces):
            # Rich text field (e.g., Title, Description, Review, Body)
            text_columns.append(col)
        elif n_unique > 100:
            # High cardinality ID or hash without sentence structure (e.g. UUID, customer_id)
            X = X.drop(columns=[col])
        else:
            discrete_categorical.append(col)

    categorical_columns = discrete_categorical

    # Create preprocessing pipelines.
    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append((
            "numerical",
            numerical_pipeline,
            numerical_columns
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns
        ))

    # Process text columns with TF-IDF Vectorizer
    for tcol in text_columns:
        transformers.append((
            f"text_{tcol}",
            TfidfVectorizer(max_features=50, stop_words="english"),
            tcol
        ))

    if not transformers:
        raise ValueError("The dataset has no valid feature columns left after preprocessing.")

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Determine if task is classification
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

    # Convert sparse matrices to dense arrays for fast, universal model compatibility
    if issparse(X_train_processed):
        X_train_processed = X_train_processed.toarray()
    if issparse(X_test_processed):
        X_test_processed = X_test_processed.toarray()

    # Extract feature names for explainability
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = numerical_columns + categorical_columns + text_columns

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "text_columns": text_columns,
        "feature_names": feature_names,
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    }


def diagnose_regression_fit(train_r2: float, test_r2: float, train_rmse: float, test_rmse: float) -> dict:
    """
    Automated diagnostic engine to detect overfitting, underfitting, or good fit for regression models.
    """
    gap = train_r2 - test_r2

    if train_r2 < 0.35 and test_r2 < 0.35:
        status = "Underfitting"
        badge = "Underfitting (High Bias)"
        severity = "High"
        summary = "Model is too simple and failed to capture relationships in this data."
        detail = (
            f"Both Training R² ({round(train_r2, 2)}) and Test R² ({round(test_r2, 2)}) are low. "
            f"The model cannot capture the patterns in this dataset."
        )
        action = "Try nonlinear algorithms (Random Forest, Extra Trees), create new interaction features, or collect more informative variables."
    elif gap > 0.15:
        status = "Overfitting"
        badge = "Overfitting (High Variance)"
        severity = "High" if gap > 0.30 else "Moderate"
        summary = "Model memorized training noise and performs significantly worse on new test data."
        detail = (
            f"Training R² is high ({round(train_r2, 2)}), but Test R² drops sharply to ({round(test_r2, 2)}), "
            f"revealing a {round(gap * 100, 1)}% generalization gap."
        )
        action = "Reduce model complexity (e.g. limit max tree depth, increase min samples per leaf), introduce regularization, or gather more training data."
    else:
        status = "Good Fit"
        badge = "Good Fit (Balanced)"
        severity = "Optimal"
        summary = "Model generalizes well to unseen data without memorizing training noise."
        detail = (
            f"Training R² ({round(train_r2, 2)}) and Test R² ({round(test_r2, 2)}) are tightly aligned "
            f"with a minimal gap of only {round(abs(gap) * 100, 1)}%."
        )
        action = "This model is well-balanced and safe to deploy for batch inference."

    return {
        "status": status,
        "badge": badge,
        "severity": severity,
        "generalization_gap": round(float(gap), 4),
        "train_score": round(float(train_r2), 4),
        "test_score": round(float(test_r2), 4),
        "metric_name": "R² Score",
        "summary": summary,
        "detail": detail,
        "action": action
    }


def diagnose_classification_fit(train_acc: float, test_acc: float, train_f1: float, test_f1: float) -> dict:
    """
    Automated diagnostic engine to detect overfitting, underfitting, or good fit for classification models.
    """
    gap = train_acc - test_acc

    if train_acc < 0.10 and test_acc < 0.10:
        status = "Underfitting"
        badge = "Invalid Target / Text Column"
        severity = "Critical"
        summary = "Accuracy is near 0%. The selected target column contains unique text or IDs rather than discrete categories."
        detail = (
            f"Training Accuracy ({round(train_acc * 100, 1)}%) and Test Accuracy ({round(test_acc * 100, 1)}%) "
            f"are near zero. Standard classification algorithms cannot learn when almost every row has a unique text label."
        )
        action = "Select a discrete categorical target column (e.g. 'Class Index', 'Category', 'Status') with <= 50 distinct classes."
    elif train_acc < 0.60 and test_acc < 0.60:
        status = "Underfitting"
        badge = "Underfitting (High Bias)"
        severity = "High"
        summary = "Model is too simple and struggles to classify both training and test samples."
        detail = (
            f"Training Accuracy ({round(train_acc * 100, 1)}%) and Test Accuracy ({round(test_acc * 100, 1)}%) "
            f"are both low. The model cannot separate the classes effectively."
        )
        action = "Switch to more expressive algorithms, check feature correlations, or engineer informative features."
    elif gap > 0.12:
        status = "Overfitting"
        badge = "Overfitting (High Variance)"
        severity = "High" if gap > 0.25 else "Moderate"
        summary = "Model memorized training examples and struggles on unseen records."
        detail = (
            f"Training Accuracy is {round(train_acc * 100, 1)}%, but Test Accuracy falls to {round(test_acc * 100, 1)}%, "
            f"leaving a notable {round(gap * 100, 1)}% generalization gap."
        )
        action = "Prune tree depth, increase regularization (higher C for Logistic Regression), or use cross-validation to select simpler parameters."
    else:
        status = "Good Fit"
        badge = "Good Fit (Balanced)"
        severity = "Optimal"
        summary = "Model generalizes reliably to new unseen test examples."
        detail = (
            f"Training Accuracy ({round(train_acc * 100, 1)}%) and Test Accuracy ({round(test_acc * 100, 1)}%) "
            f"are close (gap: {round(abs(gap) * 100, 1)}%)."
        )
        action = "This model is well-calibrated and recommended for batch predictions."

    return {
        "status": status,
        "badge": badge,
        "severity": severity,
        "generalization_gap": round(float(gap), 4),
        "train_score": round(float(train_acc), 4),
        "test_score": round(float(test_acc), 4),
        "metric_name": "Accuracy",
        "summary": summary,
        "detail": detail,
        "action": action
    }


def train_regression_models(prepared: dict, primary_metric: str = "rmse") -> dict:
    """
    Train and compare several regression models with both training and testing evaluations,
    cross-validation, and automated overfitting/underfitting diagnostics.
    """
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    if not (
        np.issubdtype(y_train.dtype, np.number)
        and np.issubdtype(y_test.dtype, np.number)
    ):
        raise ValueError("Regression requires a numerical target column.")

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain at least two distinct target values.")

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

        # Train metrics (for overfitting detection)
        train_predictions = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, train_predictions)

        # Test metrics
        test_predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, test_predictions)
        mse = mean_squared_error(y_test, test_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, test_predictions)

        # Cross-validation on training data
        cv_folds = min(5, max(2, len(y_train) // 10))
        cv_mean = None
        cv_std = None
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
            cv_mean = round(float(np.mean(cv_scores)), 4)
            cv_std = round(float(np.std(cv_scores)), 4)
        except Exception:
            pass

        # Fit diagnosis (overfitting / underfitting)
        fit_diagnosis = diagnose_regression_fit(train_r2, r2, train_rmse, rmse)

        # Sample points for scatter plot visualization
        max_points = 200
        n_points = len(y_test)
        indices = np.random.choice(n_points, min(max_points, n_points), replace=False)
        y_test_array = np.array(y_test)

        plot_data = []
        for idx in indices:
            actual = float(y_test_array[idx])
            pred = float(test_predictions[idx])
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
            "train_r2": round(float(train_r2), 4),
            "train_rmse": round(float(train_rmse), 4),
            "train_mae": round(float(train_mae), 4),
            "test_r2": round(float(r2), 4),
            "test_rmse": round(float(rmse), 4),
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
            "fit_diagnosis": fit_diagnosis,
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

    best_diag = results[best_model_name]["fit_diagnosis"]
    recommendation = {
        "model": best_model_name,
        "reason": f"Selected based on best test {primary_metric} score of {best_score} ({best_diag['badge']})"
    }

    return {
        "results": results,
        "trained_models": trained_models,
        "recommendation": recommendation
    }


def train_classification_models(prepared: dict, primary_metric: str = "accuracy") -> dict:
    """
    Train and compare several classification models with both training and testing evaluations,
    cross-validation, and automated overfitting/underfitting diagnostics.
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
    avg_method = 'weighted'

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Train metrics (for overfitting detection)
        train_predictions = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_predictions)
        train_f1 = f1_score(y_train, train_predictions, average=avg_method, zero_division=0)

        # Test metrics
        test_predictions = model.predict(X_test)
        acc = accuracy_score(y_test, test_predictions)
        prec = precision_score(y_test, test_predictions, average=avg_method, zero_division=0)
        rec = recall_score(y_test, test_predictions, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, test_predictions, average=avg_method, zero_division=0)
        cm = confusion_matrix(y_test, test_predictions)

        # Cross-validation
        min_class_samples = int(y_train.value_counts().min())
        cv_folds = max(2, min(5, min_class_samples))
        cv_mean = None
        cv_std = None
        if min_class_samples >= 2:
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                cv_mean = round(float(np.mean(cv_scores)), 4)
                cv_std = round(float(np.std(cv_scores)), 4)
            except Exception:
                pass

        # Fit diagnosis (overfitting / underfitting)
        fit_diagnosis = diagnose_classification_fit(train_acc, acc, train_f1, f1)

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
            "train_accuracy": round(float(train_acc), 4),
            "train_f1": round(float(train_f1), 4),
            "test_accuracy": round(float(acc), 4),
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "fit_diagnosis": fit_diagnosis,
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

    best_diag = results[best_model_name]["fit_diagnosis"]
    recommendation = {
        "model": best_model_name,
        "reason": f"Selected based on best test {primary_metric} score of {best_score} ({best_diag['badge']})"
    }

    return {
        "results": results,
        "trained_models": trained_models,
        "recommendation": recommendation
    }
