from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import clone
import polars.selectors as cs
import polars as pl

import numpy as np

kfold = KFold(5, shuffle=True, random_state=42)


def cv_quick_score_regression(
    estimator, X: np.ndarray, y: np.ndarray, cv: KFold = kfold
) -> tuple[np.floating, np.floating, np.floating]:
    """
    Quickly evaluate a regression model using cross-validation.

    Computes R², MSE, and MAE using k-fold cross-validation and prints the results.

    Returns
    -------
    tuple[np.floating, np.floating, np.floating]
        Tuple of (r2, mse, mae) scores. Note that MSE and MAE are returned as
        negative values from cross_val_score but printed as positive values.
    """
    print("\n")
    print("====== Start =======")
    print(f"Model: {estimator}")
    cv_mse = np.mean(
        cross_val_score(estimator, X, y, scoring="neg_mean_squared_error", cv=cv)
    )
    cv_mae = np.mean(
        cross_val_score(estimator, X, y, scoring="neg_mean_absolute_error", cv=cv)
    )
    cv_r2 = np.mean(cross_val_score(estimator, X, y, scoring="r2", cv=cv))
    print(f"R2: {cv_r2}")
    print(f"MSE: {-cv_mse}")
    print(f"MAE: {-cv_mae}")
    print("====== Finish ======")
    print("\n")
    return (cv_r2, cv_mse, cv_mae)


def cv_quick_score_classification(
    estimator, X: np.ndarray, y: np.ndarray, cv: KFold = kfold
) -> tuple[np.floating, np.floating, np.floating, np.floating]:
    """
    Quickly evaluate a classification model using cross-validation.

    Computes accuracy, precision, recall, and F1 score using k-fold cross-validation
    and prints the results.

    Returns
    -------
    tuple[np.floating, np.floating, np.floating, np.floating]
        Tuple of (accuracy, precision, recall, f1) scores.
    """
    print("\n")
    print("====== Start =======")
    print(f"Model: {estimator}")
    cv_accuracy = np.mean(cross_val_score(estimator, X, y, scoring="accuracy", cv=cv))
    cv_precision = np.mean(
        cross_val_score(estimator, X, y, scoring="precision_weighted", cv=cv)
    )
    cv_recall = np.mean(
        cross_val_score(estimator, X, y, scoring="recall_weighted", cv=cv)
    )
    cv_f1 = np.mean(cross_val_score(estimator, X, y, scoring="f1_weighted", cv=cv))
    print(f"Accuracy: {cv_accuracy:.4f}")
    print(f"Precision: {cv_precision:.4f}")
    print(f"Recall: {cv_recall:.4f}")
    print(f"F1: {cv_f1:.4f}")
    print("====== Finish ======")
    print("\n")
    return (cv_accuracy, cv_precision, cv_recall, cv_f1)


def cv_score_improvement_by_new_features(
    new_X: np.ndarray,
    baseline_X: np.ndarray,
    y: np.ndarray,
    sklearn_estimator,
    sklearn_scoring,
    cv: KFold = kfold,
) -> tuple[np.floating, np.floating]:
    """
    Compare cross-validation scores between old and new feature sets.

    Evaluates whether adding new features improves model performance by computing
    cross-validation scores on both the baseline (old) and enhanced (new) feature sets,
    then reporting the difference.

    Returns
    -------
    tuple[float, float]
        Tuple of (baseline_score, new_score) for further analysis. Also prints comparison
        results including baseline score, new score, absolute difference, and percentage change.
    """
    baseline_score = np.mean(
        cross_val_score(
            clone(sklearn_estimator), baseline_X, y, scoring=sklearn_scoring, cv=cv
        )
    )
    new_score = np.mean(
        cross_val_score(
            clone(sklearn_estimator), new_X, y, scoring=sklearn_scoring, cv=cv
        )
    )
    difference = new_score - baseline_score
    print(f"Baseline score: {baseline_score}")
    print(f"Score with new features: {new_score}")
    if difference > 0:
        print(
            f"The model improves by {abs(difference)} ({abs(difference) / abs(baseline_score) * 100:.2f}%)"
        )
    else:
        print(
            f"The model degrades by {abs(difference)}({abs(difference) / abs(baseline_score) * 100:.2f}%)"
        )
    return (baseline_score, new_score)


def make_new_X(
    new_df: pl.DataFrame, y_col: str, id_col: str = "id", test_set: bool = False
) -> np.ndarray:
    """
    Prepare feature matrix from a Polars DataFrame with one-hot encoding.

    Converts categorical (string and boolean) columns to dummy variables and
    returns a numpy array suitable for machine learning models. Automatically
    handles train/test set differences in column dropping.

    Notes
    -----
    - Uses Polars' `to_dummies()` for one-hot encoding of string and boolean columns.
    - Numeric columns are passed through unchanged.
    - The function assumes the DataFrame is a Polars DataFrame.
    """
    cat = new_df.select([cs.string(), cs.boolean()]).columns
    if not test_set:
        new_X = new_df.drop([id_col, y_col]).to_dummies(cat).to_numpy()
    else:
        new_X = new_df.drop([id_col]).to_dummies(cat).to_numpy()
    return new_X
