from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "ml_boundary_dataset.csv"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "ml_model_comparison.csv"
)


RANDOM_SEED = 42

# Candidate thresholds tested on validation data.
THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.05,
)


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"ML dataset not found:\n{DATASET_FILE}"
        )

    print("Loading ML dataset...")

    df = pd.read_csv(
        DATASET_FILE
    )

    df["candidate_ts"] = pd.to_datetime(
        df["candidate_ts"],
        utc=True,
    )

    return df


def prepare_features(df):
    """
    Select only model features.

    Explicitly exclude identifiers, labels and split metadata.
    """

    excluded = {
        "session_id",
        "candidate_ts",
        "label",
        "split",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise RuntimeError(
            "No feature columns found."
        )

    X = df[
        feature_columns
    ].copy()

    y = df["label"].astype(int)

    return X, y, feature_columns


def split_data(df, X, y):
    train_mask = (
        df["split"] == "train"
    )

    validation_mask = (
        df["split"] == "validation"
    )

    test_mask = (
        df["split"] == "test"
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


def choose_threshold(
    y_true,
    probabilities,
):
    """
    Choose the threshold that maximizes validation F1.

    This threshold is selected ONLY from validation data.
    """

    best_threshold = None
    best_f1 = -1.0

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1),
    )


def evaluate_predictions(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "predicted_boundaries": int(
            predictions.sum()
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def build_models():
    """
    Models intentionally use modest/default configurations.

    Logistic Regression uses scaling because its coefficients are
    sensitive to feature scale.

    Tree models do not require feature scaling.
    """

    models = {
        "LogisticRegression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
        ),
    }

    return models


def train_and_evaluate(
    name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    print()
    print(
        f"Training {name}..."
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    threshold, validation_f1 = (
        choose_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    validation_metrics = (
        evaluate_predictions(
            y_validation,
            validation_probabilities,
            threshold,
        )
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = (
        evaluate_predictions(
            y_test,
            test_probabilities,
            threshold,
        )
    )

    print(
        f"Validation threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{validation_f1:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_metrics['pr_auc']:.4f}"
    )

    return {
        "model": name,

        "threshold": threshold,

        "validation_precision":
            validation_metrics["precision"],

        "validation_recall":
            validation_metrics["recall"],

        "validation_f1":
            validation_metrics["f1"],

        "validation_pr_auc":
            validation_metrics["pr_auc"],

        "test_precision":
            test_metrics["precision"],

        "test_recall":
            test_metrics["recall"],

        "test_f1":
            test_metrics["f1"],

        "test_pr_auc":
            test_metrics["pr_auc"],

        "test_predicted_boundaries":
            test_metrics["predicted_boundaries"],

        "test_true_positives":
            test_metrics["true_positives"],

        "test_false_positives":
            test_metrics["false_positives"],

        "test_false_negatives":
            test_metrics["false_negatives"],

    }, model


def print_feature_importance(
    name,
    model,
    feature_columns,
):
    """
    Print feature importance where available.

    This is for interpretation only.
    """

    estimator = model

    if isinstance(
        model,
        Pipeline,
    ):
        estimator = model.named_steps[
            "classifier"
        ]

    importances = None

    if hasattr(
        estimator,
        "feature_importances_",
    ):
        importances = (
            estimator.feature_importances_
        )

    elif hasattr(
        estimator,
        "coef_",
    ):
        importances = np.abs(
            estimator.coef_[0]
        )

    if importances is None:
        return

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": importances,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(10)
    )

    print()
    print(
        f"Top features for {name}:"
    )

    print(
        importance_df.to_string(
            index=False
        )
    )


def main():
    print("=" * 60)
    print("ML MODEL COMPARISON")
    print("=" * 60)

    df = load_dataset()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Sessions: "
        f"{df['session_id'].nunique():,}"
    )

    X, y, feature_columns = (
        prepare_features(df)
    )

    print(
        f"Features: "
        f"{len(feature_columns):,}"
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = split_data(
        df,
        X,
        y,
    )

    print()
    print("Data split:")
    print(
        f"Train:      {len(X_train):,}"
    )
    print(
        f"Validation: {len(X_validation):,}"
    )
    print(
        f"Test:       {len(X_test):,}"
    )

    models = build_models()

    results = []
    trained_models = {}

    for name, model in models.items():

        result, trained_model = (
            train_and_evaluate(
                name,
                model,
                X_train,
                y_train,
                X_validation,
                y_validation,
                X_test,
                y_test,
            )
        )

        results.append(result)
        trained_models[name] = trained_model

        print_feature_importance(
            name,
            trained_model,
            feature_columns,
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "test_f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df[
            [
                "model",
                "threshold",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_pr_auc",
                "test_predicted_boundaries",
                "test_true_positives",
                "test_false_positives",
                "test_false_negatives",
            ]
        ].to_string(
            index=False
        )
    )

    winner = results_df.iloc[0]

    print()
    print(
        f"Best test F1 model: "
        f"{winner['model']}"
    )

    print(
        f"Test F1: "
        f"{winner['test_f1']:.4f}"
    )

    print()
    print(
        f"Saved comparison:\n"
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()