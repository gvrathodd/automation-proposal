from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "ml_boundary_dataset.csv"
)

GT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "gt_boundaries.csv"
)

BOUNDARY_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ml_boundaries_postprocessed.csv"
)

EVAL_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ml_boundary_temporal_evaluation.csv"
)

TEST_PROB_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ml_test_probabilities.csv"
)

SESSION_STATS_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ml_test_session_statistics.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

# Selected from the validation set during model comparison.
CLASSIFICATION_THRESHOLD = 0.40

# Candidate grid spacing from V2.
CANDIDATE_SPACING_SECONDS = 2

# Positive candidate points within this distance are treated
# as belonging to the same boundary episode.
MERGE_GAP_SECONDS = 4

# Evaluate at multiple temporal tolerances.
TOLERANCES = [1, 2, 3, 5]


# ============================================================
# Load ML dataset
# ============================================================

def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"ML dataset not found:\n{DATASET_FILE}"
        )

    print("Loading ML dataset...")

    df = pd.read_csv(
        DATASET_FILE
    )

    required_columns = {
        "session_id",
        "candidate_ts",
        "label",
        "split",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "ML dataset is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    df["candidate_ts"] = pd.to_datetime(
        df["candidate_ts"],
        utc=True,
        errors="coerce",
    )

    if df["candidate_ts"].isna().any():
        raise RuntimeError(
            "Invalid candidate_ts values found."
        )

    return df


# ============================================================
# Feature selection
# ============================================================

def get_feature_columns(df):
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
            "No ML feature columns found."
        )

    return feature_columns


# ============================================================
# Train HistGradientBoosting
# ============================================================

def train_model(X_train, y_train):
    """
    Same configuration used during the original
    model-comparison experiment, with runtime diagnostics.
    """

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=RANDOM_SEED,
    )

    print()
    print("-" * 70)
    print("MODEL TRAINING DIAGNOSTICS")
    print("-" * 70)

    print(
        f"Training matrix shape: "
        f"{X_train.shape}"
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Training features: "
        f"{X_train.shape[1]:,}"
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    print(
        f"Positive labels: "
        f"{positive_count:,}"
    )

    print(
        f"Negative labels: "
        f"{negative_count:,}"
    )

    positive_rate = (
        positive_count / len(y_train)
        if len(y_train) > 0
        else 0.0
    )

    print(
        f"Positive rate: "
        f"{positive_rate:.6f}"
    )

    print()
    print("Feature dtypes:")

    print(
        X_train.dtypes.to_string()
    )

    print()
    print("Feature summary:")

    print(
        X_train.describe(
            include="all"
        ).T[
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        ].to_string()
    )

    print()
    print("Model configuration:")
    print(
        f"  max_iter = "
        f"{model.max_iter}"
    )
    print(
        f"  learning_rate = "
        f"{model.learning_rate}"
    )
    print(
        f"  max_leaf_nodes = "
        f"{model.max_leaf_nodes}"
    )

    print()
    print("Starting model.fit()...")

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Training time: "
        f"{elapsed_seconds:.3f} seconds"
    )

    # HistGradientBoosting may stop before max_iter
    # depending on early stopping.
    print(
        f"Actual boosting iterations: "
        f"{model.n_iter_}"
    )

    if hasattr(model, "n_iter_no_change"):
        print(
            f"n_iter_no_change: "
            f"{model.n_iter_no_change}"
        )

    print(
        f"Requested max_iter: "
        f"{model.max_iter}"
    )

    if model.n_iter_ < model.max_iter:
        print(
            "NOTE: training stopped before max_iter."
        )

    print("-" * 70)

    return model


# ============================================================
# Select positive candidates
# ============================================================

def select_positive_candidates(test, probabilities):
    result = test[
        [
            "session_id",
            "candidate_ts",
        ]
    ].copy()

    result["probability"] = probabilities

    result["predicted_positive"] = (
        result["probability"]
        >= CLASSIFICATION_THRESHOLD
    )

    return result


# ============================================================
# Merge nearby candidate detections
# ============================================================

def merge_candidates(
    positive_candidates,
):
    """
    Convert multiple nearby positive candidate points
    into one boundary episode.

    The highest-probability candidate represents the
    boundary timestamp for the episode.
    """

    predictions = []

    if positive_candidates.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "predicted_ts",
                "probability",
                "cluster_size",
            ]
        )

    for session_id, group in (
        positive_candidates
        .groupby("session_id")
    ):

        group = group.sort_values(
            "candidate_ts"
        )

        cluster = []
        previous_time = None

        for _, row in group.iterrows():

            current_time = row[
                "candidate_ts"
            ]

            if (
                previous_time is None
                or (
                    current_time
                    - previous_time
                ).total_seconds()
                <= MERGE_GAP_SECONDS
            ):
                cluster.append(row)

            else:
                cluster_df = pd.DataFrame(
                    cluster
                )

                best = cluster_df.loc[
                    cluster_df[
                        "probability"
                    ].idxmax()
                ]

                predictions.append(
                    {
                        "session_id": session_id,
                        "predicted_ts": best[
                            "candidate_ts"
                        ],
                        "probability": best[
                            "probability"
                        ],
                        "cluster_size": len(
                            cluster_df
                        ),
                    }
                )

                cluster = [row]

            previous_time = current_time

        # Flush final cluster.
        if cluster:

            cluster_df = pd.DataFrame(
                cluster
            )

            best = cluster_df.loc[
                cluster_df[
                    "probability"
                ].idxmax()
            ]

            predictions.append(
                {
                    "session_id": session_id,
                    "predicted_ts": best[
                        "candidate_ts"
                    ],
                    "probability": best[
                        "probability"
                    ],
                    "cluster_size": len(
                        cluster_df
                    ),
                }
            )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# Load GT
# ============================================================

def load_ground_truth():
    if not GT_FILE.exists():
        raise FileNotFoundError(
            f"GT file not found:\n{GT_FILE}"
        )

    gt = pd.read_csv(
        GT_FILE
    )

    required_columns = {
        "session_id",
        "ts",
    }

    missing = (
        required_columns
        - set(gt.columns)
    )

    if missing:
        raise RuntimeError(
            "GT file is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    gt["ts"] = pd.to_datetime(
        gt["ts"],
        utc=True,
        errors="coerce",
    )

    if gt["ts"].isna().any():
        raise RuntimeError(
            "Invalid GT timestamps found."
        )

    # Same normalization used throughout the project.
    gt = (
        gt[
            [
                "session_id",
                "ts",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return gt


# ============================================================
# Temporal matching
# ============================================================

def evaluate_at_tolerance(
    predictions,
    gt,
    tolerance_seconds,
):
    """
    One-to-one temporal matching.

    Each prediction can match at most one GT boundary.
    Each GT boundary can match at most one prediction.

    Matching is done by nearest timestamp within tolerance.
    """

    if predictions.empty:
        return {
            "tolerance_seconds":
                tolerance_seconds,
            "gt_boundaries":
                len(gt),
            "predicted_boundaries":
                0,
            "matched_boundaries":
                0,
            "precision":
                0.0,
            "recall":
                0.0,
            "f1":
                0.0,
            "mean_boundary_error_seconds":
                None,
            "median_boundary_error_seconds":
                None,
        }

    matched_gt = set()
    errors = []

    # Process predictions in chronological order.
    predictions = (
        predictions
        .sort_values(
            [
                "session_id",
                "predicted_ts",
            ]
        )
        .reset_index(drop=True)
    )

    matched_predictions = 0

    for _, prediction in predictions.iterrows():

        session_id = prediction[
            "session_id"
        ]

        predicted_ts = prediction[
            "predicted_ts"
        ]

        session_gt = gt[
            gt["session_id"]
            == session_id
        ]

        best_index = None
        best_error = None

        for gt_index, gt_row in (
            session_gt.iterrows()
        ):

            key = (
                session_id,
                gt_index,
            )

            if key in matched_gt:
                continue

            error = abs(
                (
                    predicted_ts
                    - gt_row["ts"]
                ).total_seconds()
            )

            if error <= tolerance_seconds:

                if (
                    best_error is None
                    or error < best_error
                ):
                    best_error = error
                    best_index = gt_index

        if best_index is not None:

            matched_gt.add(
                (
                    session_id,
                    best_index,
                )
            )

            matched_predictions += 1

            errors.append(
                best_error
            )

    total_predictions = len(
        predictions
    )

    total_gt = len(
        gt
    )

    precision = (
        matched_predictions
        / total_predictions
        if total_predictions
        else 0.0
    )

    recall = (
        matched_predictions
        / total_gt
        if total_gt
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tolerance_seconds":
            tolerance_seconds,
        "gt_boundaries":
            total_gt,
        "predicted_boundaries":
            total_predictions,
        "matched_boundaries":
            matched_predictions,
        "precision":
            precision,
        "recall":
            recall,
        "f1":
            f1,
        "mean_boundary_error_seconds":
            (
                float(np.mean(errors))
                if errors
                else None
            ),
        "median_boundary_error_seconds":
            (
                float(np.median(errors))
                if errors
                else None
            ),
    }


# ============================================================
# Session-level sanity statistics
# ============================================================

def build_session_statistics(
    predictions,
    test,
):
    """
    Statistics only for the held-out test sessions.
    """

    test_sessions = (
        test[
            "session_id"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # Start with every test session so sessions with zero
    # predicted boundaries remain visible.
    stats = pd.DataFrame(
        {
            "session_id":
                test_sessions
        }
    )

    event_counts = (
        test.groupby(
            "session_id"
        )
        .size()
        .rename(
            "candidate_count"
        )
    )

    stats = stats.merge(
        event_counts,
        on="session_id",
        how="left",
    )

    predicted_counts = (
        predictions.groupby(
            "session_id"
        )
        .size()
        .rename(
            "predicted_boundary_count"
        )
    )

    stats = stats.merge(
        predicted_counts,
        on="session_id",
        how="left",
    )

    stats[
        "predicted_boundary_count"
    ] = (
        stats[
            "predicted_boundary_count"
        ]
        .fillna(0)
        .astype(int)
    )

    # A session with N boundaries has approximately
    # N+1 segments.
    stats[
        "predicted_segment_count"
    ] = (
        stats[
            "predicted_boundary_count"
        ]
        + 1
    )

    return stats


# ============================================================
# Probability diagnostics
# ============================================================

def print_probability_diagnostics(
    probabilities,
    predicted_positive,
):
    print()
    print("-" * 70)
    print("TEST PROBABILITY DIAGNOSTICS")
    print("-" * 70)

    print(
        f"Probability count: "
        f"{len(probabilities):,}"
    )

    print(
        f"Probability min: "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Probability max: "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Probability mean: "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"Probability median: "
        f"{np.median(probabilities):.6f}"
    )

    print(
        f"Threshold: "
        f"{CLASSIFICATION_THRESHOLD:.2f}"
    )

    positive_count = int(
        predicted_positive.sum()
    )

    print(
        f"Predicted positive candidates: "
        f"{positive_count:,}"
    )

    print(
        f"Predicted positive rate: "
        f"{positive_count / len(probabilities):.6f}"
    )

    print()
    print("Probability quantiles:")

    quantiles = np.quantile(
        probabilities,
        [
            0.00,
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        ],
    )

    quantile_names = [
        "0%",
        "1%",
        "5%",
        "10%",
        "25%",
        "50%",
        "75%",
        "90%",
        "95%",
        "99%",
        "100%",
    ]

    for name, value in zip(
        quantile_names,
        quantiles,
    ):
        print(
            f"  {name:>4}: "
            f"{value:.6f}"
        )

    print("-" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ML BOUNDARY POST-PROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Sessions: "
        f"{df['session_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # 2. Select features
    # --------------------------------------------------------

    feature_columns = (
        get_feature_columns(
            df
        )
    )

    print(
        f"Features used: "
        f"{len(feature_columns):,}"
    )

    print()
    print("Feature columns:")
    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # 3. Split
    # --------------------------------------------------------

    train = df[
        df["split"] == "train"
    ].copy()

    validation = df[
        df["split"] == "validation"
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    train_sessions = set(
        train["session_id"]
    )

    validation_sessions = set(
        validation["session_id"]
    )

    test_sessions = set(
        test["session_id"]
    )

    # Explicitly verify there is no session overlap.
    if train_sessions & test_sessions:
        raise RuntimeError(
            "Session leakage between train and test."
        )

    if validation_sessions & test_sessions:
        raise RuntimeError(
            "Session leakage between validation and test."
        )

    if train_sessions & validation_sessions:
        raise RuntimeError(
            "Session leakage between train and validation."
        )

    print()
    print(
        f"Training sessions: "
        f"{len(train_sessions):,}"
    )

    print(
        f"Validation sessions: "
        f"{len(validation_sessions):,}"
    )

    print(
        f"Test sessions: "
        f"{len(test_sessions):,}"
    )

    print()
    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    # --------------------------------------------------------
    # 3b. Dataset integrity diagnostics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("DATASET INTEGRITY DIAGNOSTICS")
    print("-" * 70)

    print(
        f"Total rows: "
        f"{len(df):,}"
    )

    print(
        f"Duplicate candidate rows: "
        f"{df.duplicated().sum():,}"
    )

    duplicate_candidates = (
        df.duplicated(
            subset=[
                "session_id",
                "candidate_ts",
            ]
        ).sum()
    )

    print(
        f"Duplicate "
        f"(session_id, candidate_ts) "
        f"pairs: "
        f"{duplicate_candidates:,}"
    )

    print()
    print("Rows by split:")

    print(
        df["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Labels by split:")

    label_split_table = (
        pd.crosstab(
            df["split"],
            df["label"],
        )
        .sort_index()
    )

    print(
        label_split_table.to_string()
    )

    # Check numerical features for NaN/infinity.
    X_all = df[
        feature_columns
    ]

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            X_all[column]
        )
    ]

    if non_numeric:
        raise RuntimeError(
            "Non-numeric ML feature columns found:\n"
            + "\n".join(non_numeric)
        )

    nan_counts = (
        X_all
        .isna()
        .sum()
    )

    nan_features = nan_counts[
        nan_counts > 0
    ]

    if not nan_features.empty:
        print()
        print(
            "WARNING: NaNs detected:"
        )
        print(
            nan_features.to_string()
        )
    else:
        print(
            "NaN feature values: 0"
        )

    finite_mask = np.isfinite(
        X_all.to_numpy(
            dtype=float
        )
    )

    non_finite_count = (
        (~finite_mask).sum()
    )

    print(
        f"Non-finite feature values: "
        f"{non_finite_count:,}"
    )

    if non_finite_count > 0:
        raise RuntimeError(
            "Non-finite feature values found."
        )

    print("-" * 70)

    # --------------------------------------------------------
    # 4. Train selected model
    # --------------------------------------------------------

    print()
    print(
        "Training HistGradientBoosting..."
    )

    model = train_model(
        train[feature_columns],
        train["label"].astype(int),
    )

    # --------------------------------------------------------
    # 5. Generate test probabilities
    # --------------------------------------------------------

    print()
    print(
        "Generating test probabilities..."
    )

    prediction_start = time.perf_counter()

    probabilities = model.predict_proba(
        test[feature_columns]
    )[:, 1]

    prediction_elapsed = (
        time.perf_counter()
        - prediction_start
    )

    print(
        f"Prediction time: "
        f"{prediction_elapsed:.3f} seconds"
    )

    test_results = (
        select_positive_candidates(
            test,
            probabilities,
        )
    )

    print_probability_diagnostics(
        probabilities,
        test_results[
            "predicted_positive"
        ].to_numpy(),
    )

    test_results.to_csv(
        TEST_PROB_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 6. Select positive candidates
    # --------------------------------------------------------

    positive_candidates = (
        test_results[
            test_results[
                "predicted_positive"
            ]
        ]
        .copy()
    )

    print()
    print(
        "Positive candidates before merging: "
        f"{len(positive_candidates):,}"
    )

    # --------------------------------------------------------
    # 7. Convert candidate points into actual boundaries
    # --------------------------------------------------------

    predictions = merge_candidates(
        positive_candidates
    )

    print(
        "Post-processed boundaries: "
        f"{len(predictions):,}"
    )

    predictions.to_csv(
        BOUNDARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 8. Load GT and restrict it to TEST SESSIONS
    # --------------------------------------------------------

    gt_all = load_ground_truth()

    gt = gt_all[
        gt_all["session_id"].isin(
            test_sessions
        )
    ].copy()

    gt = (
        gt.sort_values(
            [
                "session_id",
                "ts",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "GT boundaries across all Dataset A sessions: "
        f"{len(gt_all):,}"
    )

    print(
        "GT boundaries in TEST sessions only: "
        f"{len(gt):,}"
    )

    # Safety check: every prediction must belong to a test session.
    prediction_sessions = set(
        predictions["session_id"]
    )

    unexpected_sessions = (
        prediction_sessions
        - test_sessions
    )

    if unexpected_sessions:
        raise RuntimeError(
            "Predictions were generated for sessions "
            "outside the test split:\n"
            + "\n".join(
                sorted(
                    unexpected_sessions
                )
            )
        )

    # --------------------------------------------------------
    # 9. Evaluate temporal boundaries
    # --------------------------------------------------------

    evaluation_rows = []

    for tolerance in TOLERANCES:

        metrics = evaluate_at_tolerance(
            predictions,
            gt,
            tolerance,
        )

        evaluation_rows.append(
            metrics
        )

    evaluation_df = (
        pd.DataFrame(
            evaluation_rows
        )
    )

    evaluation_df.to_csv(
        EVAL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 10. Session sanity statistics
    # --------------------------------------------------------

    session_stats = (
        build_session_statistics(
            predictions,
            test,
        )
    )

    session_stats.to_csv(
        SESSION_STATS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 11. Print session statistics
    # --------------------------------------------------------

    print()
    print(
        "Predicted boundaries per TEST session:"
    )

    print(
        session_stats[
            "predicted_boundary_count"
        ]
        .describe()
        .to_string()
    )

    print()
    print(
        "Predicted segments per TEST session:"
    )

    print(
        session_stats[
            "predicted_segment_count"
        ]
        .describe()
        .to_string()
    )

    print()
    print(
        "Full test-session boundary counts:"
    )

    print(
        session_stats[
            [
                "session_id",
                "predicted_boundary_count",
                "predicted_segment_count",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 12. Print temporal evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TEMPORAL BOUNDARY EVALUATION — TEST SESSIONS ONLY"
    )
    print("=" * 70)

    print(
        evaluation_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 13. Print files
    # --------------------------------------------------------

    print()
    print(
        f"Saved probabilities:\n"
        f"{TEST_PROB_OUTPUT}"
    )

    print(
        f"Saved boundaries:\n"
        f"{BOUNDARY_OUTPUT}"
    )

    print(
        f"Saved evaluation:\n"
        f"{EVAL_OUTPUT}"
    )

    print(
        f"Saved session statistics:\n"
        f"{SESSION_STATS_OUTPUT}"
    )


if __name__ == "__main__":
    main()