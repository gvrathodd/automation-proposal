from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from loader import (
    load_all_events,
    sort_session_events,
    get_timestamp_ms,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "raw_data"
    / "dataset-downloads"
)

GT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "gt_boundaries.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

VALIDATION_RESULTS_FILE = (
    OUTPUT_DIR
    / "segment_model_validation_results.csv"
)

TEST_RESULTS_FILE = (
    OUTPUT_DIR
    / "segment_model_test_results.csv"
)

TEST_BOUNDARIES_FILE = (
    OUTPUT_DIR
    / "segment_model_test_boundaries.csv"
)

VALIDATION_PROBABILITIES_FILE = (
    OUTPUT_DIR
    / "segment_model_validation_probabilities.csv"
)

SPLIT_FILE = (
    OUTPUT_DIR
    / "segment_model_split.csv"
)

DATASET_OUTPUT_FILE = (
    OUTPUT_DIR
    / "raw_segment_ml_dataset.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

# Candidate spacing.
CANDIDATE_SPACING_SECONDS = 2

# Activity window on either side of candidate.
WINDOW_SECONDS = 6

# Temporal consolidation after classification.
MERGE_GAP_SECONDS = 2

# Model capacities to test.
MAX_ITER_VALUES = [
    200,
    300,
    500,
    800,
]

# Classification thresholds.
THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.05,
)

# Temporal evaluation tolerances.
TOLERANCES = [
    1,
    2,
    3,
    5,
]


# ============================================================
# Feature definitions
# ============================================================

FEATURE_COLUMNS = [
    "before_event_count",
    "after_event_count",
    "before_app_switches",
    "after_app_switches",
    "before_navigation",
    "after_navigation",
    "before_browser_clicks",
    "after_browser_clicks",
    "before_mouse_clicks",
    "after_mouse_clicks",
    "before_window_changes",
    "after_window_changes",
    "before_unique_apps",
    "after_unique_apps",
    "before_keyboard",
    "after_keyboard",
]


# ============================================================
# Event helpers
# ============================================================

def get_event_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}

    app_name = active_app.get("app_name")

    if isinstance(app_name, str):
        return app_name

    return None


def get_event_type(event):
    event_type = event.get("event_type")

    if isinstance(event_type, str):
        return event_type

    return ""


def is_keyboard_event(event_type):
    return (
        event_type == "keystroke"
        or event_type == "text_input_complete"
    )


def is_app_switch_event(event_type):
    return event_type == "app_switch"


def is_navigation_event(event_type):
    return event_type == "browser_navigation"


def is_browser_click_event(event_type):
    return event_type == "browser_click"


def is_mouse_click_event(event_type):
    return event_type == "mouse_click"


def is_window_change_event(event_type):
    return (
        event_type == "window_title_change"
        or event_type == "window_state_change"
    )


# ============================================================
# Prepare raw session events
# ============================================================

def prepare_session_events(events):
    """
    Convert raw events into compact records.

    Physical chunk layout is ignored. The existing loader
    reconstructs the session and sorts events chronologically.
    """

    ordered_events = sort_session_events(events)

    prepared = []

    for event in ordered_events:

        timestamp_ms = get_timestamp_ms(event)

        if timestamp_ms is None:
            continue

        prepared.append(
            {
                "timestamp_ms": timestamp_ms,
                "event_type": get_event_type(event),
                "app": get_event_app(event),
            }
        )

    return prepared


# ============================================================
# Window feature calculation
# ============================================================

def calculate_window_features(
    events,
    start_ms,
    end_ms,
):
    """
    Calculate raw activity features in [start_ms, end_ms).
    """

    selected = [
        event
        for event in events
        if (
            start_ms
            <= event["timestamp_ms"]
            < end_ms
        )
    ]

    return {
        "event_count":
            len(selected),

        "app_switches":
            sum(
                is_app_switch_event(
                    event["event_type"]
                )
                for event in selected
            ),

        "navigation":
            sum(
                is_navigation_event(
                    event["event_type"]
                )
                for event in selected
            ),

        "browser_clicks":
            sum(
                is_browser_click_event(
                    event["event_type"]
                )
                for event in selected
            ),

        "mouse_clicks":
            sum(
                is_mouse_click_event(
                    event["event_type"]
                )
                for event in selected
            ),

        "window_changes":
            sum(
                is_window_change_event(
                    event["event_type"]
                )
                for event in selected
            ),

        "unique_apps":
            len(
                {
                    event["app"]
                    for event in selected
                    if event["app"]
                }
            ),

        "keyboard":
            sum(
                is_keyboard_event(
                    event["event_type"]
                )
                for event in selected
            ),
    }


# ============================================================
# Candidate generation for one session
# ============================================================

def build_session_candidates(
    session_id,
    events,
):
    """
    Build candidates at 2-second intervals.

    Features are calculated from raw event activity.
    Ground truth is NOT used here.
    """

    if not events:
        return pd.DataFrame()

    first_ms = events[0]["timestamp_ms"]
    last_ms = events[-1]["timestamp_ms"]

    spacing_ms = (
        CANDIDATE_SPACING_SECONDS
        * 1000
    )

    window_ms = (
        WINDOW_SECONDS
        * 1000
    )

    rows = []

    candidate_ms = first_ms

    while candidate_ms <= last_ms:

        before = calculate_window_features(
            events,
            candidate_ms - window_ms,
            candidate_ms,
        )

        after = calculate_window_features(
            events,
            candidate_ms,
            candidate_ms + window_ms,
        )

        rows.append(
            {
                "session_id":
                    session_id,

                "candidate_ts":
                    pd.to_datetime(
                        candidate_ms,
                        unit="ms",
                        utc=True,
                    ),

                "before_event_count":
                    before["event_count"],

                "after_event_count":
                    after["event_count"],

                "before_app_switches":
                    before["app_switches"],

                "after_app_switches":
                    after["app_switches"],

                "before_navigation":
                    before["navigation"],

                "after_navigation":
                    after["navigation"],

                "before_browser_clicks":
                    before["browser_clicks"],

                "after_browser_clicks":
                    after["browser_clicks"],

                "before_mouse_clicks":
                    before["mouse_clicks"],

                "after_mouse_clicks":
                    after["mouse_clicks"],

                "before_window_changes":
                    before["window_changes"],

                "after_window_changes":
                    after["window_changes"],

                "before_unique_apps":
                    before["unique_apps"],

                "after_unique_apps":
                    after["unique_apps"],

                "before_keyboard":
                    before["keyboard"],

                "after_keyboard":
                    after["keyboard"],
            }
        )

        candidate_ms += spacing_ms

    return pd.DataFrame(rows)


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

    required = {
        "session_id",
        "ts",
    }

    missing = required - set(gt.columns)

    if missing:
        raise RuntimeError(
            "GT file is missing columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    gt["ts"] = pd.to_datetime(
        gt["ts"],
        utc=True,
        errors="coerce",
    )

    if gt["ts"].isna().any():
        raise RuntimeError(
            "Invalid GT timestamps detected."
        )

    gt = (
        gt[
            [
                "session_id",
                "ts",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "session_id",
                "ts",
            ]
        )
        .reset_index(drop=True)
    )

    return gt


# ============================================================
# Build deterministic session split
# ============================================================

def build_session_split(
    session_ids
):
    """
    Same deterministic 70/15/15 session split used previously.
    """

    sessions = sorted(
        session_ids
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rng.shuffle(
        sessions
    )

    n_sessions = len(sessions)

    n_train = int(
        n_sessions * 0.70
    )

    n_validation = int(
        n_sessions * 0.15
    )

    train_sessions = sessions[
        :n_train
    ]

    validation_sessions = sessions[
        n_train:
        n_train + n_validation
    ]

    test_sessions = sessions[
        n_train + n_validation:
    ]

    rows = []

    for session_id in train_sessions:
        rows.append(
            {
                "session_id":
                    session_id,
                "split":
                    "train",
            }
        )

    for session_id in validation_sessions:
        rows.append(
            {
                "session_id":
                    session_id,
                "split":
                    "validation",
            }
        )

    for session_id in test_sessions:
        rows.append(
            {
                "session_id":
                    session_id,
                "split":
                    "test",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Create tight labels
# ============================================================

def create_tight_labels(
    candidates,
    gt,
):
    """
    Assign one positive candidate per GT boundary whenever
    the session has candidates.

    For each GT timestamp, the nearest candidate in that
    session is marked positive.

    All features are generated from raw events only.
    GT is used exclusively to construct labels.
    """

    candidates = candidates.copy()

    candidates["label"] = 0

    positive_indices = set()

    gt_by_session = {
        session_id:
            group["ts"]
            .sort_values()
            .tolist()
        for session_id, group
        in gt.groupby("session_id")
    }

    for session_id, gt_timestamps in (
        gt_by_session.items()
    ):

        session_candidates = candidates[
            candidates["session_id"]
            == session_id
        ]

        if session_candidates.empty:
            continue

        candidate_ts = session_candidates[
            "candidate_ts"
        ]

        for gt_ts in gt_timestamps:

            deltas = (
                candidate_ts
                - gt_ts
            ).abs()

            nearest_index = deltas.idxmin()

            positive_indices.add(
                nearest_index
            )

    if positive_indices:

        candidates.loc[
            list(positive_indices),
            "label",
        ] = 1

    return candidates


# ============================================================
# Build raw Dataset A ML dataset
# ============================================================

def build_raw_dataset():
    print("=" * 70)
    print(
        "BUILDING SEGMENTATION DATASET "
        "DIRECTLY FROM RAW JSON"
    )
    print("=" * 70)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATA_ROOT}"
        )

    print(
        f"Data root:\n{DATA_ROOT}"
    )

    print()
    print(
        "Loading reconstructed sessions..."
    )

    all_sessions = load_all_events(
        DATA_ROOT
    )

    # Determine Dataset A from event metadata.
    dataset_a_sessions = {}

    for session_id, events in all_sessions.items():

        datasets = {
            event.get("_dataset")
            for event in events
            if event.get("_dataset")
        }

        if "A" in datasets:
            dataset_a_sessions[
                session_id
            ] = events

    print(
        f"Dataset A sessions: "
        f"{len(dataset_a_sessions):,}"
    )

    if not dataset_a_sessions:
        raise RuntimeError(
            "No Dataset A sessions found."
        )

    gt = load_ground_truth()

    split_df = build_session_split(
        dataset_a_sessions.keys()
    )

    split_df.to_csv(
        SPLIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    split_map = dict(
        zip(
            split_df[
                "session_id"
            ],
            split_df[
                "split"
            ],
        )
    )

    all_candidate_frames = []

    start_time = time.perf_counter()

    session_items = list(
        dataset_a_sessions.items()
    )

    for position, (
        session_id,
        raw_events,
    ) in enumerate(
        session_items,
        start=1,
    ):

        prepared_events = (
            prepare_session_events(
                raw_events
            )
        )

        candidates = (
            build_session_candidates(
                session_id,
                prepared_events,
            )
        )

        if not candidates.empty:
            all_candidate_frames.append(
                candidates
            )

        if (
            position % 10 == 0
            or position == len(session_items)
        ):
            print(
                f"Processed "
                f"{position:,}/"
                f"{len(session_items):,} "
                f"sessions"
            )

    generation_time = (
        time.perf_counter()
        - start_time
    )

    if not all_candidate_frames:
        raise RuntimeError(
            "No candidate rows generated."
        )

    candidates = pd.concat(
        all_candidate_frames,
        ignore_index=True,
    )

    print()
    print(
        f"Candidate rows before labels: "
        f"{len(candidates):,}"
    )

    print(
        f"Candidate generation time: "
        f"{generation_time:.3f}s"
    )

    candidates[
        "split"
    ] = candidates[
        "session_id"
    ].map(
        split_map
    )

    if candidates["split"].isna().any():
        raise RuntimeError(
            "Some candidates have no session split."
        )

    candidates = create_tight_labels(
        candidates,
        gt,
    )

    print()
    print(
        "Label distribution:"
    )

    print(
        candidates[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Rows by split:"
    )

    print(
        candidates[
            "split"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Labels by split:"
    )

    print(
        candidates.groupby(
            [
                "split",
                "label",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .to_string()
    )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    duplicates = (
        candidates
        .duplicated(
            subset=[
                "session_id",
                "candidate_ts",
            ]
        )
        .sum()
    )

    if duplicates:
        raise RuntimeError(
            f"Duplicate candidate timestamps: "
            f"{duplicates}"
        )

    numeric = candidates[
        FEATURE_COLUMNS
    ]

    if numeric.isna().any().any():
        raise RuntimeError(
            "NaN values found in features."
        )

    if not np.isfinite(
        numeric.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Non-finite values found in features."
        )

    # Remove constant features.
    constant_features = [
        column
        for column in FEATURE_COLUMNS
        if candidates[column].nunique() <= 1
    ]

    if constant_features:

        print()
        print(
            "Constant features removed:"
        )

        for column in constant_features:
            print(
                f"  - {column}"
            )

    active_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in constant_features
    ]

    print()
    print(
        f"Active features: "
        f"{len(active_features):,}"
    )

    for feature in active_features:
        print(
            f"  - {feature}"
        )

    # Save an explicit copy of the new raw-derived dataset.
    output_columns = [
        "session_id",
        "candidate_ts",
        "label",
        "split",
    ] + active_features

    candidates[
        output_columns
    ].to_csv(
        DATASET_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Saved raw-derived ML dataset:\n"
        f"{DATASET_OUTPUT_FILE}"
    )

    return (
        candidates,
        gt,
        split_df,
        active_features,
    )


# ============================================================
# Merge candidates into boundary predictions
# ============================================================

def merge_candidates(
    positive_candidates,
    merge_gap_seconds,
):
    """
    Chain-based temporal consolidation.

    Consecutive positive candidates within the merge gap
    belong to the same boundary episode.

    The highest-probability candidate is retained.
    """

    if positive_candidates.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "predicted_ts",
                "probability",
                "cluster_size",
            ]
        )

    predictions = []

    for session_id, group in (
        positive_candidates
        .groupby("session_id")
    ):

        group = (
            group
            .sort_values(
                "candidate_ts"
            )
            .reset_index(drop=True)
        )

        cluster = []
        previous_ts = None

        for _, row in group.iterrows():

            current_ts = row[
                "candidate_ts"
            ]

            if (
                previous_ts is None
                or (
                    current_ts
                    - previous_ts
                ).total_seconds()
                <= merge_gap_seconds
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
                        "session_id":
                            session_id,
                        "predicted_ts":
                            best[
                                "candidate_ts"
                            ],
                        "probability":
                            best[
                                "probability"
                            ],
                        "cluster_size":
                            len(cluster_df),
                    }
                )

                cluster = [row]

            previous_ts = current_ts

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
                    "session_id":
                        session_id,
                    "predicted_ts":
                        best[
                            "candidate_ts"
                        ],
                    "probability":
                        best[
                            "probability"
                        ],
                    "cluster_size":
                        len(cluster_df),
                }
            )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# Temporal evaluation
# ============================================================

def evaluate_temporal(
    predictions,
    gt,
    tolerance_seconds,
):
    """
    One-to-one nearest temporal matching.
    """

    total_predictions = len(
        predictions
    )

    total_gt = len(
        gt
    )

    if total_predictions == 0:
        return {
            "matched":
                0,
            "precision":
                0.0,
            "recall":
                0.0,
            "f1":
                0.0,
            "mean_error":
                np.nan,
            "median_error":
                np.nan,
        }

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

    matched_gt = set()
    errors = []

    for _, prediction in (
        predictions.iterrows()
    ):

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

            if (
                error <= tolerance_seconds
                and (
                    best_error is None
                    or error < best_error
                )
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

            errors.append(
                best_error
            )

    matched = len(errors)

    precision = (
        matched
        / total_predictions
    )

    recall = (
        matched
        / total_gt
        if total_gt
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall
        else 0.0
    )

    return {
        "matched":
            matched,
        "precision":
            precision,
        "recall":
            recall,
        "f1":
            f1,
        "mean_error":
            (
                float(
                    np.mean(errors)
                )
                if errors
                else np.nan
            ),
        "median_error":
            (
                float(
                    np.median(errors)
                )
                if errors
                else np.nan
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "DIRECT RAW-JSON "
        "SEGMENTATION MODEL"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Build dataset directly from raw JSON
    # --------------------------------------------------------

    (
        candidates,
        gt,
        split_df,
        active_features,
    ) = build_raw_dataset()

    # --------------------------------------------------------
    # 2. Get train / validation / test
    # --------------------------------------------------------

    train_df = candidates[
        candidates["split"]
        == "train"
    ].copy()

    validation_df = candidates[
        candidates["split"]
        == "validation"
    ].copy()

    test_df = candidates[
        candidates["split"]
        == "test"
    ].copy()

    train_sessions = set(
        train_df[
            "session_id"
        ]
    )

    validation_sessions = set(
        validation_df[
            "session_id"
        ]
    )

    test_sessions = set(
        test_df[
            "session_id"
        ]
    )

    # Explicit session-level leakage checks.
    if train_sessions & validation_sessions:
        raise RuntimeError(
            "Train/validation session leakage detected."
        )

    if train_sessions & test_sessions:
        raise RuntimeError(
            "Train/test session leakage detected."
        )

    if validation_sessions & test_sessions:
        raise RuntimeError(
            "Validation/test session leakage detected."
        )

    print()
    print(
        "=" * 70
    )
    print(
        "SESSION SPLIT"
    )
    print(
        "=" * 70
    )

    print(
        f"Train sessions: "
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
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # 3. Prepare matrices
    # --------------------------------------------------------

    X_train = train_df[
        active_features
    ]

    y_train = train_df[
        "label"
    ].astype(int)

    X_validation = validation_df[
        active_features
    ]

    validation_gt = gt[
        gt["session_id"].isin(
            validation_sessions
        )
    ].copy()

    test_gt = gt[
        gt["session_id"].isin(
            test_sessions
        )
    ].copy()

    # --------------------------------------------------------
    # 4. Train models and select using VALIDATION ONLY
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "VALIDATION MODEL SELECTION"
    )
    print(
        "=" * 70
    )

    validation_results = []

    best_result = None
    best_validation_base = None

    for max_iter in MAX_ITER_VALUES:

        print()
        print(
            "-" * 70
        )

        print(
            f"Training HistGradientBoosting "
            f"with max_iter={max_iter}"
        )

        model = (
            HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=RANDOM_SEED,

                # We intentionally disable sklearn's internal
                # classification early stopping.
                #
                # Our selection criterion is final temporal
                # segmentation F1 on the validation sessions.
                early_stopping=False,
            )
        )

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        train_time = (
            time.perf_counter()
            - start
        )

        print(
            f"Training time: "
            f"{train_time:.3f}s"
        )

        print(
            f"Actual boosting iterations: "
            f"{model.n_iter_}"
        )

        validation_probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_base = (
            validation_df[
                [
                    "session_id",
                    "candidate_ts",
                ]
            ]
            .copy()
        )

        validation_base[
            "probability"
        ] = validation_probabilities

        for threshold in THRESHOLDS:

            positive_candidates = (
                validation_base[
                    validation_base[
                        "probability"
                    ]
                    >= threshold
                ]
                .copy()
            )

            predictions = merge_candidates(
                positive_candidates,
                MERGE_GAP_SECONDS,
            )

            metrics = evaluate_temporal(
                predictions,
                validation_gt,
                5,
            )

            result = {
                "max_iter":
                    max_iter,

                "threshold":
                    float(threshold),

                "actual_iterations":
                    int(
                        model.n_iter_
                    ),

                "train_time_seconds":
                    train_time,

                "positive_candidates":
                    len(
                        positive_candidates
                    ),

                "predicted_boundaries":
                    len(
                        predictions
                    ),

                "gt_boundaries":
                    len(
                        validation_gt
                    ),

                "matched_boundaries":
                    metrics[
                        "matched"
                    ],

                "precision":
                    metrics[
                        "precision"
                    ],

                "recall":
                    metrics[
                        "recall"
                    ],

                "f1":
                    metrics[
                        "f1"
                    ],

                "mean_error_seconds":
                    metrics[
                        "mean_error"
                    ],

                "median_error_seconds":
                    metrics[
                        "median_error"
                    ],
            }

            validation_results.append(
                result
            )

            if (
                best_result is None
                or result["f1"]
                > best_result["f1"]
            ):
                best_result = result
                # Keep the probability table for the model that produced
                # the currently best validation configuration. This is
                # later used by the local-peak post-processing experiment.
                best_validation_base = validation_base.copy()

        print(
            "Best validation F1 after "
            f"max_iter={max_iter}: "
            f"{best_result['f1']:.4f}"
        )

    validation_results_df = (
        pd.DataFrame(
            validation_results
        )
        .sort_values(
            [
                "f1",
                "recall",
                "precision",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    validation_results_df.to_csv(
        VALIDATION_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if best_validation_base is None:
        raise RuntimeError(
            "Best validation probability table was not created."
        )

    best_validation_base.to_csv(
        VALIDATION_PROBABILITIES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Saved selected validation probabilities:\n"
        f"{VALIDATION_PROBABILITIES_FILE}"
    )

    # --------------------------------------------------------
    # 5. Print validation results
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "VALIDATION RESULTS — TOP CONFIGURATIONS"
    )
    print(
        "=" * 70
    )

    print(
        validation_results_df[
            [
                "max_iter",
                "threshold",
                "actual_iterations",
                "positive_candidates",
                "predicted_boundaries",
                "gt_boundaries",
                "precision",
                "recall",
                "f1",
                "mean_error_seconds",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 70
    )
    print(
        "SELECTED VALIDATION CONFIGURATION"
    )
    print(
        "=" * 70
    )

    selected_max_iter = int(
        best_result[
            "max_iter"
        ]
    )

    selected_threshold = float(
        best_result[
            "threshold"
        ]
    )

    print(
        f"max_iter: "
        f"{selected_max_iter}"
    )

    print(
        f"threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation precision: "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{best_result['f1']:.4f}"
    )

    print(
        f"Validation predicted boundaries: "
        f"{best_result['predicted_boundaries']}"
    )

    print(
        f"Validation GT boundaries: "
        f"{best_result['gt_boundaries']}"
    )

    if best_result["f1"] >= 0.80:

        print()
        print(
            "VALIDATION TARGET REACHED: "
            "F1 >= 0.80"
        )

    else:

        print()
        print(
            "VALIDATION TARGET NOT REACHED: "
            "F1 < 0.80"
        )

    # --------------------------------------------------------
    # 6. Train selected model ONE FINAL TIME
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL MODEL — TRAINING ON TRAIN SPLIT"
    )
    print(
        "=" * 70
    )

    final_model = (
        HistGradientBoostingClassifier(
            max_iter=selected_max_iter,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
            early_stopping=False,
        )
    )

    final_start = time.perf_counter()

    final_model.fit(
        X_train,
        y_train,
    )

    final_train_time = (
        time.perf_counter()
        - final_start
    )

    print(
        f"Final training time: "
        f"{final_train_time:.3f}s"
    )

    print(
        f"Final actual iterations: "
        f"{final_model.n_iter_}"
    )

    # --------------------------------------------------------
    # 7. Evaluate once on untouched TEST split
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL TEST EVALUATION"
    )
    print(
        "=" * 70
    )

    test_probabilities = (
        final_model.predict_proba(
            test_df[
                active_features
            ]
        )[:, 1]
    )

    test_base = (
        test_df[
            [
                "session_id",
                "candidate_ts",
            ]
        ]
        .copy()
    )

    test_base[
        "probability"
    ] = test_probabilities

    positive_test_candidates = (
        test_base[
            test_base[
                "probability"
            ]
            >= selected_threshold
        ]
        .copy()
    )

    test_predictions = merge_candidates(
        positive_test_candidates,
        MERGE_GAP_SECONDS,
    )

    print()
    print(
        f"Test positive candidates: "
        f"{len(positive_test_candidates):,}"
    )

    print(
        f"Test predicted boundaries: "
        f"{len(test_predictions):,}"
    )

    print(
        f"Test GT boundaries: "
        f"{len(test_gt):,}"
    )

    # --------------------------------------------------------
    # 8. Test temporal metrics
    # --------------------------------------------------------

    test_results = []

    for tolerance in TOLERANCES:

        metrics = evaluate_temporal(
            test_predictions,
            test_gt,
            tolerance,
        )

        test_results.append(
            {
                "tolerance_seconds":
                    tolerance,

                "gt_boundaries":
                    len(test_gt),

                "predicted_boundaries":
                    len(
                        test_predictions
                    ),

                "matched_boundaries":
                    metrics[
                        "matched"
                    ],

                "precision":
                    metrics[
                        "precision"
                    ],

                "recall":
                    metrics[
                        "recall"
                    ],

                "f1":
                    metrics[
                        "f1"
                    ],

                "mean_boundary_error_seconds":
                    metrics[
                        "mean_error"
                    ],

                "median_boundary_error_seconds":
                    metrics[
                        "median_error"
                    ],
            }
        )

    test_results_df = pd.DataFrame(
        test_results
    )

    print()
    print(
        "TEST TEMPORAL RESULTS"
    )

    print(
        test_results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. Save test outputs
    # --------------------------------------------------------

    test_results_df.to_csv(
        TEST_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    test_predictions.to_csv(
        TEST_BOUNDARIES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 10. Test session boundary counts
    # --------------------------------------------------------

    print()
    print(
        "TEST SESSION BOUNDARY COUNTS"
    )

    session_count_rows = []

    for session_id in sorted(
        test_sessions
    ):

        predicted_count = int(
            (
                test_predictions[
                    "session_id"
                ]
                == session_id
            ).sum()
        )

        gt_count = int(
            (
                test_gt[
                    "session_id"
                ]
                == session_id
            ).sum()
        )

        count_error = abs(
            predicted_count
            - gt_count
        )

        relative_error = (
            count_error / gt_count
            if gt_count
            else np.nan
        )

        session_count_rows.append(
            {
                "session_id":
                    session_id,

                "predicted_boundaries":
                    predicted_count,

                "gt_boundaries":
                    gt_count,

                "predicted_segments":
                    predicted_count + 1,

                "gt_segments":
                    gt_count + 1,

                "absolute_count_error":
                    count_error,

                "relative_count_error":
                    relative_error,
            }
        )

        print(
            f"{session_id}: "
            f"predicted={predicted_count}, "
            f"GT={gt_count}, "
            f"error={count_error}"
        )

    session_count_df = pd.DataFrame(
        session_count_rows
    )

    session_count_output = (
        OUTPUT_DIR
        / "segment_model_test_session_counts.csv"
    )

    session_count_df.to_csv(
        session_count_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Mean session count error: "
        f"{session_count_df['absolute_count_error'].mean():.2f}"
    )

    print(
        f"Median session count error: "
        f"{session_count_df['absolute_count_error'].median():.2f}"
    )

    print(
        f"Mean relative session count error: "
        f"{session_count_df['relative_count_error'].mean():.4f}"
    )

    # --------------------------------------------------------
    # 11. Save final probability table
    # --------------------------------------------------------

    test_probability_output = (
        OUTPUT_DIR
        / "segment_model_test_probabilities.csv"
    )

    test_base.to_csv(
        test_probability_output,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 12. Final output summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "EXPERIMENT COMPLETE"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "Selected configuration:"
    )

    print(
        f"  max_iter = "
        f"{selected_max_iter}"
    )

    print(
        f"  threshold = "
        f"{selected_threshold:.2f}"
    )

    print(
        f"  merge_gap = "
        f"{MERGE_GAP_SECONDS}s"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  Validation results:\n"
        f"  {VALIDATION_RESULTS_FILE}"
    )

    print(
        f"  Validation probabilities:\n"
        f"  {VALIDATION_PROBABILITIES_FILE}"
    )

    print(
        f"  Test results:\n"
        f"  {TEST_RESULTS_FILE}"
    )

    print(
        f"  Test boundaries:\n"
        f"  {TEST_BOUNDARIES_FILE}"
    )

    print(
        f"  Test probabilities:\n"
        f"  {test_probability_output}"
    )

    print(
        f"  Test session counts:\n"
        f"  {session_count_output}"
    )

    print(
        f"  Raw ML dataset:\n"
        f"  {DATASET_OUTPUT_FILE}"
    )

    print(
        f"  Session split:\n"
        f"  {SPLIT_FILE}"
    )


if __name__ == "__main__":
    main()