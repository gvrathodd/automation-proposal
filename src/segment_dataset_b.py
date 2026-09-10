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

A_ML_DATASET_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "raw_segment_ml_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

BOUNDARIES_OUTPUT = (
    OUTPUT_DIR
    / "dataset_b_boundaries.csv"
)

SEGMENTS_OUTPUT = (
    OUTPUT_DIR
    / "dataset_b_segments.csv"
)


# ============================================================
# Selected configuration from Dataset A
# ============================================================

RANDOM_SEED = 42

MAX_ITER = 200
THRESHOLD = 0.20

CANDIDATE_SPACING_SECONDS = 2
WINDOW_SECONDS = 6
MERGE_GAP_SECONDS = 2


# ============================================================
# Features
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
# Prepare raw events
# ============================================================

def prepare_session_events(events):
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
# Candidate feature calculation
# ============================================================

def calculate_window_features(
    events,
    start_ms,
    end_ms,
):
    selected = [
        event
        for event in events
        if (
            start_ms
            <= event["timestamp_ms"]
            < end_ms
        )
    ]

    event_types = [
        event["event_type"]
        for event in selected
    ]

    return {
        "event_count":
            len(selected),

        "app_switches":
            sum(
                is_app_switch_event(event_type)
                for event_type in event_types
            ),

        "navigation":
            sum(
                is_navigation_event(event_type)
                for event_type in event_types
            ),

        "browser_clicks":
            sum(
                is_browser_click_event(event_type)
                for event_type in event_types
            ),

        "mouse_clicks":
            sum(
                is_mouse_click_event(event_type)
                for event_type in event_types
            ),

        "window_changes":
            sum(
                is_window_change_event(event_type)
                for event_type in event_types
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
                is_keyboard_event(event_type)
                for event_type in event_types
            ),
    }


# ============================================================
# Candidate generation
# ============================================================

def build_session_candidates(
    session_id,
    events,
):
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
# Train final Dataset A model
# ============================================================

def train_final_model():
    print("=" * 70)
    print("TRAINING FINAL DATASET A MODEL")
    print("=" * 70)

    if not A_ML_DATASET_FILE.exists():
        raise FileNotFoundError(
            "Dataset A ML dataset not found:\n"
            f"{A_ML_DATASET_FILE}"
        )

    df = pd.read_csv(
        A_ML_DATASET_FILE
    )

    required_columns = {
        "session_id",
        "candidate_ts",
        "label",
        *FEATURE_COLUMNS,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Dataset A ML dataset is missing "
            "required columns:\n"
            + "\n".join(sorted(missing))
        )

    # We intentionally use all Dataset A labels now.
    # Hyperparameters were already selected using
    # the separate validation process.
    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "label"
    ].astype(int)

    print(
        f"Dataset A training rows: "
        f"{len(df):,}"
    )

    print(
        f"Positive labels: "
        f"{int(y.sum()):,}"
    )

    print(
        f"Negative labels: "
        f"{int((y == 0).sum()):,}"
    )

    model = (
        HistGradientBoostingClassifier(
            max_iter=MAX_ITER,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
            early_stopping=False,
        )
    )

    start = time.perf_counter()

    model.fit(
        X,
        y,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Training time: "
        f"{elapsed:.3f}s"
    )

    print(
        f"Actual boosting iterations: "
        f"{model.n_iter_}"
    )

    return model


# ============================================================
# Merge positive candidates
# ============================================================

def merge_candidates(
    positive_candidates,
    merge_gap_seconds,
):
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
            .sort_values("candidate_ts")
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
# Build Dataset B segments
# ============================================================

def build_segments(
    session_events,
    boundaries,
):
    segment_rows = []

    grouped_boundaries = {
        session_id:
            group
            .sort_values("predicted_ts")
            .reset_index(drop=True)
        for session_id, group
        in boundaries.groupby("session_id")
    }

    for session_id, events in session_events.items():

        prepared = prepare_session_events(
            events
        )

        if not prepared:
            continue

        session_start = pd.to_datetime(
            prepared[0]["timestamp_ms"],
            unit="ms",
            utc=True,
        )

        session_end = pd.to_datetime(
            prepared[-1]["timestamp_ms"],
            unit="ms",
            utc=True,
        )

        session_boundaries = (
            grouped_boundaries
            .get(
                session_id,
                pd.DataFrame()
            )
        )

        boundary_times = []

        if not session_boundaries.empty:
            boundary_times = (
                session_boundaries[
                    "predicted_ts"
                ]
                .tolist()
            )

        # Remove boundaries that are not
        # strictly inside the session.
        boundary_times = [
            ts
            for ts in boundary_times
            if (
                session_start
                < ts
                < session_end
            )
        ]

        points = (
            [session_start]
            + boundary_times
            + [session_end]
        )

        for index in range(
            len(points) - 1
        ):
            segment_start = points[index]
            segment_end = points[index + 1]

            duration = (
                segment_end
                - segment_start
            ).total_seconds()

            segment_events = [
                event
                for event in prepared
                if (
                    segment_start.timestamp() * 1000
                    <= event["timestamp_ms"]
                    <= segment_end.timestamp() * 1000
                )
            ]

            apps = sorted(
                {
                    event["app"]
                    for event in segment_events
                    if event["app"]
                }
            )

            event_types = [
                event["event_type"]
                for event in segment_events
            ]

            segment_rows.append(
                {
                    "session_id":
                        session_id,

                    "segment_index":
                        index + 1,

                    "start":
                        segment_start,

                    "end":
                        segment_end,

                    "duration_seconds":
                        duration,

                    "event_count":
                        len(segment_events),

                    "unique_apps":
                        len(apps),

                    "apps":
                        " | ".join(apps),

                    "unique_event_types":
                        len(set(event_types)),

                    "event_types":
                        " | ".join(
                            sorted(
                                set(event_types)
                            )
                        ),
                }
            )

    return pd.DataFrame(
        segment_rows
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DATASET B SEGMENTATION")
    print("=" * 70)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            "Dataset root not found:\n"
            f"{DATA_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Train final model using Dataset A
    # --------------------------------------------------------

    model = train_final_model()

    # --------------------------------------------------------
    # 2. Load Dataset B
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING DATASET B")
    print("=" * 70)

    all_sessions = load_all_events(
        DATA_ROOT
    )

    dataset_b_sessions = {}

    for session_id, events in (
        all_sessions.items()
    ):

        datasets = {
            event.get("_dataset")
            for event in events
            if event.get("_dataset")
        }

        if "B" in datasets:
            dataset_b_sessions[
                session_id
            ] = events

    print(
        f"Dataset B sessions: "
        f"{len(dataset_b_sessions):,}"
    )

    if not dataset_b_sessions:
        raise RuntimeError(
            "No Dataset B sessions found."
        )

    # --------------------------------------------------------
    # 3. Generate candidates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING DATASET B CANDIDATES")
    print("=" * 70)

    candidate_frames = []

    total_start = time.perf_counter()

    session_items = list(
        dataset_b_sessions.items()
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
            candidate_frames.append(
                candidates
            )

        print(
            f"Processed "
            f"{position}/"
            f"{len(session_items)}: "
            f"{session_id} "
            f"({len(candidates):,} candidates)"
        )

    if not candidate_frames:
        raise RuntimeError(
            "No Dataset B candidates generated."
        )

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    generation_time = (
        time.perf_counter()
        - total_start
    )

    print()
    print(
        f"Total Dataset B candidates: "
        f"{len(candidates):,}"
    )

    print(
        f"Candidate generation time: "
        f"{generation_time:.3f}s"
    )

    # --------------------------------------------------------
    # 4. Predict
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("APPLYING DATASET A MODEL TO DATASET B")
    print("=" * 70)

    probabilities = (
        model.predict_proba(
            candidates[
                FEATURE_COLUMNS
            ]
        )[:, 1]
    )

    candidates[
        "probability"
    ] = probabilities

    positive_candidates = (
        candidates[
            candidates[
                "probability"
            ]
            >= THRESHOLD
        ]
        .copy()
    )

    print(
        f"Positive candidates: "
        f"{len(positive_candidates):,}"
    )

    print(
        f"Positive rate: "
        f"{len(positive_candidates) / len(candidates):.4f}"
    )

    # --------------------------------------------------------
    # 5. Consolidate
    # --------------------------------------------------------

    boundaries = merge_candidates(
        positive_candidates,
        MERGE_GAP_SECONDS,
    )

    boundaries = (
        boundaries
        .sort_values(
            [
                "session_id",
                "predicted_ts",
            ]
        )
        .reset_index(drop=True)
    )

    boundaries[
        "boundary_index"
    ] = (
        boundaries
        .groupby("session_id")
        .cumcount()
        + 1
    )

    # Put index before the other columns.
    boundaries = boundaries[
        [
            "session_id",
            "boundary_index",
            "predicted_ts",
            "probability",
            "cluster_size",
        ]
    ]

    print(
        f"Predicted Dataset B boundaries: "
        f"{len(boundaries):,}"
    )

    print(
        f"Average boundaries/session: "
        f"{len(boundaries) / len(dataset_b_sessions):.2f}"
    )

    boundaries.to_csv(
        BOUNDARIES_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 6. Build process segments
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING DATASET B PROCESS SEGMENTS")
    print("=" * 70)

    segments = build_segments(
        dataset_b_sessions,
        boundaries,
    )

    if segments.empty:
        raise RuntimeError(
            "No Dataset B segments generated."
        )

    segments = (
        segments
        .sort_values(
            [
                "session_id",
                "segment_index",
            ]
        )
        .reset_index(drop=True)
    )

    segments.to_csv(
        SEGMENTS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET B SEGMENTATION COMPLETE")
    print("=" * 70)

    print(
        f"Sessions: "
        f"{len(dataset_b_sessions):,}"
    )

    print(
        f"Boundaries: "
        f"{len(boundaries):,}"
    )

    print(
        f"Segments: "
        f"{len(segments):,}"
    )

    print(
        f"Mean segments/session: "
        f"{segments.groupby('session_id').size().mean():.2f}"
    )

    print()
    print("Segment count by session:")

    session_counts = (
        segments
        .groupby("session_id")
        .size()
        .sort_index()
    )

    for session_id, count in (
        session_counts.items()
    ):
        print(
            f"  {session_id}: "
            f"{count}"
        )

    print()
    print("Outputs:")

    print(
        f"  Boundaries:\n"
        f"  {BOUNDARIES_OUTPUT}"
    )

    print(
        f"  Segments:\n"
        f"  {SEGMENTS_OUTPUT}"
    )


if __name__ == "__main__":
    main()