
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
OUTPUTS = ROOT / "outputs"

PRED_FILE = (
    OUTPUTS / "segment_model_test_boundaries.csv"
)
SPLIT_FILE = (
    OUTPUTS / "segment_model_split.csv"
)
B_SEGMENTS_FILE = (
    OUTPUTS / "dataset_b_segments.csv"
)

OUT_A = (
    OUTPUTS / "non_process_filter_dataset_a_segments.csv"
)
OUT_A_SCORES = (
    OUTPUTS / "non_process_filter_a_cv_predictions.csv"
)
OUT_B = (
    OUTPUTS / "non_process_filter_dataset_b_scores.csv"
)

RANDOM_STATE = 42


# ============================================================
# PURPOSE
# ============================================================
#
# We are NOT changing the Step-1 boundary model.
#
# We are adding a separate binary "is this segment process work?"
# diagnostic/filter.
#
# Dataset A target:
#   A predicted segment is PROCESS_WORK when it overlaps a
#   complete GT execution interval.
#
# We report both:
#   1) any positive GT overlap
#   2) majority GT overlap (>=50% of predicted duration)
#
# The classifier is validated with leave-one-session-out CV over
# the Dataset-A test sessions. It is deliberately simple and
# interpretable.
#
# Then, after validation, the classifier is fitted on all A test
# segments and scored on all Dataset-B segments, producing a
# suggested work/other flag plus probability.
#
# This is a supporting Step-2 filter, NOT the final segments.jsonl.
# We review the B scores before deciding what gets labeled "other".
# ============================================================


def parse_ts(value):
    try:
        return pd.to_datetime(
            value,
            utc=True,
        )
    except Exception:
        return pd.NaT


def read_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Exact session/manfiest discovery
# ============================================================

def discover_manifests():
    manifests = {}

    for path in DATA_ROOT.rglob(
        "gt_manifest.json"
    ):
        manifests[
            path.parent.name
        ] = path

    return manifests


def session_manifest_path(session_id):
    manifests = discover_manifests()

    if session_id not in manifests:
        raise FileNotFoundError(
            f"No gt_manifest.json found for "
            f"session {session_id}"
        )

    return manifests[
        session_id
    ]


def manifest_session_bounds(manifest_path):
    data = read_json(
        manifest_path
    )

    session = data.get(
        "session",
        {},
    )

    start = parse_ts(
        session.get("start_ts")
    )

    end = parse_ts(
        session.get("end_ts")
    )

    return start, end


# ============================================================
# GT executions
# ============================================================

def load_complete_gt_executions():
    rows = []

    for session_id, manifest_path in sorted(
        discover_manifests().items()
    ):
        data = read_json(
            manifest_path
        )

        for process in data.get(
            "processes",
            [],
        ):
            if not isinstance(process, dict):
                continue

            process_code = str(
                process.get(
                    "code",
                    "",
                )
                or ""
            )

            process_name = str(
                process.get(
                    "family_name",
                    "",
                )
                or ""
            )

            for execution in process.get(
                "executions",
                [],
            ):
                if not isinstance(
                    execution,
                    dict,
                ):
                    continue

                start = parse_ts(
                    execution.get(
                        "start_ts"
                    )
                )

                end = parse_ts(
                    execution.get(
                        "end_ts"
                    )
                )

                if (
                    pd.isna(start)
                    or pd.isna(end)
                ):
                    continue

                duration = (
                    end - start
                ).total_seconds()

                if duration < 0:
                    continue

                rows.append(
                    {
                        "session_id":
                            session_id,
                        "start":
                            start,
                        "end":
                            end,
                        "duration_seconds":
                            duration,
                        "process_code":
                            str(
                                execution.get(
                                    "code",
                                    process_code,
                                )
                                or process_code
                            ),
                        "process_name":
                            process_name,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# Existing final Step-1 predicted boundaries
# ============================================================

def load_test_sessions():
    if not SPLIT_FILE.exists():
        return None

    split = pd.read_csv(
        SPLIT_FILE,
        keep_default_na=False,
    )

    if not {
        "session_id",
        "split",
    }.issubset(
        split.columns
    ):
        return None

    return set(
        split[
            split["split"] == "test"
        ]["session_id"]
    )


def load_predicted_boundaries():
    if not PRED_FILE.exists():
        raise FileNotFoundError(
            PRED_FILE
        )

    df = pd.read_csv(
        PRED_FILE,
        keep_default_na=False,
    )

    required = {
        "session_id",
        "predicted_ts",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Prediction file is missing:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    df["predicted_ts"] = pd.to_datetime(
        df["predicted_ts"],
        utc=True,
    )

    test_sessions = (
        load_test_sessions()
    )

    if test_sessions is not None:
        df = df[
            df["session_id"].isin(
                test_sessions
            )
        ].copy()

    return df.sort_values(
        [
            "session_id",
            "predicted_ts",
        ]
    )


def build_predicted_segments(
    boundaries
):
    rows = []

    for session_id, group in boundaries.groupby(
        "session_id",
        sort=True,
    ):

        manifest_path = (
            session_manifest_path(
                session_id
            )
        )

        session_start, session_end = (
            manifest_session_bounds(
                manifest_path
            )
        )

        if (
            pd.isna(session_start)
            or pd.isna(session_end)
        ):
            continue

        points = [
            session_start
        ]

        for timestamp in (
            group[
                "predicted_ts"
            ]
            .dropna()
            .sort_values()
            .tolist()
        ):

            if (
                timestamp > session_start
                and timestamp < session_end
            ):
                points.append(
                    timestamp
                )

        points.append(
            session_end
        )

        points = sorted(
            set(points)
        )

        for index, (
            start,
            end,
        ) in enumerate(
            zip(
                points[:-1],
                points[1:],
            )
        ):

            if end <= start:
                continue

            rows.append(
                {
                    "session_id":
                        session_id,
                    "pred_segment_index":
                        index,
                    "pred_start":
                        start,
                    "pred_end":
                        end,
                    "pred_duration_seconds":
                        (
                            end - start
                        ).total_seconds(),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Raw event extraction
# ============================================================

def event_timestamp(event):
    if event.get(
        "timestamp_ms"
    ) is not None:
        try:
            return pd.to_datetime(
                int(
                    event[
                        "timestamp_ms"
                    ]
                ),
                unit="ms",
                utc=True,
            )
        except Exception:
            pass

    for key in (
        "timestamp_iso",
        "timestamp",
        "ts_utc",
    ):
        if key in event:
            parsed = parse_ts(
                event[key]
            )

            if not pd.isna(parsed):
                return parsed

    return pd.NaT


def get_app(event):
    context = (
        event.get(
            "context"
        )
        or {}
    )

    active_app = (
        context.get(
            "active_app"
        )
        or {}
    )

    value = (
        active_app.get(
            "app_name"
        )
    )

    return (
        value.strip().lower()
        if isinstance(
            value,
            str,
        )
        and value.strip()
        else ""
    )


def get_window_title(event):
    context = (
        event.get(
            "context"
        )
        or {}
    )

    active_app = (
        context.get(
            "active_app"
        )
        or {}
    )

    value = (
        active_app.get(
            "window_title"
        )
    )

    return (
        value.strip().lower()
        if isinstance(
            value,
            str,
        )
        and value.strip()
        else ""
    )


def get_url(event):
    context = (
        event.get(
            "context"
        )
        or {}
    )

    tab = (
        context.get(
            "active_browser_tab"
        )
        or {}
    )

    if isinstance(
        tab,
        dict,
    ):
        for key in (
            "url",
            "href",
        ):
            value = tab.get(
                key
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                return value.strip()

    payload = (
        event.get(
            "payload"
        )
        or {}
    )

    if isinstance(
        payload,
        dict,
    ):
        for key in (
            "url",
            "current_url",
            "target_url",
            "href",
        ):
            value = payload.get(
                key
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                return value.strip()

    return ""


def normalized_path(url):
    if not url:
        return ""

    try:
        from urllib.parse import urlsplit

        path = (
            urlsplit(
                url
            ).path
            or ""
        )
    except Exception:
        path = url

    path = re.sub(
        r"/(?:E\d{3,8}|[A-Z]{1,8}-\d{4,12}(?:-\d{1,6})?|\d{3,})",
        "/{id}",
        path,
        flags=re.I,
    )

    path = re.sub(
        r"/+",
        "/",
        path,
    )

    return path.lower()


def load_session_events(
    session_id
):
    folder = (
        session_manifest_path(
            session_id
        ).parent
    )

    events = []

    for event_file in sorted(
        folder.rglob(
            "events.jsonl"
        )
    ):
        try:
            with event_file.open(
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        event = json.loads(
                            line
                        )
                    except Exception:
                        continue

                    timestamp = event_timestamp(
                        event
                    )

                    if pd.isna(
                        timestamp
                    ):
                        continue

                    events.append(
                        {
                            "timestamp":
                                timestamp,
                            "event":
                                event,
                        }
                    )

        except OSError:
            continue

    events.sort(
        key=lambda x:
            x["timestamp"]
    )

    return events


def aggregate_segment_features(
    events,
    start,
    end,
):
    event_count = 0
    screenshot_count = 0
    app_switch_count = 0
    browser_click_count = 0
    browser_form_input_count = 0
    browser_navigation_count = 0
    mouse_click_count = 0
    mouse_scroll_count = 0
    keyboard_count = 0
    shortcut_count = 0
    clipboard_count = 0
    window_change_count = 0

    apps = set()
    paths = set()
    windows = set()
    layers = set()
    event_types = set()

    for item in events:

        timestamp = item["timestamp"]

        if (
            timestamp < start
            or timestamp > end
        ):
            continue

        event = item["event"]

        event_count += 1

        event_type = str(
            event.get(
                "event_type",
                event.get(
                    "event",
                    "",
                ),
            )
            or ""
        ).lower()

        layer = str(
            event.get(
                "layer",
                "",
            )
            or ""
        )

        if layer:
            layers.add(
                layer
            )

        if event_type:
            event_types.add(
                event_type
            )

        app = get_app(event)

        if app:
            apps.add(
                app
            )

        path = normalized_path(
            get_url(event)
        )

        if path:
            paths.add(
                path
            )

        window = get_window_title(
            event
        )

        if window:
            windows.add(
                window
            )

        if event_type == "screenshot_smart":
            screenshot_count += 1

        elif event_type == "app_switch":
            app_switch_count += 1

        elif event_type == "browser_click":
            browser_click_count += 1

        elif event_type == "browser_form_input":
            browser_form_input_count += 1

        elif event_type == "browser_navigation":
            browser_navigation_count += 1

        elif event_type == "mouse_click":
            mouse_click_count += 1

        elif event_type == "mouse_scroll":
            mouse_scroll_count += 1

        elif event_type == "keystroke":
            keyboard_count += 1

        elif event_type == "shortcut":
            shortcut_count += 1

        elif event_type == "clipboard_change":
            clipboard_count += 1

        elif event_type in {
            "window_title_change",
            "window_state_change",
        }:
            window_change_count += 1

    return {
        "event_count":
            event_count,
        "screenshot_count":
            screenshot_count,
        "app_switch_count":
            app_switch_count,
        "browser_click_count":
            browser_click_count,
        "browser_form_input_count":
            browser_form_input_count,
        "browser_navigation_count":
            browser_navigation_count,
        "mouse_click_count":
            mouse_click_count,
        "mouse_scroll_count":
            mouse_scroll_count,
        "keyboard_count":
            keyboard_count,
        "shortcut_count":
            shortcut_count,
        "clipboard_count":
            clipboard_count,
        "window_change_count":
            window_change_count,
        "unique_app_count":
            len(apps),
        "unique_path_count":
            len(paths),
        "unique_window_count":
            len(windows),
        "unique_layer_count":
            len(layers),
        "unique_event_type_count":
            len(event_types),
        "has_browser":
            int(
                any(
                    t.startswith(
                        "browser_"
                    )
                    for t in event_types
                )
            ),
        "has_clipboard":
            int(
                clipboard_count > 0
            ),
        "has_office":
            int(
                any(
                    (
                        "word" in app
                        or "excel" in app
                    )
                    for app in apps
                )
            ),
        "has_notepad":
            int(
                any(
                    "notepad" in app
                    for app in apps
                )
            ),
    }


# ============================================================
# GT overlap labeling
# ============================================================

def overlap_seconds(
    seg_start,
    seg_end,
    gt_group,
):
    total = 0.0

    for _, gt in gt_group.iterrows():

        if (
            gt["end"]
            <= seg_start
            or gt["start"]
            >= seg_end
        ):
            continue

        start = max(
            seg_start,
            gt["start"],
        )

        end = min(
            seg_end,
            gt["end"],
        )

        if end > start:
            total += (
                end - start
            ).total_seconds()

    return total


def build_a_training_table():
    print(
        "Loading complete Dataset-A GT executions..."
    )

    gt = (
        load_complete_gt_executions()
    )

    print(
        f"Complete GT executions: "
        f"{len(gt):,}"
    )

    print(
        "Loading existing final Step-1 predicted test boundaries..."
    )

    boundaries = (
        load_predicted_boundaries()
    )

    segments = (
        build_predicted_segments(
            boundaries
        )
    )

    print(
        f"Predicted test segments: "
        f"{len(segments):,}"
    )

    feature_cache = {}

    rows = []

    for i, (
        session_id,
        group,
    ) in enumerate(
        segments.groupby(
            "session_id",
            sort=True,
        ),
        start=1,
    ):

        print(
            f"Extracting segment features "
            f"{i}/"
            f"{segments['session_id'].nunique()}: "
            f"{session_id}"
        )

        feature_cache[
            session_id
        ] = load_session_events(
            session_id
        )

        gt_group = gt[
            gt[
                "session_id"
            ]
            == session_id
        ]

        for _, segment in group.iterrows():

            features = aggregate_segment_features(
                feature_cache[
                    session_id
                ],
                segment[
                    "pred_start"
                ],
                segment[
                    "pred_end"
                ],
            )

            overlap = overlap_seconds(
                segment[
                    "pred_start"
                ],
                segment[
                    "pred_end"
                ],
                gt_group,
            )

            segment_duration = (
                segment[
                    "pred_duration_seconds"
                ]
            )

            overlap_fraction = (
                overlap / segment_duration
                if segment_duration > 0
                else 0.0
            )

            row = {
                "session_id":
                    session_id,
                "pred_segment_index":
                    int(
                        segment[
                            "pred_segment_index"
                        ]
                    ),
                "start":
                    segment[
                        "pred_start"
                    ],
                "end":
                    segment[
                        "pred_end"
                    ],
                "duration_seconds":
                    segment_duration,
                "gt_overlap_seconds":
                    overlap,
                "gt_overlap_fraction":
                    overlap_fraction,

                # Any overlap is a permissive "process-related"
                # target.
                "process_work_any_overlap":
                    int(
                        overlap > 0
                    ),

                # Majority overlap is a cleaner classification target.
                "process_work_majority_overlap":
                    int(
                        overlap_fraction
                        >= 0.50
                    ),
            }

            row.update(
                features
            )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Leave-one-session-out validation
# ============================================================

FEATURES = [
    "duration_seconds",
    "event_count",
    "screenshot_count",
    "app_switch_count",
    "browser_click_count",
    "browser_form_input_count",
    "browser_navigation_count",
    "mouse_click_count",
    "mouse_scroll_count",
    "keyboard_count",
    "shortcut_count",
    "clipboard_count",
    "window_change_count",
    "unique_app_count",
    "unique_path_count",
    "unique_window_count",
    "unique_layer_count",
    "unique_event_type_count",
    "has_browser",
    "has_clipboard",
    "has_office",
    "has_notepad",
]


def make_model():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def cross_validate(
    df,
    target_column,
):
    all_predictions = []

    sessions = sorted(
        df[
            "session_id"
        ].unique()
    )

    for session_id in sessions:

        train = df[
            df[
                "session_id"
            ]
            != session_id
        ]

        test = df[
            df[
                "session_id"
            ]
            == session_id
        ]

        if (
            train[target_column].nunique()
            < 2
        ):
            continue

        model = make_model()

        model.fit(
            train[FEATURES],
            train[target_column],
        )

        probability = model.predict_proba(
            test[FEATURES]
        )[:, 1]

        prediction = (
            probability >= 0.50
        ).astype(int)

        output = test[
            [
                "session_id",
                "pred_segment_index",
                target_column,
            ]
        ].copy()

        output[
            "predicted_probability"
        ] = probability

        output[
            "predicted"
        ] = prediction

        all_predictions.append(
            output
        )

    if not all_predictions:
        return pd.DataFrame()

    return pd.concat(
        all_predictions,
        ignore_index=True,
    )


def print_metrics(
    predictions,
    target_column,
):
    if predictions.empty:
        return

    y_true = predictions[
        target_column
    ]

    y_pred = predictions[
        "predicted"
    ]

    print()
    print(
        f"Target: {target_column}"
    )

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_true, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"F1:        "
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )


# ============================================================
# Dataset B scoring
# ============================================================

def score_dataset_b(
    a_df,
    target_column,
):
    if not B_SEGMENTS_FILE.exists():
        print()
        print(
            "Dataset B segment file not found:"
        )
        print(
            B_SEGMENTS_FILE
        )
        print(
            "Skipping B scoring."
        )
        return

    b = pd.read_csv(
        B_SEGMENTS_FILE,
        keep_default_na=False,
    )

    required = {
        "session_id",
        "start",
        "end",
    }

    missing = (
        required
        - set(b.columns)
    )

    if missing:
        print(
            "Dataset B segment file missing:"
        )
        print(
            sorted(missing)
        )
        return

    b["start"] = pd.to_datetime(
        b["start"],
        utc=True,
    )

    b["end"] = pd.to_datetime(
        b["end"],
        utc=True,
    )

    b[
        "duration_seconds"
    ] = (
        b["end"]
        - b["start"]
    ).dt.total_seconds()

    cache = {}

    rows = []

    sessions = sorted(
        b["session_id"].unique()
    )

    for i, session_id in enumerate(
        sessions,
        start=1,
    ):

        print(
            f"Scoring Dataset B "
            f"{i}/{len(sessions)}: "
            f"{session_id}"
        )

        # Dataset B has no gt_manifest, so locate its session folder
        # directly under a dataset_b subtree, with a deterministic
        # preference for the directory containing events.jsonl.
        matches = []

        for dataset_root in DATA_ROOT.rglob(
            "dataset_b"
        ):
            candidate = (
                dataset_root
                / session_id
            )

            if candidate.exists():
                matches.append(
                    candidate
                )

        if not matches:

            # Fallback: search for exact session name and keep only
            # directories with events.jsonl somewhere below.
            matches = [
                p
                for p in DATA_ROOT.rglob(
                    session_id
                )
                if p.is_dir()
                and any(
                    p.rglob(
                        "events.jsonl"
                    )
                )
                and "dataset_b" in {
                    part.lower()
                    for part in p.parts
                }
            ]

        if not matches:
            print(
                f"  WARNING: Dataset B session "
                f"not found: {session_id}"
            )
            continue

        if len(matches) > 1:
            print(
                f"  WARNING: multiple Dataset B "
                f"matches; using first deterministic "
                f"sorted path."
            )
            matches = sorted(
                matches
            )

        folder = matches[0]

        if session_id not in cache:
            # Inline loader because B has no manifest.
            events = []

            for event_file in sorted(
                folder.rglob(
                    "events.jsonl"
                )
            ):
                try:
                    with event_file.open(
                        "r",
                        encoding="utf-8",
                    ) as f:

                        for line in f:
                            line = line.strip()

                            if not line:
                                continue

                            try:
                                event = json.loads(
                                    line
                                )
                            except Exception:
                                continue

                            timestamp = event_timestamp(
                                event
                            )

                            if pd.isna(
                                timestamp
                            ):
                                continue

                            events.append(
                                {
                                    "timestamp":
                                        timestamp,
                                    "event":
                                        event,
                                }
                            )

                except OSError:
                    continue

            events.sort(
                key=lambda x:
                    x["timestamp"]
            )

            cache[
                session_id
            ] = events

        for _, segment in b[
            b[
                "session_id"
            ]
            == session_id
        ].iterrows():

            row = {
                "session_id":
                    session_id,
                "segment_index":
                    segment.get(
                        "segment_index",
                        -1,
                    ),
                "start":
                    segment[
                        "start"
                    ],
                "end":
                    segment[
                        "end"
                    ],
                "duration_seconds":
                    segment[
                        "duration_seconds"
                    ],
            }

            row.update(
                aggregate_segment_features(
                    cache[
                        session_id
                    ],
                    segment[
                        "start"
                    ],
                    segment[
                        "end"
                    ],
                )
            )

            rows.append(
                row
            )

    if not rows:
        print(
            "No Dataset B segments were scored."
        )
        return

    b_features = pd.DataFrame(
        rows
    )

    model = make_model()

    model.fit(
        a_df[
            FEATURES
        ],
        a_df[
            target_column
        ],
    )

    b_probability = (
        model.predict_proba(
            b_features[
                FEATURES
            ]
        )[:, 1]
    )

    b_features[
        "process_probability"
    ] = b_probability

    b_features[
        "suggested_status"
    ] = np.where(
        b_probability >= 0.50,
        "process_candidate",
        "other_candidate",
    )

    b_features[
        "high_confidence_other"
    ] = (
        b_probability < 0.20
    )

    b_features[
        "high_confidence_process"
    ] = (
        b_probability >= 0.80
    )

    b_features.to_csv(
        OUT_B,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 70
    )
    print(
        "DATASET B FILTER SCORE DISTRIBUTION"
    )
    print(
        "=" * 70
    )

    print(
        b_features[
            "process_probability"
        ].describe().to_string()
    )

    print()
    print(
        "Suggested status:"
    )

    print(
        b_features[
            "suggested_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "High-confidence process:"
        f" {int(b_features['high_confidence_process'].sum()):,}"
    )

    print(
        "High-confidence other:"
        f" {int(b_features['high_confidence_other'].sum()):,}"
    )

    print()
    print(
        f"Saved Dataset B scores:\n{OUT_B}"
    )


def main():
    print("=" * 70)
    print(
        "NON-PROCESS FILTER — DATASET A VALIDATION"
    )
    print("=" * 70)

    OUTPUTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    a_df = build_a_training_table()

    if a_df.empty:
        raise RuntimeError(
            "No Dataset A predicted segments built."
        )

    a_df.to_csv(
        OUT_A,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 70
    )
    print(
        "DATASET-A PROCESS/GAP TARGET DISTRIBUTION"
    )
    print(
        "=" * 70
    )

    for column in (
        "process_work_any_overlap",
        "process_work_majority_overlap",
    ):
        print()
        print(
            column
        )
        print(
            a_df[
                column
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print()
    print(
        "GT overlap fraction distribution:"
    )

    print(
        a_df[
            "gt_overlap_fraction"
        ].describe().to_string()
    )

    # --------------------------------------------------------
    # We use majority-overlap as the cleaner training target.
    # Any-overlap is still reported because it directly answers
    # the more permissive "does it touch process work?" question.
    # --------------------------------------------------------

    target = (
        "process_work_majority_overlap"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "LEAVE-ONE-SESSION-OUT VALIDATION"
    )
    print(
        "=" * 70
    )

    predictions = cross_validate(
        a_df,
        target,
    )

    predictions.to_csv(
        OUT_A_SCORES,
        index=False,
        encoding="utf-8-sig",
    )

    print_metrics(
        predictions,
        target,
    )

    print()
    print(
        f"Saved A CV predictions:\n{OUT_A_SCORES}"
    )

    # --------------------------------------------------------
    # Only after validation, score B.
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "SCORING DATASET B"
    )
    print(
        "=" * 70
    )

    score_dataset_b(
        a_df,
        target,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "NON-PROCESS FILTER COMPLETE"
    )
    print(
        "=" * 70
    )

    print(
        "This does NOT alter segments.jsonl yet."
    )

    print(
        "Review Dataset B filter scores before deciding "
        "which segments should become 'other'."
    )


if __name__ == "__main__":
    main()
