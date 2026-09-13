
from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import re
import json

import numpy as np
import pandas as pd

from loader import load_all_events, sort_session_events, get_timestamp_ms


# ============================================================
# STEP 2 — DATASET B PROCESS DISCOVERY
# ============================================================
#
# This script starts FROM THE STEP-1 SEGMENTS.
#
# It does NOT perform segmentation.
# It does NOT use Dataset A ground truth.
# It does NOT assume that an app sequence is a business process.
#
# Goal:
#   1. Enrich each Step-1 segment with raw-event behavior.
#   2. Build reusable behavioral signatures.
#   3. Discover recurring candidate process families.
#   4. Rank families for manual semantic validation.
#
# The family grouping is deliberately conservative:
#   - business page/path
#   - coarse application workflow
#   - interaction pattern
#   - clipboard/form behavior
#
# These are HYPOTHESES, not final process labels.
# Screenshots are then used to validate the strongest families.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "raw_data"
    / "dataset-downloads"
)

SEGMENTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_segments.csv"
)

SCREENSHOT_SELECTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_screenshot_selection_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

BEHAVIOR_FILE = (
    OUTPUT_DIR
    / "step2_dataset_b_segment_behavior.csv"
)

FAMILY_FILE = (
    OUTPUT_DIR
    / "step2_dataset_b_process_families.csv"
)

OCCURRENCE_FILE = (
    OUTPUT_DIR
    / "step2_dataset_b_family_occurrences.csv"
)

TRANSITION_FILE = (
    OUTPUT_DIR
    / "step2_dataset_b_transition_patterns.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "step2_dataset_b_validation_queue.csv"
)


# ============================================================
# Configuration
# ============================================================

# Small duration buckets are more useful for discovery than
# exact duration matching.
DURATION_BINS = [
    0,
    10,
    20,
    40,
    60,
    120,
    300,
    np.inf,
]

DURATION_LABELS = [
    "<10s",
    "10-20s",
    "20-40s",
    "40-60s",
    "1-2m",
    "2-5m",
    "5m+",
]


# Only use the first / last few apps for a compact sequence.
MAX_APP_SEQUENCE_LENGTH = 8


# ============================================================
# Helpers
# ============================================================

def normalize_app(value):
    if not isinstance(value, str):
        return "UNKNOWN"

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or "UNKNOWN"


def get_event_type(event):
    value = event.get("event_type")

    return value if isinstance(value, str) else ""


def get_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}

    return normalize_app(
        active_app.get("app_name")
    )


def get_url(event):
    payload = event.get("payload") or {}
    context = event.get("context") or {}

    for value in [
        payload.get("url"),
        payload.get("current_url"),
        payload.get("target_url"),
        context.get("url"),
        context.get("current_url"),
        context.get("browser_url"),
    ]:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def browser_path(url):
    if not url:
        return ""

    try:
        value = url.split(
            "#/",
            1,
        )[1]
    except IndexError:
        value = url

    value = value.split(
        "?",
        1,
    )[0]

    value = value.split(
        "#",
        1,
    )[0]

    return value.strip("/")


def get_extracted_text(event):
    context = event.get("context") or {}

    value = context.get(
        "extracted_text"
    )

    if isinstance(value, str):
        return value.strip()

    return ""


def collapse_sequence(values):
    output = []

    for value in values:

        if not value:
            continue

        if (
            not output
            or output[-1] != value
        ):
            output.append(value)

    return output


def truncate_sequence(values, maximum):
    if len(values) <= maximum:
        return values

    left = maximum // 2
    right = maximum - left

    return (
        values[:left]
        + ["..."]
        + values[-right:]
    )


def safe_mode(series):
    modes = series.mode()

    if modes.empty:
        return "UNKNOWN"

    return modes.iloc[0]


def duration_bucket(seconds):
    try:
        value = float(seconds)
    except (
        TypeError,
        ValueError,
    ):
        return "UNKNOWN"

    for index in range(
        len(DURATION_BINS) - 1
    ):
        if (
            DURATION_BINS[index]
            <= value
            < DURATION_BINS[index + 1]
        ):
            return DURATION_LABELS[index]

    return DURATION_LABELS[-1]


# ============================================================
# Raw events
# ============================================================

def prepare_events(events):
    prepared = []

    for event in sort_session_events(events):

        timestamp_ms = get_timestamp_ms(
            event
        )

        if timestamp_ms is None:
            continue

        url = get_url(event)

        prepared.append(
            {
                "timestamp_ms":
                    int(timestamp_ms),

                "timestamp":
                    pd.to_datetime(
                        int(timestamp_ms),
                        unit="ms",
                        utc=True,
                    ),

                "event_type":
                    get_event_type(event),

                "app":
                    get_app(event),

                "url":
                    url,

                "browser_path":
                    browser_path(url),

                "text":
                    get_extracted_text(event),
            }
        )

    return prepared


def events_for_segment(
    prepared_events,
    start,
    end,
):
    start_ms = int(
        start.timestamp() * 1000
    )

    end_ms = int(
        end.timestamp() * 1000
    )

    return [
        event
        for event in prepared_events
        if (
            start_ms
            <= event["timestamp_ms"]
            <= end_ms
        )
    ]


# ============================================================
# Segment behavioral representation
# ============================================================

def build_segment_behavior(
    segment,
    events,
    screenshot_count,
):
    start = pd.Timestamp(
        segment["start"]
    )

    end = pd.Timestamp(
        segment["end"]
    )

    duration = float(
        segment["duration_seconds"]
    )

    event_types = [
        event["event_type"]
        for event in events
        if event["event_type"]
    ]

    apps = [
        event["app"]
        for event in events
        if event["app"] != "UNKNOWN"
    ]

    collapsed_apps = collapse_sequence(
        apps
    )

    collapsed_apps = truncate_sequence(
        collapsed_apps,
        MAX_APP_SEQUENCE_LENGTH,
    )

    transitions = []

    for left, right in zip(
        collapsed_apps,
        collapsed_apps[1:],
    ):
        if (
            left == "..."
            or right == "..."
        ):
            continue

        transitions.append(
            (
                left,
                right,
            )
        )

    transition_strings = [
        f"{left} -> {right}"
        for left, right in transitions
    ]

    counts = Counter(
        event_types
    )

    paths = [
        event["browser_path"]
        for event in events
        if event["browser_path"]
    ]

    unique_paths = list(
        dict.fromkeys(paths)
    )

    texts = [
        event["text"]
        for event in events
        if event["text"]
    ]

    unique_apps = list(
        dict.fromkeys(apps)
    )

    # --------------------------------------------------------
    # Interaction indicators
    # --------------------------------------------------------

    clipboard_count = counts[
        "clipboard_change"
    ]

    keyboard_count = (
        counts["keystroke"]
        + counts["text_input_complete"]
    )

    browser_form_count = counts[
        "browser_form_input"
    ]

    browser_click_count = counts[
        "browser_click"
    ]

    mouse_click_count = counts[
        "mouse_click"
    ]

    navigation_count = counts[
        "browser_navigation"
    ]

    screenshot_flag = (
        screenshot_count > 0
    )

    # --------------------------------------------------------
    # Coarse business-page class
    # --------------------------------------------------------

    page_classes = []

    for path in unique_paths:

        lower = path.lower()

        if "payroll-items" in lower:
            page_classes.append(
                "PAYROLL_ITEMS"
            )

        elif "leave-applications" in lower:
            page_classes.append(
                "LEAVE_APPLICATIONS"
            )

        elif (
            "employee" in lower
            or "staff" in lower
        ):
            page_classes.append(
                "EMPLOYEE"
            )

        else:
            page_classes.append(
                "OTHER_WEB_PAGE"
            )

    page_classes = list(
        dict.fromkeys(page_classes)
    )

    if not page_classes:
        page_class = "NO_PAGE"
    elif len(page_classes) == 1:
        page_class = page_classes[0]
    else:
        page_class = "MULTI_PAGE"

    # --------------------------------------------------------
    # Behavioral classes
    # --------------------------------------------------------

    has_cross_app = (
        len(collapsed_apps) >= 2
    )

    has_clipboard = (
        clipboard_count > 0
    )

    has_form_input = (
        browser_form_count > 0
    )

    has_navigation = (
        navigation_count > 0
    )

    if (
        has_clipboard
        and has_form_input
        and has_cross_app
    ):
        transfer_class = (
            "CROSS_APP_DATA_TRANSFER"
        )

    elif (
        has_clipboard
        and has_cross_app
    ):
        transfer_class = (
            "CROSS_APP_CLIPBOARD"
        )

    elif has_form_input:
        transfer_class = "FORM_INTERACTION"

    elif has_navigation:
        transfer_class = "NAVIGATION"

    elif has_cross_app:
        transfer_class = (
            "CROSS_APP_OTHER"
        )

    else:
        transfer_class = (
            "SINGLE_APP_ACTIVITY"
        )

    return {
        "session_id":
            segment["session_id"],

        "segment_index":
            int(segment["segment_index"]),

        "start":
            start,

        "end":
            end,

        "duration_seconds":
            duration,

        "duration_bucket":
            duration_bucket(duration),

        "event_count":
            len(events),

        "unique_event_type_count":
            len(set(event_types)),

        "event_types":
            " | ".join(
                sorted(
                    set(event_types)
                )
            ),

        "app_sequence":
            " > ".join(
                collapsed_apps
            ),

        "unique_app_count":
            len(unique_apps),

        "apps":
            " | ".join(unique_apps),

        "transition_sequence":
            " | ".join(
                transition_strings
            ),

        "important_transition_count":
            sum(
                transition
                in {
                    (
                        "Microsoft Edge",
                        "Notepad",
                    ),
                    (
                        "Notepad",
                        "Microsoft Edge",
                    ),
                    (
                        "Microsoft Edge",
                        "Microsoft Word",
                    ),
                    (
                        "Microsoft Word",
                        "Microsoft Edge",
                    ),
                    (
                        "Microsoft Edge",
                        "Microsoft Excel",
                    ),
                    (
                        "Microsoft Excel",
                        "Microsoft Edge",
                    ),
                }
                for transition in transitions
            ),

        "clipboard_change_count":
            clipboard_count,

        "keyboard_count":
            keyboard_count,

        "browser_form_input_count":
            browser_form_count,

        "browser_click_count":
            browser_click_count,

        "mouse_click_count":
            mouse_click_count,

        "browser_navigation_count":
            navigation_count,

        "browser_paths":
            " | ".join(
                unique_paths[:12]
            ),

        "unique_browser_path_count":
            len(
                set(unique_paths)
            ),

        "page_class":
            page_class,

        "transfer_class":
            transfer_class,

        "extracted_text_event_count":
            len(texts),

        "screenshot_count":
            screenshot_count,

        "screenshot_available":
            screenshot_flag,
    }


# ============================================================
# Process-family signature
# ============================================================

def family_signature(row):
    """
    Coarse signature.

    The intent is to group likely variants of the same workflow,
    not create one family per exact event sequence.
    """

    sequence = row["app_sequence"]

    # Normalize common cross-application workflow shapes.
    if (
        "Microsoft Edge"
        in sequence
        and "Microsoft Word"
        in sequence
    ):
        workflow = "EDGE_WORD_WORKFLOW"

    elif (
        "Microsoft Edge"
        in sequence
        and "Notepad"
        in sequence
    ):
        workflow = "EDGE_NOTEPAD_WORKFLOW"

    elif (
        "Microsoft Edge"
        in sequence
        and "Microsoft Excel"
        in sequence
    ):
        workflow = "EDGE_EXCEL_WORKFLOW"

    elif (
        row["unique_app_count"] >= 2
    ):
        workflow = "OTHER_CROSS_APP"

    else:
        workflow = "EDGE_ONLY_OR_SINGLE_APP"

    return (
        f"{workflow}"
        f" | {row['page_class']}"
        f" | {row['transfer_class']}"
        f" | {row['duration_bucket']}"
    )


# ============================================================
# Transition-pattern analysis
# ============================================================

def build_transition_table(
    behavior,
):
    rows = []

    for _, row in behavior.iterrows():

        transitions = (
            row[
                "transition_sequence"
            ]
            .split(" | ")
        )

        if not transitions:
            continue

        for transition in transitions:

            if " -> " not in transition:
                continue

            left, right = transition.split(
                " -> ",
                1,
            )

            rows.append(
                {
                    "session_id":
                        row["session_id"],

                    "segment_index":
                        row["segment_index"],

                    "duration_seconds":
                        row["duration_seconds"],

                    "page_class":
                        row["page_class"],

                    "transfer_class":
                        row["transfer_class"],

                    "from_app":
                        left,

                    "to_app":
                        right,

                    "clipboard_change_count":
                        row[
                            "clipboard_change_count"
                        ],

                    "browser_form_input_count":
                        row[
                            "browser_form_input_count"
                        ],

                    "browser_navigation_count":
                        row[
                            "browser_navigation_count"
                        ],

                    "screenshot_count":
                        row[
                            "screenshot_count"
                        ],
                }
            )

    transition_df = pd.DataFrame(
        rows
    )

    if transition_df.empty:
        return transition_df

    return (
        transition_df
        .groupby(
            [
                "from_app",
                "to_app",
            ]
        )
        .agg(
            occurrence_count=(
                "segment_index",
                "size",
            ),
            session_count=(
                "session_id",
                "nunique",
            ),
            total_duration_seconds=(
                "duration_seconds",
                "sum",
            ),
            median_duration_seconds=(
                "duration_seconds",
                "median",
            ),
            mean_clipboard_changes=(
                "clipboard_change_count",
                "mean",
            ),
            mean_form_inputs=(
                "browser_form_input_count",
                "mean",
            ),
            mean_navigations=(
                "browser_navigation_count",
                "mean",
            ),
            mean_screenshot_count=(
                "screenshot_count",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "occurrence_count",
                "total_duration_seconds",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# Family ranking
# ============================================================

def build_family_table(
    behavior,
):
    df = behavior.copy()

    df[
        "family_signature"
    ] = df.apply(
        family_signature,
        axis=1,
    )

    family = (
        df.groupby(
            "family_signature"
        )
        .agg(
            segment_count=(
                "segment_index",
                "size",
            ),
            session_count=(
                "session_id",
                "nunique",
            ),
            total_duration_seconds=(
                "duration_seconds",
                "sum",
            ),
            median_duration_seconds=(
                "duration_seconds",
                "median",
            ),
            mean_duration_seconds=(
                "duration_seconds",
                "mean",
            ),
            total_event_count=(
                "event_count",
                "sum",
            ),
            mean_clipboard_changes=(
                "clipboard_change_count",
                "mean",
            ),
            mean_form_inputs=(
                "browser_form_input_count",
                "mean",
            ),
            mean_navigations=(
                "browser_navigation_count",
                "mean",
            ),
            mean_screenshots=(
                "screenshot_count",
                "mean",
            ),
            screenshot_coverage=(
                "screenshot_available",
                "mean",
            ),
            transfer_rate=(
                "transfer_class",
                lambda x:
                    (
                        x
                        == "CROSS_APP_DATA_TRANSFER"
                    ).mean(),
            ),
            dominant_page=(
                "page_class",
                safe_mode,
            ),
            dominant_transfer_class=(
                "transfer_class",
                safe_mode,
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Candidate-priority score
    #
    # NOT a claim of ROI.
    # It is only a screening score for what to inspect next.
    # --------------------------------------------------------

    family["frequency_score"] = np.log1p(
        family["segment_count"]
    )

    family["coverage_score"] = np.log1p(
        family["session_count"]
    )

    family["time_score"] = np.log1p(
        family[
            "total_duration_seconds"
        ]
    )

    family["transfer_score"] = (
        family["transfer_rate"]
    )

    family["screening_score"] = (
        family["frequency_score"]
        * family["coverage_score"]
        * (
            1.0
            + family["transfer_score"]
        )
        * (
            1.0
            + 0.15
            * np.log1p(
                family[
                    "total_duration_seconds"
                ]
            )
        )
    )

    family = family.sort_values(
        [
            "screening_score",
            "segment_count",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    family["family_id"] = [
        f"PF{index:03d}"
        for index in range(
            1,
            len(family) + 1,
        )
    ]

    family_map = dict(
        zip(
            family[
                "family_signature"
            ],
            family["family_id"],
        )
    )

    df["family_id"] = (
        df["family_signature"]
        .map(family_map)
    )

    family[
        "rank"
    ] = np.arange(
        1,
        len(family) + 1,
    )

    return df, family


# ============================================================
# Manual validation queue
# ============================================================

def build_validation_queue(
    behavior,
    family,
):
    """
    Select a few representative segments per promising family.

    Preference:
      - different sessions
      - longer segments
      - screenshot availability
      - clipboard/form activity
    """

    ranked_family_ids = (
        family[
            "family_id"
        ]
        .head(10)
        .tolist()
    )

    rows = []

    for family_id in ranked_family_ids:

        subset = behavior[
            behavior["family_id"]
            == family_id
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(
            [
                "screenshot_available",
                "clipboard_change_count",
                "browser_form_input_count",
                "duration_seconds",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )

        chosen = []
        sessions_seen = set()

        # Prefer one sample from different sessions.
        for _, row in subset.iterrows():

            session_id = row[
                "session_id"
            ]

            if session_id in sessions_seen:
                continue

            chosen.append(row)
            sessions_seen.add(
                session_id
            )

            if len(chosen) >= 3:
                break

        for sample_number, row in enumerate(
            chosen,
            start=1,
        ):
            rows.append(
                {
                    "family_id":
                        family_id,

                    "family_rank":
                        int(
                            family[
                                family["family_id"]
                                == family_id
                            ]["rank"].iloc[0]
                        ),

                    "sample_number":
                        sample_number,

                    "session_id":
                        row["session_id"],

                    "segment_index":
                        int(
                            row["segment_index"]
                        ),

                    "start":
                        row["start"],

                    "end":
                        row["end"],

                    "duration_seconds":
                        row[
                            "duration_seconds"
                        ],

                    "app_sequence":
                        row["app_sequence"],

                    "page_class":
                        row["page_class"],

                    "browser_paths":
                        row["browser_paths"],

                    "clipboard_change_count":
                        row[
                            "clipboard_change_count"
                        ],

                    "browser_form_input_count":
                        row[
                            "browser_form_input_count"
                        ],

                    "browser_navigation_count":
                        row[
                            "browser_navigation_count"
                        ],

                    "screenshot_count":
                        row["screenshot_count"],

                    "transfer_class":
                        row["transfer_class"],

                    "validation_status":
                        "NOT_YET_VALIDATED",
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 2 — DATASET B PROCESS DISCOVERY")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load Step-1 segments.
    # --------------------------------------------------------

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Step-1 output:\n"
            f"{SEGMENTS_FILE}"
        )

    segments = pd.read_csv(
        SEGMENTS_FILE,
        keep_default_na=False,
    )

    required = {
        "session_id",
        "segment_index",
        "start",
        "end",
        "duration_seconds",
    }

    missing = required - set(
        segments.columns
    )

    if missing:
        raise RuntimeError(
            "Missing columns in Step-1 segments:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    segments["segment_index"] = (
        pd.to_numeric(
            segments["segment_index"],
            errors="raise",
        ).astype(int)
    )

    segments["start"] = pd.to_datetime(
        segments["start"],
        utc=True,
    )

    segments["end"] = pd.to_datetime(
        segments["end"],
        utc=True,
    )

    print(
        f"Step-1 segments loaded: "
        f"{len(segments):,}"
    )

    # --------------------------------------------------------
    # Load screenshot-selection coverage.
    # --------------------------------------------------------

    screenshot_counts = {}

    if SCREENSHOT_SELECTION_FILE.exists():

        screenshot_selection = pd.read_csv(
            SCREENSHOT_SELECTION_FILE,
            keep_default_na=False,
        )

        screenshot_counts = (
            screenshot_selection
            .groupby(
                [
                    "session_id",
                    "segment_index",
                ]
            )
            .size()
            .to_dict()
        )

        print(
            f"Screenshot-selection rows: "
            f"{len(screenshot_selection):,}"
        )

    else:
        print(
            "WARNING: screenshot selection file not found."
        )

    # --------------------------------------------------------
    # Load Dataset B raw sessions.
    # --------------------------------------------------------

    print()
    print("Loading raw Dataset B sessions...")

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

    # --------------------------------------------------------
    # Prepare raw sessions.
    # --------------------------------------------------------

    prepared_sessions = {}

    for session_id in segments[
        "session_id"
    ].unique():

        if session_id not in dataset_b_sessions:
            raise RuntimeError(
                f"Raw Dataset B session missing: "
                f"{session_id}"
            )

        prepared_sessions[
            session_id
        ] = prepare_events(
            dataset_b_sessions[
                session_id
            ]
        )

    # --------------------------------------------------------
    # Build behavior rows.
    # --------------------------------------------------------

    print()
    print(
        "Extracting behavior from Step-1 segments..."
    )

    behavior_rows = []

    for _, segment in segments.iterrows():

        key = (
            segment["session_id"],
            int(segment["segment_index"]),
        )

        screenshot_count = screenshot_counts.get(
            key,
            0,
        )

        events = events_for_segment(
            prepared_sessions[
                segment["session_id"]
            ],
            segment["start"],
            segment["end"],
        )

        behavior_rows.append(
            build_segment_behavior(
                segment,
                events,
                screenshot_count,
            )
        )

    behavior = pd.DataFrame(
        behavior_rows
    )

    print(
        f"Behavior rows: "
        f"{len(behavior):,}"
    )

    # --------------------------------------------------------
    # Build families.
    # --------------------------------------------------------

    behavior, family = build_family_table(
        behavior
    )

    # --------------------------------------------------------
    # Transition patterns.
    # --------------------------------------------------------

    transitions = build_transition_table(
        behavior
    )

    # --------------------------------------------------------
    # Family occurrences.
    # --------------------------------------------------------

    occurrence_columns = [
        "family_id",
        "family_signature",
        "session_id",
        "segment_index",
        "start",
        "end",
        "duration_seconds",
        "app_sequence",
        "page_class",
        "browser_paths",
        "clipboard_change_count",
        "browser_form_input_count",
        "browser_navigation_count",
        "transfer_class",
        "screenshot_count",
    ]

    occurrences = (
        behavior[
            occurrence_columns
        ]
        .sort_values(
            [
                "family_id",
                "session_id",
                "segment_index",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validation queue.
    # --------------------------------------------------------

    validation_queue = (
        build_validation_queue(
            behavior,
            family,
        )
    )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------

    behavior.to_csv(
        BEHAVIOR_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    family.to_csv(
        FAMILY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    occurrences.to_csv(
        OCCURRENCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    transitions.to_csv(
        TRANSITION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    validation_queue.to_csv(
        VALIDATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Print summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP PROCESS-FAMILY HYPOTHESES")
    print("=" * 70)

    display_columns = [
        "family_id",
        "rank",
        "segment_count",
        "session_count",
        "total_duration_seconds",
        "median_duration_seconds",
        "mean_clipboard_changes",
        "mean_form_inputs",
        "mean_navigations",
        "transfer_rate",
        "dominant_page",
        "dominant_transfer_class",
        "screening_score",
        "family_signature",
    ]

    print(
        family[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("TOP APPLICATION TRANSITIONS")
    print("=" * 70)

    if transitions.empty:
        print("No transitions found.")
    else:
        print(
            transitions
            .head(20)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("MANUAL VALIDATION QUEUE")
    print("=" * 70)

    if validation_queue.empty:
        print(
            "No validation candidates generated."
        )
    else:
        print(
            validation_queue[
                [
                    "family_id",
                    "family_rank",
                    "sample_number",
                    "session_id",
                    "segment_index",
                    "duration_seconds",
                    "app_sequence",
                    "page_class",
                    "browser_paths",
                    "clipboard_change_count",
                    "browser_form_input_count",
                    "screenshot_count",
                    "transfer_class",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("STEP 2 PROCESS DISCOVERY COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Segment behavior:\n"
        f"{BEHAVIOR_FILE}"
    )

    print(
        f"\nProcess families:\n"
        f"{FAMILY_FILE}"
    )

    print(
        f"\nFamily occurrences:\n"
        f"{OCCURRENCE_FILE}"
    )

    print(
        f"\nTransition patterns:\n"
        f"{TRANSITION_FILE}"
    )

    print(
        f"\nValidation queue:\n"
        f"{VALIDATION_FILE}"
    )


if __name__ == "__main__":
    main()
