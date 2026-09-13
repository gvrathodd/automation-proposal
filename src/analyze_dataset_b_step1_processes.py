
from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from loader import load_all_events, sort_session_events, get_timestamp_ms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
SEGMENTS_FILE = PROJECT_ROOT / "outputs" / "dataset_b_segments.csv"
INVENTORY_FILE = PROJECT_ROOT / "outputs" / "dataset_b_segment_inventory.csv"
SELECTION_FILE = PROJECT_ROOT / "outputs" / "dataset_b_screenshot_selection_v2.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

SEGMENT_BEHAVIOR_FILE = (
    OUTPUT_DIR / "dataset_b_segment_behavior.csv"
)

PROCESS_FAMILIES_FILE = (
    OUTPUT_DIR / "dataset_b_process_family_candidates.csv"
)

CANDIDATE_OCCURRENCES_FILE = (
    OUTPUT_DIR / "dataset_b_process_family_occurrences.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Consecutive duplicate applications are collapsed because
# Edge -> Edge -> Edge does not add workflow information.
MAX_SEQUENCE_LEN = 12

# Application transitions that are especially useful for
# understanding cross-application data movement.
IMPORTANT_TRANSITIONS = {
    ("Microsoft Edge", "Notepad"),
    ("Notepad", "Microsoft Edge"),
    ("Microsoft Edge", "Microsoft Word"),
    ("Microsoft Word", "Microsoft Edge"),
    ("Microsoft Edge", "Microsoft Excel"),
    ("Microsoft Excel", "Microsoft Edge"),
}


# ============================================================
# HELPERS
# ============================================================

def normalize_app(value):
    if not isinstance(value, str):
        return "UNKNOWN"

    value = " ".join(value.split()).strip()

    return value if value else "UNKNOWN"


def event_type(event):
    value = event.get("event_type")

    return value if isinstance(value, str) else ""


def app_name(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}

    value = active_app.get("app_name")

    return normalize_app(value)


def browser_url(event):
    payload = event.get("payload") or {}
    context = event.get("context") or {}

    for value in (
        payload.get("url"),
        payload.get("current_url"),
        payload.get("target_url"),
        context.get("url"),
        context.get("current_url"),
        context.get("browser_url"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def browser_path(url):
    if not url:
        return ""

    try:
        return url.split("#/", 1)[1].split("?", 1)[0].strip("/")
    except IndexError:
        return ""


def extracted_text(event):
    context = event.get("context") or {}

    value = context.get("extracted_text")

    return value.strip() if isinstance(value, str) else ""


def clipboard_change(event_type_value):
    return event_type_value == "clipboard_change"


def keyboard_event(event_type_value):
    return event_type_value in {
        "keystroke",
        "text_input_complete",
    }


def collapse(values):
    output = []

    for value in values:
        if not value:
            continue

        if not output or output[-1] != value:
            output.append(value)

    return output


def truncate(values, limit=MAX_SEQUENCE_LEN):
    if len(values) <= limit:
        return values

    return (
        values[: limit // 2]
        + ["..."]
        + values[-(limit // 2):]
    )


# ============================================================
# RAW SESSION PREPARATION
# ============================================================

def prepare_events(events):
    prepared = []

    for event in sort_session_events(events):

        timestamp_ms = get_timestamp_ms(event)

        if timestamp_ms is None:
            continue

        prepared.append(
            {
                "timestamp_ms": int(timestamp_ms),
                "timestamp": pd.to_datetime(
                    int(timestamp_ms),
                    unit="ms",
                    utc=True,
                ),
                "event_type": event_type(event),
                "app": app_name(event),
                "url": browser_url(event),
                "browser_path": browser_path(
                    browser_url(event)
                ),
                "text": extracted_text(event),
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
# SEGMENT BEHAVIOR
# ============================================================

def analyze_segment(
    segment,
    prepared_events,
    screenshot_count=0,
):
    start = pd.Timestamp(
        segment["start"]
    )

    end = pd.Timestamp(
        segment["end"]
    )

    events = events_for_segment(
        prepared_events,
        start,
        end,
    )

    event_types = [
        x["event_type"]
        for x in events
        if x["event_type"]
    ]

    apps = [
        normalize_app(x["app"])
        for x in events
        if x["app"]
    ]

    apps_collapsed = truncate(
        collapse(apps)
    )

    transitions = []

    for left, right in zip(
        apps_collapsed,
        apps_collapsed[1:],
    ):
        if left == "...":
            continue
        if right == "...":
            continue

        transitions.append(
            (left, right)
        )

    important_transition_count = sum(
        transition in IMPORTANT_TRANSITIONS
        for transition in transitions
    )

    important_transition_names = [
        f"{left} -> {right}"
        for left, right in transitions
        if (
            left,
            right,
        ) in IMPORTANT_TRANSITIONS
    ]

    urls = [
        x["url"]
        for x in events
        if x["url"]
    ]

    browser_paths = [
        x["browser_path"]
        for x in events
        if x["browser_path"]
    ]

    unique_paths = list(
        dict.fromkeys(
            browser_paths
        )
    )

    unique_apps = list(
        dict.fromkeys(apps)
    )

    text_events = [
        x["text"]
        for x in events
        if x["text"]
    ]

    counts = Counter(
        event_types
    )

    # Indicators of data movement / editing.
    clipboard_count = counts[
        "clipboard_change"
    ]

    keyboard_count = sum(
        keyboard_event(x)
        for x in event_types
    )

    form_input_count = counts[
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

    # Directional data-transfer heuristic.
    #
    # This does NOT claim that a transfer actually occurred.
    # It identifies segments containing both app switching
    # and clipboard/form-input activity.
    data_transfer_signal = (
        (
            clipboard_count > 0
            and len(apps_collapsed) >= 2
        )
        or (
            form_input_count > 0
            and len(apps_collapsed) >= 2
        )
    )

    browser_paths_collapsed = truncate(
        collapse(browser_paths),
        12,
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
            float(segment["duration_seconds"]),

        "event_count":
            len(events),

        "unique_event_types":
            len(set(event_types)),

        "event_types":
            " | ".join(
                sorted(
                    set(event_types)
                )
            ),

        "app_sequence":
            " > ".join(
                apps_collapsed
            ),

        "unique_app_count":
            len(unique_apps),

        "apps":
            " | ".join(
                unique_apps
            ),

        "important_transition_count":
            important_transition_count,

        "important_transitions":
            " | ".join(
                important_transition_names
            ),

        "clipboard_change_count":
            clipboard_count,

        "keyboard_count":
            keyboard_count,

        "browser_form_input_count":
            form_input_count,

        "browser_click_count":
            browser_click_count,

        "mouse_click_count":
            mouse_click_count,

        "browser_navigation_count":
            navigation_count,

        "browser_paths":
            " | ".join(
                browser_paths_collapsed
            ),

        "unique_browser_path_count":
            len(
                set(unique_paths)
            ),

        "extracted_text_event_count":
            len(text_events),

        "screenshot_count":
            int(screenshot_count),

        "data_transfer_signal":
            data_transfer_signal,
    }


# ============================================================
# FAMILY KEY
# ============================================================

def build_family_key(row):
    """
    Deliberately coarse.

    We want repeated behavioral families, not exact clones.
    Exact click counts are NOT included.
    """

    sequence = row["app_sequence"]

    # Normalize long sequences around meaningful transitions.
    if "Microsoft Edge > Notepad > Microsoft Edge" in sequence:
        workflow_core = "EDGE-NOTEPAD-EDGE"
    elif "Microsoft Edge > Microsoft Word > Microsoft Edge" in sequence:
        workflow_core = "EDGE-WORD-EDGE"
    elif "Microsoft Edge > Microsoft Excel > Microsoft Edge" in sequence:
        workflow_core = "EDGE-EXCEL-EDGE"
    else:
        # For other workflows, use the collapsed sequence itself.
        workflow_core = sequence

    paths = row["browser_paths"]

    # Keep only a coarse page signature.
    if "payroll-items" in paths:
        page_group = "PAYROLL_ITEMS"
    elif "leave-applications" in paths:
        page_group = "LEAVE_APPLICATIONS"
    else:
        page_group = "OTHER_WEB_PAGE"

    if row["clipboard_change_count"] > 0:
        clipboard_group = "CLIPBOARD"
    else:
        clipboard_group = "NO_CLIPBOARD"

    if row["browser_form_input_count"] > 0:
        form_group = "FORM_INPUT"
    else:
        form_group = "NO_FORM_INPUT"

    return (
        f"{workflow_core}"
        f" | {page_group}"
        f" | {clipboard_group}"
        f" | {form_group}"
    )


# ============================================================
# FAMILY RANKING
# ============================================================

def build_family_table(
    behavior,
):
    df = behavior.copy()

    df["family_key"] = df.apply(
        build_family_key,
        axis=1,
    )

    grouped = (
        df.groupby("family_key")
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
            min_duration_seconds=(
                "duration_seconds",
                "min",
            ),
            max_duration_seconds=(
                "duration_seconds",
                "max",
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
            data_transfer_rate=(
                "data_transfer_signal",
                "mean",
            ),
            screenshot_coverage=(
                "screenshot_count",
                lambda x:
                    (x > 0).mean(),
            ),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        [
            "total_duration_seconds",
            "segment_count",
            "session_count",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    grouped["family_id"] = [
        f"PF{index:03d}"
        for index in range(
            1,
            len(grouped) + 1,
        )
    ]

    family_map = dict(
        zip(
            grouped["family_key"],
            grouped["family_id"],
        )
    )

    df["family_id"] = (
        df["family_key"]
        .map(family_map)
    )

    return df, grouped


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DATASET B STEP-1 SEGMENT -> PROCESS FAMILY ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{SEGMENTS_FILE}"
        )

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{INVENTORY_FILE}"
        )

    print()
    print("Loading Step-1 Dataset B segments...")

    segments = pd.read_csv(
        SEGMENTS_FILE,
        keep_default_na=False,
    )

    inventory = pd.read_csv(
        INVENTORY_FILE,
        keep_default_na=False,
    )

    segments["start"] = pd.to_datetime(
        segments["start"],
        utc=True,
    )

    segments["end"] = pd.to_datetime(
        segments["end"],
        utc=True,
    )

    inventory["start"] = pd.to_datetime(
        inventory["start"],
        utc=True,
    )

    inventory["end"] = pd.to_datetime(
        inventory["end"],
        utc=True,
    )

    print(
        f"Step-1 segments: "
        f"{len(segments):,}"
    )

    # Screenshot coverage.
    screenshot_counts = {}

    if SELECTION_FILE.exists():

        selection = pd.read_csv(
            SELECTION_FILE,
            keep_default_na=False,
        )

        screenshot_counts = (
            selection
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
            f"{len(selection):,}"
        )

    else:
        print(
            "WARNING: screenshot selection file not found. "
            "Continuing with screenshot_count=0."
        )

    # --------------------------------------------------------
    # Load raw Dataset B sessions.
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
    # Prepare only required sessions.
    # --------------------------------------------------------

    prepared_sessions = {}

    for session_id in segments[
        "session_id"
    ].unique():

        if session_id not in dataset_b_sessions:
            raise RuntimeError(
                f"Dataset B session missing: "
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
    # Build behavior table.
    # --------------------------------------------------------

    print()
    print(
        "Extracting behavior from "
        "Step-1-generated segments..."
    )

    records = []

    for _, segment in segments.iterrows():

        key = (
            segment["session_id"],
            int(segment["segment_index"]),
        )

        screenshot_count = screenshot_counts.get(
            key,
            0,
        )

        records.append(
            analyze_segment(
                segment,
                prepared_sessions[
                    segment["session_id"]
                ],
                screenshot_count,
            )
        )

    behavior = pd.DataFrame(
        records
    )

    print(
        f"Behavior rows: "
        f"{len(behavior):,}"
    )

    # --------------------------------------------------------
    # Build process-family hypotheses.
    # --------------------------------------------------------

    behavior, families = (
        build_family_table(
            behavior
        )
    )

    behavior.to_csv(
        SEGMENT_BEHAVIOR_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    families.to_csv(
        PROCESS_FAMILIES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Family occurrence file.
    # --------------------------------------------------------

    occurrence_columns = [
        "family_id",
        "family_key",
        "session_id",
        "segment_index",
        "start",
        "end",
        "duration_seconds",
        "app_sequence",
        "browser_paths",
        "clipboard_change_count",
        "browser_form_input_count",
        "browser_navigation_count",
        "data_transfer_signal",
        "screenshot_count",
    ]

    occurrences = (
        behavior[
            [
                "family_id",
                "family_key",
                "session_id",
                "segment_index",
                "start",
                "end",
                "duration_seconds",
                "app_sequence",
                "browser_paths",
                "clipboard_change_count",
                "browser_form_input_count",
                "browser_navigation_count",
                "data_transfer_signal",
                "screenshot_count",
            ]
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

    occurrences.to_csv(
        CANDIDATE_OCCURRENCES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Print top candidate families.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP PROCESS-FAMILY HYPOTHESES")
    print("=" * 70)

    display_columns = [
        "family_id",
        "segment_count",
        "session_count",
        "total_duration_seconds",
        "median_duration_seconds",
        "mean_clipboard_changes",
        "mean_form_inputs",
        "mean_navigations",
        "data_transfer_rate",
        "screenshot_coverage",
        "family_key",
    ]

    print(
        families[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("STRONG CROSS-APPLICATION FAMILIES")
    print("=" * 70)

    cross_app = families[
        families[
            "family_key"
        ].str.contains(
            "EDGE-"
            "|NOTEPAD"
            "|WORD"
            "|EXCEL",
            case=False,
            regex=True,
        )
    ]

    print(
        cross_app[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("STEP-1 SEGMENT -> PROCESS ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Segment behavior:\n"
        f"{SEGMENT_BEHAVIOR_FILE}"
    )

    print(
        f"\nProcess family hypotheses:\n"
        f"{PROCESS_FAMILIES_FILE}"
    )

    print(
        f"\nFamily occurrences:\n"
        f"{CANDIDATE_OCCURRENCES_FILE}"
    )


if __name__ == "__main__":
    main()
