
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import math
import shutil

import pandas as pd

from loader import (
    load_all_events,
    sort_session_events,
    get_timestamp_ms,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "raw_data"
    / "dataset-downloads"
)

SEGMENT_INVENTORY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_segment_inventory.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

SELECTION_FILE = (
    OUTPUT_DIR
    / "dataset_b_screenshot_selection.csv"
)

SELECTED_DIR = (
    OUTPUT_DIR
    / "dataset_b_selected_screenshots"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum number of screenshots selected from one segment.
MAX_SELECTED_PER_SEGMENT = 5

# Small temporal neighborhood used to identify an event-driven
# screenshot as "near" a transition/interaction.
EVENT_NEIGHBOR_SECONDS = 1.5

# Screenshots too close to a previously selected screenshot
# are suppressed so we do not select visually redundant images.
MIN_SCREENSHOT_SEPARATION_SECONDS = 1.0

# Event types that indicate a potentially meaningful state change.
HIGH_INFORMATION_EVENTS = {
    "app_switch",
    "browser_navigation",
    "browser_form_input",
    "browser_click",
    "clipboard_change",
    "mouse_click",
    "shortcut",
    "text_input_complete",
    "window_title_change",
    "window_state_change",
    "dialog_opened",
    "dialog_closed",
    "upload_started",
    "upload_completed",
}

# Event types that are useful but lower priority.
MEDIUM_INFORMATION_EVENTS = {
    "keystroke",
    "mouse_scroll",
    "browser_alert",
    "browser_tab_event",
}

# Preferred application states for coverage.
# This is not used as a semantic label; it simply prevents
# selecting all screenshots from the same application.
MAX_PER_APP = 3


# ============================================================
# BASIC HELPERS
# ============================================================

def get_event_type(event):
    value = event.get("event_type")
    return value if isinstance(value, str) else ""


def get_event_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}

    value = active_app.get("app_name")

    if isinstance(value, str) and value.strip():
        return value.strip()

    return "UNKNOWN"


def get_timestamp_ms(event):
    value = get_timestamp_ms_original(event)
    return value


def get_timestamp_ms_original(event):
    value = event.get("timestamp_ms")

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_screenshot_filenames(event):
    payload = event.get("payload") or {}
    reference = payload.get("file_reference")

    if not reference:
        return []

    output = []

    if isinstance(reference, dict):
        value = reference.get("filename")

        if isinstance(value, str) and value.strip():
            output.append(value.strip())

    elif isinstance(reference, list):
        for item in reference:
            if not isinstance(item, dict):
                continue

            value = item.get("filename")

            if isinstance(value, str) and value.strip():
                output.append(value.strip())

    return output


def to_utc_timestamp(value):
    return pd.to_datetime(
        value,
        utc=True,
    )


def timestamp_ms_to_timestamp(value):
    return pd.to_datetime(
        int(value),
        unit="ms",
        utc=True,
    )


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ============================================================
# LOAD + PREPARE RAW EVENTS
# ============================================================

def prepare_events(events):
    prepared = []

    for event in sort_session_events(events):

        timestamp_ms = get_timestamp_ms_original(
            event
        )

        if timestamp_ms is None:
            continue

        screenshots = get_screenshot_filenames(
            event
        )

        prepared.append(
            {
                "timestamp_ms":
                    int(timestamp_ms),

                "timestamp":
                    timestamp_ms_to_timestamp(
                        timestamp_ms
                    ),

                "event_type":
                    get_event_type(event),

                "app":
                    get_event_app(event),

                "screenshots":
                    screenshots,
            }
        )

    return prepared


# ============================================================
# SCREENSHOT INDEX
# ============================================================

def build_screenshot_index():
    print("Indexing screenshot files...")

    index = defaultdict(list)

    for path in DATA_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            continue

        index[path.name].append(path)

    print(
        f"Indexed unique screenshot filenames: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# EVENT INFORMATION SCORE
# ============================================================

def event_information_score(event_type):
    if event_type in HIGH_INFORMATION_EVENTS:
        return 5.0

    if event_type in MEDIUM_INFORMATION_EVENTS:
        return 2.0

    if event_type == "screenshot_smart":
        return 1.0

    return 0.0


# ============================================================
# FIND SEGMENT SCREENSHOTS
# ============================================================

def collect_segment_screenshots(
    prepared_events,
    start,
    end,
):
    """
    Return unique screenshots referenced by events in and
    immediately around the segment.

    Each screenshot keeps:
      - timestamp
      - app
      - source event
      - nearby high-information event score
    """

    margin = pd.Timedelta(
        seconds=EVENT_NEIGHBOR_SECONDS
    )

    window_start = start - margin
    window_end = end + margin

    records = {}

    for event in prepared_events:

        timestamp = event["timestamp"]

        if not (
            window_start
            <= timestamp
            <= window_end
        ):
            continue

        for filename in event["screenshots"]:

            # The same filename can be referenced more than
            # once. Keep the earliest occurrence.
            if filename in records:
                continue

            local_score = 1.0

            # We need to know whether another meaningful event
            # happened immediately before this screenshot.
            for other in prepared_events:

                other_timestamp = other["timestamp"]

                distance = abs(
                    (
                        timestamp
                        - other_timestamp
                    ).total_seconds()
                )

                if (
                    distance
                    <= EVENT_NEIGHBOR_SECONDS
                ):
                    local_score += (
                        event_information_score(
                            other["event_type"]
                        )
                    )

            records[filename] = {
                "filename":
                    filename,

                "timestamp":
                    timestamp,

                "app":
                    event["app"],

                "source_event_type":
                    event["event_type"],

                "local_information_score":
                    local_score,
            }

    return list(
        records.values()
    )


# ============================================================
# SELECT TRANSITION-ANCHORED SCREENSHOTS
# ============================================================

def select_screenshots_for_segment(
    candidates,
    start,
    end,
):
    """
    Deterministic, auditable ranking.

    Score components:
      1. high-information event proximity
      2. app-transition coverage
      3. position within segment
      4. temporal spacing
      5. redundancy penalty

    The selected records retain the reason.
    """

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda item: item["timestamp"]
    )

    # --------------------------------------------------------
    # Identify application transitions.
    # --------------------------------------------------------

    transitions = []

    previous_app = None

    for candidate in candidates:

        app = candidate["app"]

        if (
            previous_app is not None
            and app != previous_app
        ):
            transitions.append(
                {
                    "timestamp":
                        candidate["timestamp"],
                    "from_app":
                        previous_app,
                    "to_app":
                        app,
                }
            )

        previous_app = app

    midpoint = (
        start
        + (end - start) / 2
    )

    # --------------------------------------------------------
    # Score each candidate.
    # --------------------------------------------------------

    scored = []

    for candidate in candidates:

        timestamp = candidate["timestamp"]

        score = (
            candidate[
                "local_information_score"
            ]
        )

        reasons = []

        # Strong score for being close to a transition.
        nearest_transition = None
        nearest_transition_distance = None

        for transition in transitions:

            distance = abs(
                (
                    timestamp
                    - transition["timestamp"]
                ).total_seconds()
            )

            if (
                nearest_transition_distance
                is None
                or distance
                < nearest_transition_distance
            ):
                nearest_transition = transition
                nearest_transition_distance = distance

        if (
            nearest_transition is not None
            and nearest_transition_distance
            <= EVENT_NEIGHBOR_SECONDS
        ):
            score += 8.0

            reasons.append(
                "near app transition "
                f"{nearest_transition['from_app']} "
                f"→ {nearest_transition['to_app']}"
            )

        # High-information source event.
        event_type = candidate[
            "source_event_type"
        ]

        if event_type in HIGH_INFORMATION_EVENTS:
            score += 5.0

            reasons.append(
                f"triggered by {event_type}"
            )

        elif event_type in MEDIUM_INFORMATION_EVENTS:
            score += 2.0

            reasons.append(
                f"triggered by {event_type}"
            )

        # Segment position coverage.
        duration = (
            end - start
        ).total_seconds()

        if duration > 0:

            relative_position = (
                timestamp - start
            ).total_seconds() / duration

            if (
                0.15
                <= relative_position
                <= 0.35
            ):
                score += 2.0
                reasons.append(
                    "early-segment coverage"
                )

            elif (
                0.35
                < relative_position
                < 0.65
            ):
                score += 1.5
                reasons.append(
                    "mid-segment coverage"
                )

            elif (
                0.65
                <= relative_position
                <= 0.90
            ):
                score += 2.0
                reasons.append(
                    "late-segment coverage"
                )

        # Prefer screenshots near the actual segment interior
        # over screenshots from the 2-second margin.
        if start <= timestamp <= end:
            score += 1.0
            reasons.append(
                "inside segment"
            )
        else:
            reasons.append(
                "context-margin screenshot"
            )

        # Transition fallback if no semantic event exists.
        if (
            not reasons
            or len(reasons) == 1
        ):
            distance_from_midpoint = abs(
                (
                    timestamp
                    - midpoint
                ).total_seconds()
            )

            score += max(
                0.0,
                1.0
                - (
                    distance_from_midpoint
                    / max(duration / 2, 1.0)
                ),
            )

        scored.append(
            {
                **candidate,
                "score":
                    score,
                "reason":
                    "; ".join(
                        reasons
                    ),
                "nearest_transition":
                    (
                        (
                            nearest_transition[
                                "from_app"
                            ]
                            + " → "
                            + nearest_transition[
                                "to_app"
                            ]
                        )
                        if nearest_transition
                        else ""
                    ),
            }
        )

    # --------------------------------------------------------
    # Greedy selection with diversity constraints.
    # --------------------------------------------------------

    selected = []
    app_counts = defaultdict(int)

    # First force a temporal anchor near the start/middle/end
    # when possible. We still use the same scoring system.
    anchors = [
        ("START", start),
        (
            "MIDDLE",
            midpoint,
        ),
        ("END", end),
    ]

    for anchor_name, anchor_time in anchors:

        eligible = []

        for candidate in scored:

            if candidate in selected:
                continue

            distance = abs(
                (
                    candidate["timestamp"]
                    - anchor_time
                ).total_seconds()
            )

            eligible.append(
                (
                    distance,
                    candidate,
                )
            )

        if not eligible:
            continue

        eligible.sort(
            key=lambda item: (
                item[0],
                -item[1]["score"],
            )
        )

        best = eligible[0][1]

        # Apply separation check.
        too_close = any(
            abs(
                (
                    best["timestamp"]
                    - chosen["timestamp"]
                ).total_seconds()
            )
            < MIN_SCREENSHOT_SEPARATION_SECONDS
            for chosen in selected
        )

        if too_close:
            continue

        selected.append(best)
        app_counts[
            best["app"]
        ] += 1

        best[
            "reason"
        ] = (
            f"{anchor_name.lower()} anchor; "
            + best["reason"]
        )

    # Fill remaining slots by score.
    remaining = sorted(
        [
            candidate
            for candidate in scored
            if candidate not in selected
        ],
        key=lambda item: (
            -item["score"],
            item["timestamp"],
        ),
    )

    for candidate in remaining:

        if len(selected) >= MAX_SELECTED_PER_SEGMENT:
            break

        if (
            app_counts[
                candidate["app"]
            ]
            >= MAX_PER_APP
        ):
            continue

        too_close = any(
            abs(
                (
                    candidate["timestamp"]
                    - chosen["timestamp"]
                ).total_seconds()
            )
            < MIN_SCREENSHOT_SEPARATION_SECONDS
            for chosen in selected
        )

        if too_close:
            continue

        selected.append(candidate)
        app_counts[
            candidate["app"]
        ] += 1

    selected.sort(
        key=lambda item: item["timestamp"]
    )

    # Reassign deterministic selection order.
    for index, item in enumerate(
        selected,
        start=1,
    ):
        item["selection_order"] = index

    return selected


# ============================================================
# COPY SELECTED IMAGES
# ============================================================

def copy_selected_screenshots(
    target_root,
    selected_records,
    screenshot_index,
    session_id,
    segment_index,
):
    target_dir = (
        target_root
        / session_id
        / f"segment_{segment_index}"
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_records = []

    for record in selected_records:

        filename = record[
            "filename"
        ]

        paths = screenshot_index.get(
            filename,
            [],
        )

        copied_paths = []

        for source_path in paths:

            destination = (
                target_dir
                / (
                    f"{record['selection_order']:02d}_"
                    + filename
                )
            )

            try:
                shutil.copy2(
                    source_path,
                    destination,
                )

                copied_paths.append(
                    str(destination)
                )

            except OSError:
                continue

            # Usually there should be only one path.
            # If the filename is ambiguous, preserve every
            # valid physical match but keep the ambiguity visible.

        output_records.append(
            {
                **record,
                "source_paths":
                    " | ".join(
                        str(path)
                        for path in paths
                    ),
                "copied_paths":
                    " | ".join(
                        copied_paths
                    ),
                "ambiguous_filename":
                    len(paths) > 1,
            }
        )

    return output_records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DATASET B SCREENSHOT SELECTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SEGMENT_INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing segment inventory:\n"
            f"{SEGMENT_INVENTORY_FILE}"
        )

    print()
    print("Loading segment inventory...")

    inventory = pd.read_csv(
        SEGMENT_INVENTORY_FILE,
        keep_default_na=False,
    )

    required_columns = {
        "session_id",
        "segment_index",
        "start",
        "end",
    }

    missing = (
        required_columns
        - set(inventory.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    inventory["segment_index"] = (
        pd.to_numeric(
            inventory[
                "segment_index"
            ],
            errors="raise",
        ).astype(int)
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
        f"Segments to inspect: "
        f"{len(inventory):,}"
    )

    # --------------------------------------------------------
    # Raw Dataset B sessions.
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
    # Prepare event timelines.
    # --------------------------------------------------------

    print()
    print("Preparing raw event timelines...")

    prepared_sessions = {}

    for session_id in inventory[
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
    # Screenshot index.
    # --------------------------------------------------------

    screenshot_index = (
        build_screenshot_index()
    )

    # --------------------------------------------------------
    # Reset selected output directory.
    # --------------------------------------------------------

    if SELECTED_DIR.exists():

        print()
        print(
            "Removing previous selected screenshot directory..."
        )

        shutil.rmtree(
            SELECTED_DIR
        )

    SELECTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Process segments.
    # --------------------------------------------------------

    print()
    print(
        "Selecting screenshots..."
    )

    records = []

    for row_number, (_, segment) in enumerate(
        inventory.iterrows(),
        start=1,
    ):

        session_id = segment[
            "session_id"
        ]

        segment_index = int(
            segment[
                "segment_index"
            ]
        )

        start = segment[
            "start"
        ]

        end = segment[
            "end"
        ]

        prepared = prepared_sessions[
            session_id
        ]

        candidates = collect_segment_screenshots(
            prepared,
            start,
            end,
        )

        selected = select_screenshots_for_segment(
            candidates,
            start,
            end,
        )

        copied = copy_selected_screenshots(
            SELECTED_DIR,
            selected,
            screenshot_index,
            session_id,
            segment_index,
        )

        for item in copied:

            records.append(
                {
                    "session_id":
                        session_id,

                    "segment_index":
                        segment_index,

                    "segment_start":
                        start,

                    "segment_end":
                        end,

                    "segment_duration_seconds":
                        (
                            end - start
                        ).total_seconds(),

                    "selection_order":
                        item[
                            "selection_order"
                        ],

                    "screenshot_timestamp":
                        item[
                            "timestamp"
                        ],

                    "seconds_from_segment_start":
                        (
                            item[
                                "timestamp"
                            ]
                            - start
                        ).total_seconds(),

                    "app":
                        item["app"],

                    "source_event_type":
                        item[
                            "source_event_type"
                        ],

                    "selection_score":
                        item["score"],

                    "selection_reason":
                        item["reason"],

                    "nearest_app_transition":
                        item[
                            "nearest_transition"
                        ],

                    "filename":
                        item["filename"],

                    "source_paths":
                        item[
                            "source_paths"
                        ],

                    "copied_paths":
                        item[
                            "copied_paths"
                        ],

                    "ambiguous_filename":
                        item[
                            "ambiguous_filename"
                        ],
                }
            )

        if (
            row_number % 50 == 0
            or row_number == len(inventory)
        ):
            print(
                f"Processed "
                f"{row_number}/"
                f"{len(inventory)} segments"
            )

    # --------------------------------------------------------
    # Save audit file.
    # --------------------------------------------------------

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        SELECTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SCREENSHOT SELECTION SUMMARY")
    print("=" * 70)

    print(
        f"Segments analyzed: "
        f"{len(inventory):,}"
    )

    print(
        f"Screenshots selected: "
        f"{len(result):,}"
    )

    if len(inventory) > 0:
        print(
            f"Mean screenshots/segment: "
            f"{len(result) / len(inventory):.2f}"
        )

    segments_with_zero = (
        set(
            inventory[
                [
                    "session_id",
                    "segment_index",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        )
        -
        set(
            result[
                [
                    "session_id",
                    "segment_index",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        )
    )

    print(
        f"Segments with zero selected screenshots: "
        f"{len(segments_with_zero):,}"
    )

    if len(result) > 0:
        print()
        print(
            "Selected screenshots by app:"
        )

        print(
            result[
                "app"
            ]
            .value_counts()
            .head(15)
            .to_string()
        )

        print()
        print(
            "Selection reasons:"
        )

        reasons = (
            result[
                "selection_reason"
            ]
            .str.replace(
                r"; .*",
                "",
                regex=True,
            )
            .value_counts()
            .head(15)
        )

        print(
            reasons.to_string()
        )

    print()
    print("=" * 70)
    print("SCREENSHOT SELECTION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Audit CSV:\n"
        f"{SELECTION_FILE}"
    )

    print()
    print(
        f"Selected image directory:\n"
        f"{SELECTED_DIR}"
    )


if __name__ == "__main__":
    main()
