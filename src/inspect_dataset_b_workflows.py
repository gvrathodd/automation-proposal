from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json
import re

import pandas as pd

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

SEGMENTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_segments.csv"
)

INVENTORY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_segment_inventory.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

INSPECTION_FILE = (
    OUTPUT_DIR
    / "dataset_b_workflow_inspection.csv"
)

# Candidate workflows we want to inspect.
TARGET_PATTERNS = [
    "Microsoft Edge > Microsoft Word > Microsoft Edge",
    "Microsoft Edge > Notepad > Microsoft Edge",
]

# Number of occurrences to display per workflow.
SAMPLE_OCCURRENCES = {
    "Microsoft Edge > Microsoft Word > Microsoft Edge": 5,
    "Microsoft Edge > Notepad > Microsoft Edge": 5,
}

# Raw-event context around each segment boundary.
CONTEXT_SECONDS = 3

# Limit event text in the CSV so it remains readable.
MAX_TEXT_LENGTH = 500


# ============================================================
# General helpers
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value

    return str(value)


def truncate(value, limit=MAX_TEXT_LENGTH):
    value = normalize_text(value)

    if len(value) <= limit:
        return value

    return value[: limit - 3] + "..."


def get_event_type(event):
    value = event.get("event_type")

    if isinstance(value, str):
        return value

    return ""


def get_event_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}

    app_name = active_app.get("app_name")

    if isinstance(app_name, str) and app_name.strip():
        return app_name.strip()

    return "UNKNOWN"


def get_extracted_text(event):
    context = event.get("context") or {}

    value = context.get("extracted_text")

    if isinstance(value, str) and value.strip():
        return value.strip()

    return ""


def get_browser_url(event):
    payload = event.get("payload") or {}
    context = event.get("context") or {}

    possible_values = [
        payload.get("url"),
        payload.get("current_url"),
        payload.get("target_url"),
        context.get("url"),
        context.get("current_url"),
        context.get("browser_url"),
    ]

    for value in possible_values:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_browser_domain(url):
    if not url:
        return ""

    value = url.lower().strip()

    value = re.sub(
        r"^[a-z]+://",
        "",
        value,
    )

    value = value.split("/", 1)[0]
    value = value.split(":", 1)[0]

    if value.startswith("www."):
        value = value[4:]

    return value


def get_screenshot_filenames(event):
    """
    Collect screenshot filenames from payload.file_reference.

    The dataset schema normally uses:
        payload.file_reference.filename

    but this helper also tolerates a list/dict variant.
    """

    payload = event.get("payload") or {}

    file_reference = payload.get(
        "file_reference"
    )

    if not file_reference:
        return []

    filenames = []

    if isinstance(file_reference, dict):
        filename = file_reference.get("filename")

        if isinstance(filename, str):
            filenames.append(filename)

    elif isinstance(file_reference, list):
        for item in file_reference:
            if not isinstance(item, dict):
                continue

            filename = item.get("filename")

            if isinstance(filename, str):
                filenames.append(filename)

    return filenames


def get_timestamp_iso(event):
    timestamp = event.get(
        "timestamp_iso"
    )

    if isinstance(timestamp, str):
        return timestamp

    timestamp_ms = get_timestamp_ms(event)

    if timestamp_ms is None:
        return ""

    return (
        pd.to_datetime(
            timestamp_ms,
            unit="ms",
            utc=True,
        ).isoformat()
    )


# ============================================================
# Screenshot index
# ============================================================

def build_screenshot_index():
    """
    Build filename -> physical paths.

    There are known ambiguous filenames in the dataset, so
    multiple paths are retained rather than silently choosing one.
    """

    print("Indexing screenshots...")

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

        index[path.name].append(
            str(path)
        )

    print(
        f"Screenshot filenames indexed: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# Pattern matching
# ============================================================

def contains_target_pattern(
    app_sequence,
    target_pattern,
):
    """
    Check whether target_pattern appears inside app_sequence.

    Missing values from CSV/pandas are treated as no sequence.
    """

    if not isinstance(app_sequence, str):
        return False

    app_sequence = app_sequence.strip()

    if not app_sequence:
        return False

    normalized = [
        part.strip()
        for part in app_sequence.split(">")
        if part.strip()
    ]

    target = [
        part.strip()
        for part in target_pattern.split(">")
        if part.strip()
    ]

    if not normalized or not target:
        return False

    if len(normalized) < len(target):
        return False

    for start in range(
        len(normalized) - len(target) + 1
    ):
        if (
            normalized[
                start:start + len(target)
            ]
            == target
        ):
            return True

    return False


# ============================================================
# Raw event extraction
# ============================================================

def prepare_session_events(events):
    prepared = []

    for event in sort_session_events(events):

        timestamp_ms = get_timestamp_ms(
            event
        )

        if timestamp_ms is None:
            continue

        prepared.append(
            {
                "timestamp_ms":
                    int(timestamp_ms),

                "timestamp_iso":
                    get_timestamp_iso(event),

                "event_type":
                    get_event_type(event),

                "app":
                    get_event_app(event),

                "extracted_text":
                    get_extracted_text(event),

                "browser_url":
                    get_browser_url(event),

                "screenshot_filenames":
                    get_screenshot_filenames(event),

                "raw_event":
                    event,
            }
        )

    return prepared


def find_events_near_segment(
    prepared_events,
    start,
    end,
    context_seconds,
):
    start_ms = int(
        (
            start
            - pd.Timedelta(
                seconds=context_seconds
            )
        ).timestamp()
        * 1000
    )

    end_ms = int(
        (
            end
            + pd.Timedelta(
                seconds=context_seconds
            )
        ).timestamp()
        * 1000
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
# Extract compact event summary
# ============================================================

def summarize_events(events):
    event_types = []
    apps = []
    urls = []
    domains = []
    screenshots = []
    extracted_text = []

    for event in events:

        if event["event_type"]:
            event_types.append(
                event["event_type"]
            )

        if event["app"]:
            apps.append(
                event["app"]
            )

        if event["browser_url"]:
            urls.append(
                event["browser_url"]
            )

            domain = get_browser_domain(
                event["browser_url"]
            )

            if domain:
                domains.append(domain)

        screenshots.extend(
            event["screenshot_filenames"]
        )

        if event["extracted_text"]:
            extracted_text.append(
                event["extracted_text"]
            )

    # Preserve first-seen order.
    apps_unique = list(
        dict.fromkeys(apps)
    )

    domains_unique = list(
        dict.fromkeys(domains)
    )

    screenshots_unique = list(
        dict.fromkeys(screenshots)
    )

    event_types_unique = list(
        dict.fromkeys(event_types)
    )

    # A small number of representative URLs.
    urls_unique = list(
        dict.fromkeys(urls)
    )[:10]

    text_unique = list(
        dict.fromkeys(
            extracted_text
        )
    )[:10]

    return {
        "context_event_count":
            len(events),

        "context_apps":
            " | ".join(apps_unique),

        "context_event_types":
            " | ".join(event_types_unique),

        "context_domains":
            " | ".join(domains_unique),

        "context_urls":
            " | ".join(
                urls_unique
            ),

        "context_extracted_text":
            " | ".join(
                truncate(x)
                for x in text_unique
            ),

        "screenshot_count":
            len(screenshots_unique),

        "screenshot_filenames":
            " | ".join(
                screenshots_unique
            ),
    }


# ============================================================
# Human inspection rows
# ============================================================

def build_inspection_row(
    segment,
    prepared_events,
    screenshot_index,
    target_pattern,
    sample_rank,
):
    start = pd.Timestamp(
        segment["start"]
    )

    end = pd.Timestamp(
        segment["end"]
    )

    nearby_events = find_events_near_segment(
        prepared_events,
        start,
        end,
        CONTEXT_SECONDS,
    )

    summary = summarize_events(
        nearby_events
    )

    screenshot_paths = []

    for filename in (
        summary["screenshot_filenames"]
        .split(" | ")
    ):
        if not filename:
            continue

        for path in screenshot_index.get(
            filename,
            [],
        ):
            screenshot_paths.append(path)

    screenshot_paths = list(
        dict.fromkeys(
            screenshot_paths
        )
    )

    # Keep enough information to inspect ambiguous references,
    # but don't explode the CSV size.
    screenshot_paths = screenshot_paths[:10]

    # Pull the highest-value event rows from the context.
    # We prefer transitions, browser actions and clipboard.
    priority_types = {
        "app_switch",
        "browser_navigation",
        "browser_click",
        "browser_form_input",
        "clipboard_change",
        "mouse_click",
        "keystroke",
        "text_input_complete",
    }

    priority_events = [
        event
        for event in nearby_events
        if event["event_type"]
        in priority_types
    ]

    event_lines = []

    for event in priority_events[:40]:

        line = (
            f"{event['timestamp_iso']} | "
            f"{event['event_type']} | "
            f"{event['app']}"
        )

        if event["browser_url"]:
            line += (
                f" | URL={event['browser_url']}"
            )

        if event["extracted_text"]:
            line += (
                f" | TEXT="
                f"{truncate(event['extracted_text'], 200)}"
            )

        event_lines.append(line)

    return {
        "target_pattern":
            target_pattern,

        "sample_rank":
            sample_rank,

        "session_id":
            segment["session_id"],

        "segment_index":
            int(segment["segment_index"]),

        "start":
            start,

        "end":
            end,

        "duration_seconds":
            float(
                segment["duration_seconds"]
            ),

        "app_sequence":
            segment["app_sequence"],

        "dominant_app":
            segment["dominant_app"],

        "browser_navigation_count":
            int(
                segment[
                    "browser_navigation_count"
                ]
            ),

        "browser_click_count":
            int(
                segment[
                    "browser_click_count"
                ]
            ),

        "browser_form_input_count":
            int(
                segment[
                    "browser_form_input_count"
                ]
            ),

        "mouse_click_count":
            int(
                segment[
                    "mouse_click_count"
                ]
            ),

        "keyboard_count":
            int(
                segment[
                    "keyboard_count"
                ]
            ),

        "clipboard_change_count":
            int(
                segment[
                    "clipboard_change_count"
                ]
            ),

        "context_event_count":
            summary[
                "context_event_count"
            ],

        "context_apps":
            summary[
                "context_apps"
            ],

        "context_event_types":
            summary[
                "context_event_types"
            ],

        "context_domains":
            summary[
                "context_domains"
            ],

        "context_urls":
            summary[
                "context_urls"
            ],

        "context_extracted_text":
            summary[
                "context_extracted_text"
            ],

        "screenshot_count":
            summary[
                "screenshot_count"
            ],

        "screenshot_filenames":
            summary[
                "screenshot_filenames"
            ],

        "screenshot_paths":
            " | ".join(
                screenshot_paths
            ),

        "priority_event_details":
            "\n".join(
                event_lines
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DATASET B WORKFLOW SCREENSHOT INSPECTION")
    print("=" * 70)

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{SEGMENTS_FILE}"
        )

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{INVENTORY_FILE}"
        )

    print()
    print("Loading Dataset B segment inventory...")

    inventory = pd.read_csv(
        INVENTORY_FILE
    )

    segments = pd.read_csv(
        SEGMENTS_FILE
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
        f"Inventory rows: "
        f"{len(inventory):,}"
    )

    print(
        f"Segment rows: "
        f"{len(segments):,}"
    )

    # --------------------------------------------------------
    # 1. Identify candidate occurrences
    # --------------------------------------------------------

    selected = []

    for target_pattern in TARGET_PATTERNS:

        matching = inventory[
            inventory[
                "app_sequence"
            ].apply(
                lambda sequence:
                    contains_target_pattern(
                        sequence,
                        target_pattern,
                    )
            )
        ].copy()

        # Prefer longer segments first because they provide
        # more semantic context, then spread samples across
        # sessions where possible.
        matching = (
            matching
            .sort_values(
                [
                    "duration_seconds",
                    "session_id",
                    "segment_index",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
        )

        desired = SAMPLE_OCCURRENCES[
            target_pattern
        ]

        chosen = []

        used_sessions = set()

        # First pass: different sessions.
        for _, row in matching.iterrows():

            session_id = row[
                "session_id"
            ]

            if session_id in used_sessions:
                continue

            chosen.append(row)
            used_sessions.add(
                session_id
            )

            if len(chosen) >= desired:
                break

        # Second pass: fill if necessary.
        if len(chosen) < desired:

            chosen_indexes = {
                (
                    row["session_id"],
                    row["segment_index"],
                )
                for row in chosen
            }

            for _, row in matching.iterrows():

                key = (
                    row["session_id"],
                    row["segment_index"],
                )

                if key in chosen_indexes:
                    continue

                chosen.append(row)

                if len(chosen) >= desired:
                    break

        for rank, row in enumerate(
            chosen,
            start=1,
        ):
            selected.append(
                (
                    target_pattern,
                    rank,
                    row,
                )
            )

        print()
        print(
            f"{target_pattern}"
        )
        print(
            f"Matching segments: "
            f"{len(matching):,}"
        )
        print(
            f"Selected samples: "
            f"{len(chosen):,}"
        )

    if not selected:
        raise RuntimeError(
            "No target workflow occurrences found."
        )

    # --------------------------------------------------------
    # 2. Load raw sessions
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
        f"Dataset B raw sessions: "
        f"{len(dataset_b_sessions):,}"
    )

    # --------------------------------------------------------
    # 3. Prepare raw event sessions
    # --------------------------------------------------------

    prepared_sessions = {}

    needed_sessions = {
        row["session_id"]
        for _, _, row in selected
    }

    for session_id in needed_sessions:

        if session_id not in dataset_b_sessions:
            raise RuntimeError(
                f"Raw Dataset B session missing: "
                f"{session_id}"
            )

        prepared_sessions[
            session_id
        ] = prepare_session_events(
            dataset_b_sessions[
                session_id
            ]
        )

    # --------------------------------------------------------
    # 4. Screenshot index
    # --------------------------------------------------------

    screenshot_index = (
        build_screenshot_index()
    )

    # --------------------------------------------------------
    # 5. Build inspection report
    # --------------------------------------------------------

    print()
    print("Building inspection report...")

    rows = []

    for (
        target_pattern,
        rank,
        segment_row,
    ) in selected:

        session_id = (
            segment_row["session_id"]
        )

        prepared = prepared_sessions[
            session_id
        ]

        rows.append(
            build_inspection_row(
                segment_row,
                prepared,
                screenshot_index,
                target_pattern,
                rank,
            )
        )

    inspection = pd.DataFrame(
        rows
    )

    inspection = (
        inspection
        .sort_values(
            [
                "target_pattern",
                "sample_rank",
            ]
        )
        .reset_index(drop=True)
    )

    inspection.to_csv(
        INSPECTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 6. Print a useful terminal summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SELECTED WORKFLOW SAMPLES")
    print("=" * 70)

    for target_pattern in TARGET_PATTERNS:

        subset = inspection[
            inspection[
                "target_pattern"
            ]
            == target_pattern
        ]

        print()
        print(
            f"### {target_pattern}"
        )

        print(
            subset[
                [
                    "sample_rank",
                    "session_id",
                    "segment_index",
                    "duration_seconds",
                    "app_sequence",
                    "context_domains",
                    "screenshot_count",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("WORKFLOW INSPECTION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Inspection CSV:\n"
        f"{INSPECTION_FILE}"
    )

    print()
    print(
        "Open the CSV and inspect the "
        "'screenshot_paths' and "
        "'priority_event_details' columns."
    )


if __name__ == "__main__":
    main()