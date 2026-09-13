
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json
import shutil

import pandas as pd


# ============================================================
# STEP 2 — SEGMENT -> EXACT SESSION/CHUNK SCREENSHOT EVIDENCE
# ============================================================
#
# This script intentionally follows the user's requested logic:
#
#   Step-1 segment
#       ↓
#   exact session_id
#       ↓
#   exact Dataset B session folder
#       ↓
#   recursively inspect ALL chunk folders
#       ↓
#   read ALL events.jsonl
#       ↓
#   keep events inside the segment time interval
#       ↓
#   collect ALL screenshot references from those events
#       ↓
#   resolve references to screenshots in the SAME SESSION/CHUNK
#       ↓
#   preserve every screenshot, including equal timestamps
#
# It does NOT:
#   - cap screenshots
#   - rank screenshots
#   - OCR
#   - translate Japanese
#   - infer a process label
#
# This is evidence collection only.
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

FAMILY_OCCURRENCES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step2_dataset_b_family_occurrences.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step2_exact_segment_evidence"
)

SCREENSHOT_ROOT = (
    OUTPUT_DIR
    / "screenshots"
)

SCREENSHOT_EVIDENCE_FILE = (
    OUTPUT_DIR
    / "screenshot_evidence.csv"
)

EVENT_EVIDENCE_FILE = (
    OUTPUT_DIR
    / "event_evidence.csv"
)

SEGMENT_SUMMARY_FILE = (
    OUTPUT_DIR
    / "segment_summary.csv"
)


# These are the 12 Step-2 validation segments selected earlier.
# They are only an INITIAL validation batch. The collector itself
# is generic and can be pointed at any segment CSV later.
TARGETS = [
    ("PF004", "ses_20260701-190250-NEELA9BAF", 26),
    ("PF004", "ses_20260701-175747-SIDDHIGUPTAB00B", 7),
    ("PF004", "ses_20260701-191537-LAPTOP-76QMG9DE", 19),

    ("PF005", "ses_20260701-191537-LAPTOP-76QMG9DE", 32),
    ("PF005", "ses_20260701-173246-SIDDHIGUPTAB00B", 16),
    ("PF005", "ses_20260701-181413-LAPTOP-76QMG9DE", 30),

    ("PF008", "ses_20260701-175258-LAPTOP-76QMG9DE", 37),
    ("PF008", "ses_20260701-183232-LAPTOP-76QMG9DE", 5),
    ("PF008", "ses_20260701-171614-CHAITANYA0BCF", 12),

    ("PF009", "ses_20260701-183232-LAPTOP-76QMG9DE", 14),
    ("PF009", "ses_20260701-191537-LAPTOP-76QMG9DE", 13),
    ("PF009", "ses_20260701-190250-NEELA9BAF", 31),
]


# ============================================================
# BASIC HELPERS
# ============================================================

def parse_timestamp(value):
    return pd.to_datetime(
        value,
        utc=True,
    )


def timestamp_ms_to_utc(value):
    return pd.to_datetime(
        int(value),
        unit="ms",
        utc=True,
    )


def get_event_timestamp(event):
    """
    Prefer timestamp_ms when present because it is explicit and
    precise. Fall back to timestamp_iso/timestamp.
    """
    value = event.get("timestamp_ms")

    if value is not None:
        try:
            return timestamp_ms_to_utc(value)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            pass

    for key in (
        "timestamp_iso",
        "timestamp",
    ):
        value = event.get(key)

        if isinstance(value, str):
            try:
                return parse_timestamp(value)
            except Exception:
                pass

    return None


def get_event_type(event):
    value = event.get("event_type")

    return (
        value
        if isinstance(value, str)
        else ""
    )


def get_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}

    value = active_app.get(
        "app_name"
    )

    if (
        isinstance(value, str)
        and value.strip()
    ):
        return value.strip()

    return "UNKNOWN"


def get_url(event):
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
        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    return ""


def get_extracted_text(event):
    context = event.get("context") or {}

    value = context.get(
        "extracted_text"
    )

    if (
        isinstance(value, str)
        and value.strip()
    ):
        return value.strip()

    return ""


def extract_screenshot_references(event):
    """
    Return screenshot filename references exactly as stored
    in the event. Do not deduplicate here.
    """
    payload = event.get("payload") or {}

    reference = payload.get(
        "file_reference"
    )

    if not reference:
        return []

    result = []

    if isinstance(reference, dict):

        filename = reference.get(
            "filename"
        )

        if (
            isinstance(filename, str)
            and filename.strip()
        ):
            result.append(
                filename.strip()
            )

    elif isinstance(reference, list):

        for item in reference:

            if not isinstance(
                item,
                dict,
            ):
                continue

            filename = item.get(
                "filename"
            )

            if (
                isinstance(filename, str)
                and filename.strip()
            ):
                result.append(
                    filename.strip()
                )

    return result


# ============================================================
# SESSION DIRECTORY / CHUNK DISCOVERY
# ============================================================

def find_session_folder(session_id):
    """
    Find the exact Dataset B session directory by folder name.
    Search recursively below DATA_ROOT, but once found, only use
    that session directory.
    """

    direct = DATA_ROOT / session_id

    if direct.exists() and direct.is_dir():
        return direct

    matches = [
        path
        for path in DATA_ROOT.rglob(session_id)
        if path.is_dir()
    ]

    if not matches:
        raise FileNotFoundError(
            f"Dataset B session folder not found: "
            f"{session_id}"
        )

    if len(matches) > 1:
        print(
            "WARNING: multiple matching session "
            f"folders for {session_id}; using first:"
        )
        for match in matches:
            print(f"  {match}")

    return matches[0]


def discover_chunks(session_folder):
    """
    Each chunk is a directory containing events.jsonl.
    We recursively discover every events.jsonl below the exact
    session folder.
    """

    event_files = sorted(
        session_folder.rglob(
            "events.jsonl"
        )
    )

    if not event_files:
        raise FileNotFoundError(
            "No events.jsonl files found in session:\n"
            f"{session_folder}"
        )

    return event_files


def read_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    (
                        line_number,
                        json.loads(line),
                    )
                )

            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path} "
                    f"line {line_number}: {error}"
                )

    return rows


# ============================================================
# SCREENSHOT RESOLUTION
# ============================================================

def build_chunk_screenshot_index(
    chunk_folder,
):
    """
    Index screenshots ONLY within the chunk that produced the
    event. This is preferable to searching the whole dataset.
    """

    index = defaultdict(list)

    screenshot_dir = (
        chunk_folder
        / "screenshots"
    )

    if screenshot_dir.exists():
        candidates = screenshot_dir.rglob("*")
    else:
        candidates = chunk_folder.rglob("*")

    for path in candidates:

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
            path
        )

    return index


# ============================================================
# PROCESS ONE SEGMENT
# ============================================================

def collect_segment(
    family_id,
    session_id,
    segment_index,
    segment,
):
    session_folder = (
        find_session_folder(
            session_id
        )
    )

    chunk_event_files = (
        discover_chunks(
            session_folder
        )
    )

    segment_start = parse_timestamp(
        segment["start"]
    )

    segment_end = parse_timestamp(
        segment["end"]
    )

    screenshot_rows = []
    event_rows = []

    total_events = 0
    matched_events = 0

    # Important: preserve chunk order only as metadata.
    # Final event/screenshot ordering is timestamp based.
    for chunk_number, events_file in enumerate(
        chunk_event_files,
        start=1,
    ):

        chunk_folder = (
            events_file.parent
        )

        screenshot_index = (
            build_chunk_screenshot_index(
                chunk_folder
            )
        )

        event_items = read_jsonl(
            events_file
        )

        for source_line, event in event_items:

            total_events += 1

            timestamp = (
                get_event_timestamp(
                    event
                )
            )

            if timestamp is None:
                continue

            if not (
                segment_start
                <= timestamp
                <= segment_end
            ):
                continue

            matched_events += 1

            event_id = event.get(
                "event_id",
                ""
            )

            event_type = (
                get_event_type(event)
            )

            app = get_app(event)

            url = get_url(event)

            text = get_extracted_text(
                event
            )

            screenshot_refs = (
                extract_screenshot_references(
                    event
                )
            )

            event_rows.append(
                {
                    "family_id":
                        family_id,

                    "session_id":
                        session_id,

                    "segment_index":
                        segment_index,

                    "chunk_number":
                        chunk_number,

                    "chunk_folder":
                        str(chunk_folder),

                    "events_file":
                        str(events_file),

                    "source_line":
                        source_line,

                    "event_id":
                        event_id,

                    "timestamp":
                        timestamp,

                    "seconds_from_segment_start":
                        (
                            timestamp
                            - segment_start
                        ).total_seconds(),

                    "event_type":
                        event_type,

                    "app":
                        app,

                    "url":
                        url,

                    "extracted_text":
                        text,

                    "screenshot_reference_count":
                        len(
                            screenshot_refs
                        ),
                }
            )

            # ------------------------------------------------
            # Every screenshot reference remains a separate row.
            # Even if timestamps are identical.
            # Even if filenames are identical.
            # ------------------------------------------------

            for reference_number, filename in enumerate(
                screenshot_refs,
                start=1,
            ):

                resolved_paths = (
                    screenshot_index.get(
                        filename,
                        [],
                    )
                )

                if not resolved_paths:

                    screenshot_rows.append(
                        {
                            "family_id":
                                family_id,

                            "session_id":
                                session_id,

                            "segment_index":
                                segment_index,

                            "chunk_number":
                                chunk_number,

                            "chunk_folder":
                                str(chunk_folder),

                            "events_file":
                                str(events_file),

                            "source_line":
                                source_line,

                            "event_id":
                                event_id,

                            "timestamp":
                                timestamp,

                            "seconds_from_segment_start":
                                (
                                    timestamp
                                    - segment_start
                                ).total_seconds(),

                            "reference_number":
                                reference_number,

                            "filename":
                                filename,

                            "source_event_type":
                                event_type,

                            "source_app":
                                app,

                            "source_url":
                                url,

                            "resolved_source_path":
                                "",

                            "resolution_status":
                                "MISSING",

                            "ambiguous_match":
                                False,

                            "copy_path":
                                "",
                        }
                    )

                    continue

                for path_number, source_path in enumerate(
                    resolved_paths,
                    start=1,
                ):

                    screenshot_rows.append(
                        {
                            "family_id":
                                family_id,

                            "session_id":
                                session_id,

                            "segment_index":
                                segment_index,

                            "chunk_number":
                                chunk_number,

                            "chunk_folder":
                                str(chunk_folder),

                            "events_file":
                                str(events_file),

                            "source_line":
                                source_line,

                            "event_id":
                                event_id,

                            "timestamp":
                                timestamp,

                            "seconds_from_segment_start":
                                (
                                    timestamp
                                    - segment_start
                                ).total_seconds(),

                            "reference_number":
                                reference_number,

                            "filename":
                                filename,

                            "source_event_type":
                                event_type,

                            "source_app":
                                app,

                            "source_url":
                                url,

                            "resolved_source_path":
                                str(
                                    source_path
                                ),

                            "resolution_status":
                                "RESOLVED",

                            "ambiguous_match":
                                len(
                                    resolved_paths
                                ) > 1,

                            "copy_path":
                                "",
                        }
                    )

    # --------------------------------------------------------
    # Timestamp ordering with stable tie-breakers.
    #
    # Equal timestamps are NOT collapsed.
    # --------------------------------------------------------

    event_rows.sort(
        key=lambda row: (
            row["timestamp"],
            row["chunk_number"],
            row["source_line"],
        )
    )

    screenshot_rows.sort(
        key=lambda row: (
            row["timestamp"],
            row["chunk_number"],
            row["source_line"],
            row["reference_number"],
        )
    )

    # Add deterministic chronological order numbers.
    for order, row in enumerate(
        event_rows,
        start=1,
    ):
        row["segment_event_order"] = order

    for order, row in enumerate(
        screenshot_rows,
        start=1,
    ):
        row["segment_screenshot_order"] = order

    # --------------------------------------------------------
    # Copy ALL resolved screenshot files into an evidence
    # directory. Preserve duplicates/ambiguous physical matches.
    # --------------------------------------------------------

    destination_dir = (
        SCREENSHOT_ROOT
        / family_id
        / session_id
        / f"segment_{segment_index:03d}"
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copy_counter = 0

    for row in screenshot_rows:

        if (
            row["resolution_status"]
            != "RESOLVED"
        ):
            continue

        source = Path(
            row[
                "resolved_source_path"
            ]
        )

        copy_counter += 1

        destination = (
            destination_dir
            / (
                f"{row['segment_screenshot_order']:04d}_"
                f"{copy_counter:03d}_"
                f"{source.name}"
            )
        )

        try:
            shutil.copy2(
                source,
                destination,
            )

            row["copy_path"] = str(
                destination
            )

        except OSError as error:

            row["resolution_status"] = (
                "COPY_FAILED"
            )

            row["copy_path"] = (
                f"ERROR: {error}"
            )

    summary = {
        "family_id":
            family_id,

        "session_id":
            session_id,

        "segment_index":
            segment_index,

        "start":
            segment_start,

        "end":
            segment_end,

        "duration_seconds":
            (
                segment_end
                - segment_start
            ).total_seconds(),

        "session_folder":
            str(session_folder),

        "chunk_count":
            len(chunk_event_files),

        "chunk_event_files":
            " | ".join(
                str(path)
                for path
                in chunk_event_files
            ),

        "all_session_events_scanned":
            total_events,

        "events_inside_segment":
            matched_events,

        "screenshot_reference_rows":
            len(screenshot_rows),

        "unique_screenshot_filenames":
            len(
                {
                    row["filename"]
                    for row in screenshot_rows
                }
            ),

        "resolved_screenshot_rows":
            sum(
                row[
                    "resolution_status"
                ]
                == "RESOLVED"
                for row in screenshot_rows
            ),

        "missing_screenshot_rows":
            sum(
                row[
                    "resolution_status"
                ]
                == "MISSING"
                for row in screenshot_rows
            ),

        "ambiguous_screenshot_rows":
            sum(
                row[
                    "ambiguous_match"
                ]
                for row in screenshot_rows
            ),

        "copied_screenshot_rows":
            sum(
                bool(
                    row["copy_path"]
                )
                and not row[
                    "copy_path"
                ].startswith("ERROR:")
                for row in screenshot_rows
            ),
    }

    return (
        event_rows,
        screenshot_rows,
        summary,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STEP 2 — EXACT SEGMENT/SESSION/CHUNK "
        "SCREENSHOT COLLECTION"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SCREENSHOT_ROOT.exists():
        shutil.rmtree(
            SCREENSHOT_ROOT
        )

    SCREENSHOT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            SEGMENTS_FILE
        )

    print()
    print(
        "Loading Step-1 Dataset B segments..."
    )

    segments = pd.read_csv(
        SEGMENTS_FILE,
        keep_default_na=False,
    )

    segments[
        "segment_index"
    ] = pd.to_numeric(
        segments[
            "segment_index"
        ],
        errors="raise",
    ).astype(int)

    segments["start"] = pd.to_datetime(
        segments["start"],
        utc=True,
    )

    segments["end"] = pd.to_datetime(
        segments["end"],
        utc=True,
    )

    # --------------------------------------------------------
    # Locate the exact 12 target rows.
    # --------------------------------------------------------

    target_rows = []

    for family_id, session_id, segment_index in TARGETS:

        matches = segments[
            (
                segments[
                    "session_id"
                ]
                == session_id
            )
            &
            (
                segments[
                    "segment_index"
                ]
                == segment_index
            )
        ]

        if matches.empty:
            raise RuntimeError(
                "Target Step-1 segment not found:\n"
                f"{family_id} | {session_id} | "
                f"segment {segment_index}"
            )

        row = matches.iloc[0].copy()

        target_rows.append(
            (
                family_id,
                row,
            )
        )

    all_event_rows = []
    all_screenshot_rows = []
    all_summaries = []

    # --------------------------------------------------------
    # Collect exact session/chunk evidence.
    # --------------------------------------------------------

    for target_number, (
        family_id,
        segment,
    ) in enumerate(
        target_rows,
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

        print()
        print(
            f"[{target_number}/"
            f"{len(target_rows)}] "
            f"{family_id} | "
            f"{session_id} | "
            f"segment {segment_index}"
        )

        (
            event_rows,
            screenshot_rows,
            summary,
        ) = collect_segment(
            family_id,
            session_id,
            segment_index,
            segment,
        )

        for row in event_rows:
            row["target_number"] = (
                target_number
            )

        for row in screenshot_rows:
            row["target_number"] = (
                target_number
            )

        summary["target_number"] = (
            target_number
        )

        all_event_rows.extend(
            event_rows
        )

        all_screenshot_rows.extend(
            screenshot_rows
        )

        all_summaries.append(
            summary
        )

        print(
            f"  session folder: "
            f"{summary['session_folder']}"
        )

        print(
            f"  chunks: "
            f"{summary['chunk_count']}"
        )

        print(
            f"  events in segment: "
            f"{summary['events_inside_segment']}"
        )

        print(
            f"  screenshots referenced: "
            f"{summary['screenshot_reference_rows']}"
        )

        print(
            f"  screenshots resolved: "
            f"{summary['resolved_screenshot_rows']}"
        )

        print(
            f"  screenshots missing: "
            f"{summary['missing_screenshot_rows']}"
        )

    # --------------------------------------------------------
    # Global chronological order.
    # --------------------------------------------------------

    all_event_rows.sort(
        key=lambda row: (
            row["session_id"],
            row["segment_index"],
            row["timestamp"],
            row["chunk_number"],
            row["source_line"],
        )
    )

    all_screenshot_rows.sort(
        key=lambda row: (
            row["session_id"],
            row["segment_index"],
            row["timestamp"],
            row["chunk_number"],
            row["source_line"],
            row["reference_number"],
        )
    )

    # --------------------------------------------------------
    # Save CSVs.
    # --------------------------------------------------------

    pd.DataFrame(
        all_event_rows
    ).to_csv(
        EVENT_EVIDENCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        all_screenshot_rows
    ).to_csv(
        SCREENSHOT_EVIDENCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        all_summaries
    ).to_csv(
        SEGMENT_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Final audit.
    # --------------------------------------------------------

    screenshot_df = pd.DataFrame(
        all_screenshot_rows
    )

    summary_df = pd.DataFrame(
        all_summaries
    )

    print()
    print("=" * 70)
    print(
        "EXACT SEGMENT EVIDENCE COLLECTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Target segments: "
        f"{len(target_rows)}"
    )

    print(
        f"Total event rows: "
        f"{len(all_event_rows):,}"
    )

    print(
        f"Total screenshot-reference rows: "
        f"{len(all_screenshot_rows):,}"
    )

    if not screenshot_df.empty:

        resolved = (
            screenshot_df[
                "resolution_status"
            ]
            == "RESOLVED"
        ).sum()

        missing = (
            screenshot_df[
                "resolution_status"
            ]
            == "MISSING"
        ).sum()

        print(
            f"Resolved screenshot rows: "
            f"{resolved:,}"
        )

        print(
            f"Missing screenshot rows: "
            f"{missing:,}"
        )

        print(
            f"Unique screenshot filenames: "
            f"{screenshot_df['filename'].nunique():,}"
        )

        print(
            f"Rows with ambiguous physical matches: "
            f"{screenshot_df['ambiguous_match'].sum():,}"
        )

    print()
    print(
        "Per-segment summary:"
    )

    print(
        summary_df[
            [
                "family_id",
                "session_id",
                "segment_index",
                "chunk_count",
                "events_inside_segment",
                "screenshot_reference_rows",
                "resolved_screenshot_rows",
                "missing_screenshot_rows",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Screenshot evidence:\n"
        f"{SCREENSHOT_EVIDENCE_FILE}"
    )

    print(
        f"\nEvent evidence:\n"
        f"{EVENT_EVIDENCE_FILE}"
    )

    print(
        f"\nSegment summary:\n"
        f"{SEGMENT_SUMMARY_FILE}"
    )

    print(
        f"\nCopied screenshots:\n"
        f"{SCREENSHOT_ROOT}"
    )


if __name__ == "__main__":
    main()
