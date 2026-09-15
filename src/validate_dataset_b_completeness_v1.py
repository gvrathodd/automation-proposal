
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT
    / "raw_data"
    / "dataset-downloads"
)

FINAL_SEGMENTS = (
    PROJECT_ROOT / "segments_v2.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs"
)

CHUNK_AUDIT = (
    OUTPUT_DIR
    / "step2_layer1_dataset_b_chunk_audit.csv"
)

SESSION_AUDIT = (
    OUTPUT_DIR
    / "step2_layer1_dataset_b_session_audit.csv"
)

SUMMARY = (
    OUTPUT_DIR
    / "step2_layer1_dataset_b_completeness_summary.csv"
)


def parse_ts(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        # Epoch milliseconds.
        if abs(value) >= 10_000_000_000:
            return pd.to_datetime(
                int(value),
                unit="ms",
                utc=True,
            )
        return pd.to_datetime(
            int(value),
            unit="s",
            utc=True,
        )

    try:
        return pd.to_datetime(
            value,
            utc=True,
        )
    except Exception:
        return None


def event_timestamp(event):
    for key in (
        "timestamp",
        "timestamp_ms",
        "timestamp_iso",
        "ts",
        "time",
    ):
        if key in event:
            parsed = parse_ts(
                event.get(key)
            )
            if parsed is not None:
                return parsed

    return None


def read_manifest(path):
    raw = path.read_text(
        encoding="utf-8"
    ).strip()

    if not raw:
        return {}

    # Standard Dataset-B manifest.json.
    if raw.startswith("{"):
        return json.loads(raw)

    # Defensive support for JSONL-style manifest.
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return json.loads(line)

    return {}


def screenshot_count(directory):
    if not directory.is_dir():
        return 0

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    return sum(
        1
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in valid_extensions
    )


def find_dataset_b_sessions():
    sessions = set()

    for dataset_b in DATA_ROOT.rglob(
        "dataset_b"
    ):
        if not dataset_b.is_dir():
            continue

        for child in dataset_b.iterdir():
            if (
                child.is_dir()
                and child.name.startswith(
                    "ses_"
                )
            ):
                sessions.add(child)

    return sorted(
        sessions,
        key=lambda p: p.name,
    )


def audit_chunk(
    session_dir,
    events_file,
    manifest_file,
):
    manifest = read_manifest(
        manifest_file
    )

    loaded_events = []
    parse_errors = 0

    with events_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
                loaded_events.append(event)
            except Exception:
                parse_errors += 1

    timestamps = [
        event_timestamp(event)
        for event in loaded_events
    ]

    timestamps = [
        ts
        for ts in timestamps
        if ts is not None
    ]

    loaded_first = (
        min(timestamps)
        if timestamps
        else None
    )

    loaded_last = (
        max(timestamps)
        if timestamps
        else None
    )

    manifest_stats = (
        manifest.get(
            "statistics",
            {},
        )
        if isinstance(
            manifest,
            dict,
        )
        else {}
    )

    manifest_event_count = (
        manifest_stats.get(
            "total_events"
        )
    )

    manifest_time = (
        manifest.get(
            "time_range",
            {},
        )
        if isinstance(
            manifest,
            dict,
        )
        else {}
    )

    manifest_start = parse_ts(
        manifest_time.get(
            "start_iso"
        )
        or manifest_time.get(
            "start_ms"
        )
    )

    manifest_end = parse_ts(
        manifest_time.get(
            "end_iso"
        )
        or manifest_time.get(
            "end_ms"
        )
    )

    manifest_files = (
        manifest.get(
            "files",
            {},
        )
        if isinstance(
            manifest,
            dict,
        )
        else {}
    )

    screenshot_meta = (
        manifest_files.get(
            "screenshots",
            {},
        )
        if isinstance(
            manifest_files,
            dict,
        )
        else {}
    )

    manifest_screenshot_count = (
        screenshot_meta.get(
            "file_count"
        )
        if isinstance(
            screenshot_meta,
            dict,
        )
        else None
    )

    screenshot_dir = (
        events_file.parent
        / "screenshots"
    )

    loaded_screenshot_count = (
        screenshot_count(
            screenshot_dir
        )
    )

    manifest_count_match = (
        manifest_event_count
        is not None
        and int(
            manifest_event_count
        )
        == len(
            loaded_events
        )
    )

    screenshot_count_match = (
        manifest_screenshot_count
        is not None
        and int(
            manifest_screenshot_count
        )
        == loaded_screenshot_count
    )

    # "Inside manifest range" means the loaded event range is completely
    # contained by the manifest time range.
    timestamp_range_match = (
        manifest_start is not None
        and manifest_end is not None
        and (
            loaded_first is None
            or (
                manifest_start
                <= loaded_first
                <= manifest_end
                and manifest_start
                <= loaded_last
                <= manifest_end
            )
        )
    )

    # Event-type / layer counts provide a second independent reconciliation
    # against manifest statistics.
    manifest_event_types = (
        manifest_stats.get(
            "events_by_type",
            {},
        )
        if isinstance(
            manifest_stats,
            dict,
        )
        else {}
    )

    manifest_layers = (
        manifest_stats.get(
            "events_by_layer",
            {},
        )
        if isinstance(
            manifest_stats,
            dict,
        )
        else {}
    )

    loaded_event_types = Counter(
        str(
            event.get(
                "event_type",
                "",
            )
        )
        for event in loaded_events
    )

    loaded_layers = Counter(
        str(
            event.get(
                "layer",
                "",
            )
        )
        for event in loaded_events
    )

    event_type_match = (
        isinstance(
            manifest_event_types,
            dict,
        )
        and dict(
            loaded_event_types
        )
        == {
            str(k): int(v)
            for k, v
            in manifest_event_types.items()
        }
    )

    layer_match = (
        isinstance(
            manifest_layers,
            dict,
        )
        and dict(
            loaded_layers
        )
        == {
            str(k): int(v)
            for k, v
            in manifest_layers.items()
        }
    )

    row = {
        "session_id":
            session_dir.name,
        "chunk_id":
            manifest.get(
                "chunk_id",
                events_file.parent.name,
            ),
        "chunk_path":
            str(
                events_file.parent
            ),
        "events_file":
            str(
                events_file
            ),
        "manifest_file":
            str(
                manifest_file
            ),
        "manifest_event_count":
            manifest_event_count,
        "loaded_event_count":
            len(
                loaded_events
            ),
        "manifest_count_match":
            int(
                manifest_count_match
            ),
        "manifest_first_timestamp":
            manifest_start,
        "manifest_last_timestamp":
            manifest_end,
        "loaded_first_timestamp":
            loaded_first,
        "loaded_last_timestamp":
            loaded_last,
        "timestamp_range_match":
            int(
                timestamp_range_match
            ),
        "manifest_screenshot_count":
            manifest_screenshot_count,
        "loaded_screenshot_count":
            loaded_screenshot_count,
        "screenshot_count_match":
            int(
                screenshot_count_match
            ),
        "event_type_counts_match":
            int(
                event_type_match
            ),
        "layer_counts_match":
            int(
                layer_match
            ),
        "parse_errors":
            parse_errors,
        "previous_chunk_id":
            manifest.get(
                "previous_chunk_id",
                "",
            ),
        "next_chunk_id":
            manifest.get(
                "next_chunk_id",
                "",
            ),
        "has_gaps":
            bool(
                manifest.get(
                    "gaps",
                    [],
                )
            ),
    }

    checks = [
        manifest_count_match,
        screenshot_count_match,
        timestamp_range_match,
        event_type_match,
        layer_match,
        parse_errors == 0,
    ]

    row["all_internal_checks_pass"] = int(
        all(checks)
    )

    return row


def audit_session(
    session_dir
):
    chunk_rows = []

    # Each chunk should contain events.jsonl and manifest.json.
    for events_file in sorted(
        session_dir.rglob(
            "events.jsonl"
        )
    ):
        chunk_dir = (
            events_file.parent
        )

        manifest_json = (
            chunk_dir
            / "manifest.json"
        )
        manifest_jsonl = (
            chunk_dir
            / "manifest.jsonl"
        )

        manifest_file = (
            manifest_json
            if manifest_json.exists()
            else manifest_jsonl
        )

        if not manifest_file.exists():
            chunk_rows.append(
                {
                    "session_id":
                        session_dir.name,
                    "chunk_id":
                        chunk_dir.name,
                    "chunk_path":
                        str(
                            chunk_dir
                        ),
                    "events_file":
                        str(
                            events_file
                        ),
                    "manifest_file":
                        "",
                    "manifest_event_count":
                        None,
                    "loaded_event_count":
                        None,
                    "manifest_count_match":
                        0,
                    "manifest_first_timestamp":
                        None,
                    "manifest_last_timestamp":
                        None,
                    "loaded_first_timestamp":
                        None,
                    "loaded_last_timestamp":
                        None,
                    "timestamp_range_match":
                        0,
                    "manifest_screenshot_count":
                        None,
                    "loaded_screenshot_count":
                        0,
                    "screenshot_count_match":
                        0,
                    "event_type_counts_match":
                        0,
                    "layer_counts_match":
                        0,
                    "parse_errors":
                        None,
                    "previous_chunk_id":
                        "",
                    "next_chunk_id":
                        "",
                    "has_gaps":
                        False,
                    "all_internal_checks_pass":
                        0,
                }
            )
            continue

        chunk_rows.append(
            audit_chunk(
                session_dir,
                events_file,
                manifest_file,
            )
        )

    # Check chunk links/order within each session.
    chunk_rows.sort(
        key=lambda row: (
            pd.to_datetime(
                row[
                    "loaded_first_timestamp"
                ],
                utc=True,
                errors="coerce",
            )
            if row[
                "loaded_first_timestamp"
            ]
            else pd.Timestamp.max.tz_localize(
                "UTC"
            ),
            str(
                row[
                    "chunk_id"
                ]
            ),
        )
    )

    # The manifest's previous/next IDs should point to chunks inside this
    # session whenever present. We do not require adjacency by filename.
    chunk_ids = {
        row["chunk_id"]
        for row in chunk_rows
    }

    for row in chunk_rows:
        previous = row[
            "previous_chunk_id"
        ]
        next_id = row[
            "next_chunk_id"
        ]

        previous_ok = (
            not previous
            or previous in chunk_ids
        )
        next_ok = (
            not next_id
            or next_id in chunk_ids
        )

        row[
            "chunk_link_integrity"
        ] = int(
            previous_ok
            and next_ok
        )

        row[
            "all_internal_checks_pass"
        ] = int(
            row[
                "all_internal_checks_pass"
            ]
            and previous_ok
            and next_ok
        )

    session_event_count = sum(
        int(
            row[
                "loaded_event_count"
            ]
            or 0
        )
        for row in chunk_rows
    )

    session_screenshot_count = sum(
        int(
            row[
                "loaded_screenshot_count"
            ]
            or 0
        )
        for row in chunk_rows
    )

    session_chunk_count = len(
        chunk_rows
    )

    session_pass = (
        session_chunk_count > 0
        and all(
            row[
                "all_internal_checks_pass"
            ]
            == 1
            for row in chunk_rows
        )
    )

    return chunk_rows, {
        "session_id":
            session_dir.name,
        "chunk_count":
            session_chunk_count,
        "loaded_event_count":
            session_event_count,
        "loaded_screenshot_count":
            session_screenshot_count,
        "chunks_all_checks_pass":
            int(
                session_pass
            ),
    }


def audit_final_segments():
    if not FINAL_SEGMENTS.exists():
        return {
            "segments_file_exists":
                0,
            "segments_row_count":
                None,
            "segments_session_count":
                None,
        }

    rows = 0
    sessions = set()

    with FINAL_SEGMENTS.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            rows += 1

            try:
                obj = json.loads(
                    line
                )

                session_id = obj.get(
                    "session_id"
                )

                if session_id:
                    sessions.add(
                        str(
                            session_id
                        )
                    )
            except Exception:
                continue

    return {
        "segments_file_exists":
            1,
        "segments_row_count":
            rows,
        "segments_session_count":
            len(
                sessions
            ),
    }


def main():
    print("=" * 78)
    print(
        "STEP 2 — LAYER 1 DATASET-B COMPLETENESS / INTEGRITY"
    )
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sessions = find_dataset_b_sessions()

    print(
        f"Dataset-B sessions discovered: "
        f"{len(sessions)}"
    )

    all_chunk_rows = []
    session_rows = []

    for i, session_dir in enumerate(
        sessions,
        start=1,
    ):
        print(
            f"Auditing session "
            f"{i}/{len(sessions)}: "
            f"{session_dir.name}"
        )

        chunk_rows, session_row = (
            audit_session(
                session_dir
            )
        )

        all_chunk_rows.extend(
            chunk_rows
        )
        session_rows.append(
            session_row
        )

    chunk_df = pd.DataFrame(
        all_chunk_rows
    )

    session_df = pd.DataFrame(
        session_rows
    )

    chunk_df.to_csv(
        CHUNK_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    session_df.to_csv(
        SESSION_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    final_segments = (
        audit_final_segments()
    )

    total_events = int(
        chunk_df[
            "loaded_event_count"
        ].fillna(0).sum()
        if not chunk_df.empty
        else 0
    )

    total_screenshots = int(
        chunk_df[
            "loaded_screenshot_count"
        ].fillna(0).sum()
        if not chunk_df.empty
        else 0
    )

    summary_rows = [
        [
            "sessions_discovered",
            len(sessions),
        ],
        [
            "chunks_discovered",
            len(chunk_df),
        ],
        [
            "events_loaded",
            total_events,
        ],
        [
            "screenshots_found",
            total_screenshots,
        ],
        [
            "chunks_with_manifest",
            int(
                (
                    chunk_df[
                        "manifest_file"
                    ].astype(str)
                    != ""
                ).sum()
            )
            if not chunk_df.empty
            else 0,
        ],
        [
            "manifest_event_count_mismatches",
            int(
                (
                    chunk_df[
                        "manifest_count_match"
                    ]
                    != 1
                ).sum()
            )
            if not chunk_df.empty
            else 0,
        ],
        [
            "manifest_screenshot_count_mismatches",
            int(
                (
                    chunk_df[
                        "screenshot_count_match"
                    ]
                    != 1
                ).sum()
            )
            if not chunk_df.empty
            else 0,
        ],
        [
            "timestamp_range_mismatches",
            int(
                (
                    chunk_df[
                        "timestamp_range_match"
                    ]
                    != 1
                ).sum()
            )
            if not chunk_df.empty
            else 0,
        ],
        [
            "event_type_count_mismatches",
            int(
                (
                    chunk_df[
                        "event_type_counts_match"
                    ]
                    != 1
                ).sum()
            )
            if not chunk_df.empty
            else 0,
        ],
        [
            "layer_count_mismatches",
            int(
                (
                    chunk_df[
                        "layer_counts_match"
                    ]
                    != 1
                ).sum()
            )
            if not chunk_df.empty
            else 0,
        ],
        [
            "chunk_link_integrity_failures",
            int(
                (
                    chunk_df[
                        "chunk_link_integrity"
                    ]
                    != 1
                ).sum()
            )
            if not chunk_df.empty
            else 0,
        ],
        [
            "sessions_with_failures",
            int(
                (
                    session_df[
                        "chunks_all_checks_pass"
                    ]
                    != 1
                ).sum()
            )
            if not session_df.empty
            else 0,
        ],
        [
            "segments_file_exists",
            final_segments[
                "segments_file_exists"
            ],
        ],
        [
            "segments_row_count",
            final_segments[
                "segments_row_count"
            ],
        ],
        [
            "segments_session_count",
            final_segments[
                "segments_session_count"
            ],
        ],
    ]

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "metric",
            "value",
        ],
    )

    summary_df.to_csv(
        SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print(
        "LAYER 1 RESULT"
    )
    print("=" * 78)

    print(
        summary_df.to_string(
            index=False
        )
    )

    passed = (
        len(sessions) == 15
        and len(chunk_df) == 20
        and total_events == 20477
        and int(
            (
                chunk_df[
                    "all_internal_checks_pass"
                ]
                == 1
            ).sum()
        ) == len(chunk_df)
        and final_segments[
            "segments_file_exists"
        ] == 1
        and final_segments[
            "segments_row_count"
        ] == 540
    )

    print()
    if passed:
        print(
            "STATUS: PASS"
        )
        print(
            "Dataset B completeness/integrity checks passed."
        )
    else:
        print(
            "STATUS: REVIEW REQUIRED"
        )
        print(
            "One or more completeness checks failed."
        )

    print()
    print(
        "Outputs:"
    )
    print(
        CHUNK_AUDIT
    )
    print(
        SESSION_AUDIT
    )
    print(
        SUMMARY
    )


if __name__ == "__main__":
    main()
