
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Reuse the exact frozen Dataset-B segmentation implementation.
# The uploaded segment_dataset_b.py is expected to live in src/.
from segment_dataset_b import (
    DATA_ROOT,
    load_all_events,
    prepare_session_events,
    train_final_model,
    build_session_candidates,
    merge_candidates,
    FEATURE_COLUMNS,
    THRESHOLD,
    MERGE_GAP_SECONDS,
)
from loader import sort_session_events, get_timestamp_ms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FRESH_SEGMENTS_CSV = (
    OUTPUT_DIR / "dataset_b_segments_authoritative.csv"
)

FINAL_JSONL = PROJECT_ROOT / "segments.jsonl"

LABEL_AUDIT_CSV = (
    OUTPUT_DIR / "dataset_b_segments_label_audit.csv"
)

PREFIX_PATTERN = re.compile(
    r"(?:^|=)(pi|la|ob|si|rt)-"
    r"(?:note|ok)$",
    re.IGNORECASE,
)


def payload(event):
    value = event.get("payload")
    return value if isinstance(value, dict) else {}


def event_type(event):
    value = event.get("event_type")
    return value if isinstance(value, str) else ""


def collect_strings(value):
    found = []

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                found.append((str(key), item))
            else:
                found.extend(collect_strings(item))

    elif isinstance(value, list):
        for item in value:
            found.extend(collect_strings(item))

    return found


def extract_prefix_evidence(event):
    """
    Extract pi/la/ob/si/rt from DOM-style field identifiers/names.

    We deliberately do NOT use case IDs, employee IDs, names, or OCR
    content as the process label. The validated B signal is the field
    prefix itself.
    """
    et = event_type(event).lower()

    if et not in {
        "browser_click",
        "browser_form_input",
        "keystroke",
    }:
        return Counter()

    found = Counter()

    for key, value in collect_strings(payload(event)):
        combined = f"{key}={value}".strip()

        for match in PREFIX_PATTERN.finditer(
            combined.lower()
        ):
            found[match.group(1).lower()] += 1

        # Also inspect the value alone because some payload structures
        # put id/name/target_field in nested objects.
        for match in PREFIX_PATTERN.finditer(
            str(value).strip().lower()
        ):
            found[match.group(1).lower()] += 1

    return found


def label_segment(segment_events):
    """
    Initial B label rule:
      - exactly one prefix -> that prefix
      - multiple prefixes -> prefix with most evidence occurrences
      - tie -> deterministic alphabetical tie-break
      - none -> other
    """
    evidence = Counter()

    for event in segment_events:
        evidence.update(
            extract_prefix_evidence(event)
        )

    if not evidence:
        return "other", evidence, "NONE"

    max_count = max(
        evidence.values()
    )

    winners = sorted(
        prefix
        for prefix, count in evidence.items()
        if count == max_count
    )

    label = winners[0]

    if len(evidence) == 1:
        resolution = "SINGLE_PREFIX"
    else:
        resolution = "MULTI_DOMINANT_PREFIX"

    return label, evidence, resolution


def main():
    print("=" * 70)
    print("AUTHORITATIVE DATASET-B SEGMENTS + LABEL EXPORT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Using exact frozen Step-1 segmentation implementation:")
    print(f"  threshold = {THRESHOLD}")
    print(f"  merge_gap = {MERGE_GAP_SECONDS}s")

    # --------------------------------------------------------
    # 1. Train the exact final A model defined by segment_dataset_b.py
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 1 — FROZEN MODEL")
    print("=" * 70)

    model = train_final_model()

    # --------------------------------------------------------
    # 2. Load B exactly as segment_dataset_b.py does.
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("DATASET B")
    print("=" * 70)

    all_sessions = load_all_events(
        DATA_ROOT
    )

    dataset_b_sessions = {}

    for session_id, events in all_sessions.items():
        datasets = {
            event.get("_dataset")
            for event in events
            if event.get("_dataset")
        }

        if "B" in datasets:
            dataset_b_sessions[
                session_id
            ] = events

    if len(dataset_b_sessions) != 15:
        raise RuntimeError(
            "Expected 15 Dataset-B sessions, found "
            f"{len(dataset_b_sessions)}."
        )

    print(
        f"Dataset B sessions: "
        f"{len(dataset_b_sessions):,}"
    )

    # --------------------------------------------------------
    # 3. Recreate boundaries using the exact frozen pipeline.
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("REGENERATING FROZEN DATASET-B BOUNDARIES")
    print("=" * 70)

    candidate_frames = []

    for position, (
        session_id,
        raw_events,
    ) in enumerate(
        sorted(
            dataset_b_sessions.items()
        ),
        start=1,
    ):
        prepared = prepare_session_events(
            raw_events
        )

        candidates = build_session_candidates(
            session_id,
            prepared,
        )

        if not candidates.empty:
            probabilities = model.predict_proba(
                candidates[
                    FEATURE_COLUMNS
                ]
            )[:, 1]

            candidates = candidates.copy()

            candidates[
                "probability"
            ] = probabilities

            positive = candidates[
                candidates["probability"]
                >= THRESHOLD
            ].copy()

            boundaries = merge_candidates(
                positive,
                MERGE_GAP_SECONDS,
            )

            candidate_frames.append(
                boundaries
            )

        print(
            f"{position:2d}/15 "
            f"{session_id}: "
            f"{len(candidates):,} candidates"
        )

    if not candidate_frames:
        raise RuntimeError(
            "No Dataset-B boundaries were generated."
        )

    boundaries = (
        pd.concat(
            candidate_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "session_id",
                "predicted_ts",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Predicted boundaries: "
        f"{len(boundaries):,}"
    )

    # --------------------------------------------------------
    # 4. Build contiguous segments directly from raw B events.
    #    This mirrors segment_dataset_b.py's build_segments()
    #    but retains the raw segment events for labeling.
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("BUILDING AUTHORITATIVE 540-SEGMENT TABLE")
    print("=" * 70)

    grouped_boundaries = {
        session_id: group.sort_values(
            "predicted_ts"
        ).reset_index(drop=True)
        for session_id, group
        in boundaries.groupby(
            "session_id"
        )
    }

    rows = []

    for session_id, raw_events in sorted(
        dataset_b_sessions.items()
    ):
        prepared = prepare_session_events(
            raw_events
        )

        if not prepared:
            raise RuntimeError(
                f"No prepared events for {session_id}."
            )

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

        # Keep the original raw event together with its timestamp.
        # prepare_session_events() intentionally strips the raw payload,
        # but the Step-2 field-prefix labeler needs DOM attributes.
        raw_timestamped = []

        for raw_event in sort_session_events(
            raw_events
        ):
            timestamp_ms = get_timestamp_ms(
                raw_event
            )

            if timestamp_ms is None:
                continue

            raw_timestamped.append(
                {
                    "timestamp_ms":
                        timestamp_ms,
                    "event":
                        raw_event,
                }
            )

        session_boundaries = grouped_boundaries.get(
            session_id,
            pd.DataFrame(),
        )

        boundary_times = []

        if not session_boundaries.empty:
            boundary_times = [
                ts
                for ts
                in session_boundaries[
                    "predicted_ts"
                ].tolist()
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
            start = points[index]
            end = points[index + 1]

            # Build the segment event list directly while preserving
            # timestamps, avoiding any dependence on event ordering
            # beyond prepared's chronological order.
            timestamped_events = [
                item
                for item in raw_timestamped
                if (
                    start.timestamp() * 1000
                    <= item["timestamp_ms"]
                    <= end.timestamp() * 1000
                )
            ]

            label, evidence, resolution = (
                label_segment(
                    [
                        item["event"]
                        for item
                        in timestamped_events
                    ]
                )
            )

            rows.append(
                {
                    "session_id":
                        session_id,
                    "segment_index":
                        index + 1,
                    "start":
                        start,
                    "end":
                        end,
                    "duration_seconds":
                        (
                            end - start
                        ).total_seconds(),
                    "label":
                        label,
                    "label_resolution":
                        resolution,
                    "prefix_evidence":
                        json.dumps(
                            dict(
                                sorted(
                                    evidence.items()
                                )
                            ),
                            ensure_ascii=False,
                        ),
                    "event_count":
                        len(
                            timestamped_events
                        ),
                }
            )

    segments = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 5. Integrity checks.
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("INTEGRITY CHECKS")
    print("=" * 70)

    if len(segments) != 540:
        raise RuntimeError(
            "Expected exactly 540 segments, found "
            f"{len(segments)}."
        )

    if segments[
        ["session_id", "segment_index"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate session_id + segment_index."
        )

    if (
        segments["start"]
        >= segments["end"]
    ).any():
        raise RuntimeError(
            "Invalid zero/negative duration segment."
        )

    for session_id, group in segments.groupby(
        "session_id"
    ):
        group = group.sort_values(
            "segment_index"
        )

        if len(group) < 1:
            raise RuntimeError(
                f"No segments for {session_id}."
            )

        starts = group[
            "start"
        ].tolist()

        ends = group[
            "end"
        ].tolist()

        for i in range(
            len(group) - 1
        ):
            if ends[i] != starts[i + 1]:
                raise RuntimeError(
                    f"Non-contiguous segments in "
                    f"{session_id} at index {i + 1}."
                )

    print(
        "540-row check: PASSED"
    )
    print(
        "No duplicate segments: PASSED"
    )
    print(
        "Positive durations: PASSED"
    )
    print(
        "Per-session contiguity: PASSED"
    )

    label_counts = (
        segments[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Label distribution:"
    )
    print(
        label_counts.to_string()
    )

    resolution_counts = (
        segments[
            "label_resolution"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Label-resolution distribution:"
    )
    print(
        resolution_counts.to_string()
    )

    # --------------------------------------------------------
    # 6. Save authoritative CSV.
    # --------------------------------------------------------
    segments.to_csv(
        FRESH_SEGMENTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 7. Save compact audit CSV.
    # --------------------------------------------------------
    audit = segments[
        [
            "session_id",
            "segment_index",
            "start",
            "end",
            "duration_seconds",
            "label",
            "label_resolution",
            "prefix_evidence",
            "event_count",
        ]
    ].copy()

    audit.to_csv(
        LABEL_AUDIT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 8. Create required segments.jsonl.
    # --------------------------------------------------------
    with FINAL_JSONL.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for _, row in segments.iterrows():
            record = {
                "session_id":
                    str(
                        row["session_id"]
                    ),
                "start":
                    pd.Timestamp(
                        row["start"]
                    ).isoformat(
                        timespec="milliseconds"
                    ).replace(
                        "+00:00",
                        "Z",
                    ),
                "end":
                    pd.Timestamp(
                        row["end"]
                    ).isoformat(
                        timespec="milliseconds"
                    ).replace(
                        "+00:00",
                        "Z",
                    ),
                "label":
                    str(
                        row["label"]
                    ),
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # 9. Final validation of JSONL.
    # --------------------------------------------------------
    with FINAL_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:
        json_rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(json_rows) != 540:
        raise RuntimeError(
            "segments.jsonl does not contain "
            f"540 records; found {len(json_rows)}."
        )

    required = {
        "session_id",
        "start",
        "end",
        "label",
    }

    for i, record in enumerate(
        json_rows,
        start=1,
    ):
        if set(record) != required:
            raise RuntimeError(
                f"Unexpected JSON keys on line {i}: "
                f"{set(record)}"
            )

    print()
    print("=" * 70)
    print("AUTHORITATIVE EXPORT COMPLETE")
    print("=" * 70)

    print(
        f"Segments CSV:\n"
        f"  {FRESH_SEGMENTS_CSV}"
    )

    print(
        f"Label audit:\n"
        f"  {LABEL_AUDIT_CSV}"
    )

    print(
        f"Required deliverable:\n"
        f"  {FINAL_JSONL}"
    )

    print()
    print(
        "segments.jsonl: 540 records"
    )

    print(
        "Labels used: "
        + ", ".join(
            sorted(
                set(
                    segments["label"]
                )
            )
        )
    )


if __name__ == "__main__":
    main()
