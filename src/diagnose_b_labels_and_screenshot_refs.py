
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"

SEGMENTS = OUT / "dataset_b_segments_authoritative.csv"
AUDIT = OUT / "dataset_b_segments_label_audit.csv"
CORRECTED = OUT / "dataset_b_segments_authoritative_v2.csv"
FEATURES = OUT / "dataset_b_process_features_with_operator_v2.csv"

OUT_LABEL_CHANGES = OUT / "dataset_b_label_changes_review.csv"
OUT_OTHER_REVIEW = OUT / "dataset_b_other_review.csv"
OUT_SCREENSHOT_EVENT_AUDIT = (
    OUT / "dataset_b_screenshot_event_reference_audit.csv"
)


def parse_ts(event):
    value = event.get("timestamp_ms")
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass

    return None


def session_dir_candidates(session_id):
    candidates = []

    for root in DATA_ROOT.rglob("dataset_b"):
        p = root / session_id
        if p.is_dir():
            candidates.append(p)

    return sorted(set(candidates))


def load_events(session_id):
    candidates = session_dir_candidates(session_id)

    if not candidates:
        raise FileNotFoundError(
            f"Dataset B session not found: {session_id}"
        )

    scored = []

    for p in candidates:
        files = list(p.rglob("events.jsonl"))
        size = sum(
            f.stat().st_size
            for f in files
            if f.exists()
        )
        scored.append(
            (len(files), size, str(p), p)
        )

    scored.sort(reverse=True)
    session_dir = scored[0][-1]

    rows = []

    for event_file in sorted(
        session_dir.rglob("events.jsonl")
    ):
        with event_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_no, line in enumerate(
                f,
                start=1,
            ):
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except Exception:
                    continue

                rows.append(
                    {
                        "session_dir": session_dir,
                        "event_file": event_file,
                        "line_no": line_no,
                        "event": event,
                        "timestamp_ms": parse_ts(event),
                    }
                )

    return rows


def recursively_collect_paths(
    obj,
    prefix="",
):
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if isinstance(value, str):
                lower_key = str(key).lower()
                if any(
                    marker in lower_key
                    for marker in (
                        "path",
                        "file",
                        "name",
                        "image",
                        "screenshot",
                        "capture",
                    )
                ):
                    found.append(
                        (
                            key_path,
                            value,
                        )
                    )

            found.extend(
                recursively_collect_paths(
                    value,
                    key_path,
                )
            )

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(
                recursively_collect_paths(
                    value,
                    f"{prefix}[{index}]",
                )
            )

    return found


def inspect_screenshot_events(
    session_id,
    max_events=20,
):
    rows = load_events(session_id)

    screenshot_events = [
        row
        for row in rows
        if row["event"].get(
            "event_type"
        ) == "screenshot_smart"
    ]

    audit_rows = []

    for row in screenshot_events[
        :max_events
    ]:

        event = row["event"]
        payload = event.get("payload")

        payload_paths = (
            recursively_collect_paths(
                payload,
                "payload",
            )
            if isinstance(
                payload,
                (dict, list),
            )
            else []
        )

        event_paths = recursively_collect_paths(
            event,
            "event",
        )

        # Deduplicate exact key/value pairs.
        seen = set()
        all_paths = []

        for key_path, value in (
            event_paths + payload_paths
        ):
            item = (
                key_path,
                value,
            )

            if item not in seen:
                seen.add(item)
                all_paths.append(item)

        audit_rows.append(
            {
                "session_id": session_id,
                "line_no": row["line_no"],
                "event_file": str(
                    row["event_file"]
                ),
                "timestamp_ms": row[
                    "timestamp_ms"
                ],
                "top_level_keys": "|".join(
                    sorted(
                        str(k)
                        for k in event.keys()
                    )
                ),
                "payload_type": type(
                    payload
                ).__name__,
                "candidate_paths": json.dumps(
                    all_paths,
                    ensure_ascii=False,
                ),
                "full_event_json": json.dumps(
                    event,
                    ensure_ascii=False,
                ),
            }
        )

    return audit_rows


def main():
    print("=" * 70)
    print(
        "DATASET B LABEL + SCREENSHOT REFERENCE DIAGNOSTIC"
    )
    print("=" * 70)

    for path in (
        SEGMENTS,
        AUDIT,
        CORRECTED,
        FEATURES,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    segments = pd.read_csv(
        SEGMENTS,
        keep_default_na=False,
    )

    audit = pd.read_csv(
        AUDIT,
        keep_default_na=False,
    )

    corrected = pd.read_csv(
        CORRECTED,
        keep_default_na=False,
    )

    features = pd.read_csv(
        FEATURES,
        keep_default_na=False,
    )

    key = [
        "session_id",
        "segment_index",
    ]

    # --------------------------------------------------------
    # 1. Label-change inspection
    # --------------------------------------------------------

    changes = segments[
        key
        + [
            "label",
            "label_resolution",
            "prefix_evidence",
        ]
    ].merge(
        corrected[
            key
            + [
                "label",
                "label_resolution",
                "prefix_evidence",
            ]
        ],
        on=key,
        suffixes=(
            "_before",
            "_after",
        ),
        validate="one_to_one",
    )

    changes = changes[
        changes["label_before"]
        != changes["label_after"]
    ].copy()

    print()
    print("=" * 70)
    print(
        "LABEL CHANGES AFTER BTN-XX-OK SUPPORT"
    )
    print("=" * 70)

    print(
        f"Changed segments: {len(changes)}"
    )

    if not changes.empty:
        print(
            changes.to_string(
                index=False
            )
        )

    changes.to_csv(
        OUT_LABEL_CHANGES,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 2. Inspect "other"
    # --------------------------------------------------------

    other = corrected[
        corrected["label"]
        == "other"
    ].merge(
        features,
        on=key,
        how="left",
        suffixes=(
            "",
            "_feature",
        ),
        validate="one_to_one",
    )

    print()
    print("=" * 70)
    print(
        "OTHER SEGMENTS — REVIEW"
    )
    print("=" * 70)

    print(
        f"Other segments: {len(other)}"
    )

    review_columns = [
        "session_id",
        "segment_index",
        "duration_seconds",
        "event_count",
        "label_resolution",
        "prefix_evidence",
    ]

    for column in (
        "browser_click_count",
        "browser_form_input_count",
        "browser_navigation_count",
        "clipboard_count",
        "window_change_count",
        "unique_window_identities",
        "unique_apps",
    ):
        if column in other.columns:
            review_columns.append(column)

    other_review = (
        other[
            [
                x
                for x in review_columns
                if x in other.columns
            ]
        ]
        .sort_values(
            [
                "event_count",
                "duration_seconds",
            ],
            ascending=False,
        )
    )

    print(
        other_review.to_string(
            index=False
        )
    )

    other_review.to_csv(
        OUT_OTHER_REVIEW,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 3. Screenshot event reference audit
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SCREENSHOT EVENT REFERENCE AUDIT"
    )
    print("=" * 70)

    # Use one ordinary B session with screenshots.
    # Inspect several events, then build the definitive resolver.
    first_session = sorted(
        segments[
            "session_id"
        ].astype(str).unique()
    )[0]

    print(
        f"Inspecting session: {first_session}"
    )

    screenshot_rows = inspect_screenshot_events(
        first_session,
        max_events=20,
    )

    if not screenshot_rows:
        raise RuntimeError(
            "No screenshot_smart events found "
            f"in {first_session}."
        )

    screenshot_audit = pd.DataFrame(
        screenshot_rows
    )

    print(
        f"Screenshot events inspected: "
        f"{len(screenshot_audit)}"
    )

    print()
    print(
        "Candidate screenshot reference fields:"
    )

    for _, row in screenshot_audit.iterrows():
        print()
        print(
            f"line={row['line_no']} "
            f"timestamp_ms={row['timestamp_ms']}"
        )
        print(
            row["candidate_paths"]
        )

    screenshot_audit.to_csv(
        OUT_SCREENSHOT_EVENT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Label changes:\n  {OUT_LABEL_CHANGES}"
    )

    print(
        f"Other review:\n  {OUT_OTHER_REVIEW}"
    )

    print(
        f"Screenshot event audit:\n  "
        f"{OUT_SCREENSHOT_EVENT_AUDIT}"
    )

    print()
    print(
        "Do not run OCR yet. The screenshot event audit "
        "is required to build the correct file resolver."
    )


if __name__ == "__main__":
    main()
