
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
FINAL_SEGMENTS = PROJECT_ROOT / "segments_v2.jsonl"

OUT = PROJECT_ROOT / "outputs"
EVIDENCE_OUT = OUT / "step2_layer3_authoritative_segment_evidence.csv"
SUMMARY_OUT = OUT / "step2_layer3_integrity_summary.csv"


EXPECTED_SEGMENTS = 540
EXPECTED_EVENTS = 20_477


def parse_timestamp(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if abs(float(value)) >= 10_000_000_000:
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
            ts = parse_timestamp(event.get(key))
            if ts is not None:
                return ts
    return None


def recursive_find(obj, keys):
    found = []

    if isinstance(obj, dict):
        for k, value in obj.items():
            if k in keys:
                found.append(value)
            found.extend(
                recursive_find(value, keys)
            )

    elif isinstance(obj, list):
        for value in obj:
            found.extend(
                recursive_find(value, keys)
            )

    return found


def extract_apps(event):
    values = recursive_find(
        event,
        {
            "app_name",
            "application",
            "process_name",
            "new_app",
            "previous_app",
        },
    )

    result = set()

    def walk(value):
        if isinstance(value, str):
            value = value.strip()
            if value:
                result.add(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for value in values:
        walk(value)

    return result


def extract_windows(event):
    values = recursive_find(
        event,
        {
            "window_title",
            "title",
        },
    )

    result = set()

    def walk(value):
        if isinstance(value, str):
            value = value.strip()
            if value:
                result.add(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for value in values:
        walk(value)

    return result


def extract_urls(event):
    values = recursive_find(
        event,
        {
            "url",
            "href",
            "current_url",
            "browser_url",
        },
    )

    result = set()

    def walk(value):
        if isinstance(value, str):
            value = value.strip()
            if (
                value.startswith("http://")
                or value.startswith("https://")
                or value.startswith("file://")
                or "/#" in value
            ):
                result.add(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for value in values:
        walk(value)

    return result


def extract_dom_values(event):
    values = recursive_find(
        event,
        {
            "id",
            "name",
            "placeholder",
            "target_field",
            "field",
            "element",
        },
    )

    result = set()

    def walk(value):
        if isinstance(value, str):
            value = value.strip()
            if value:
                result.add(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for value in values:
        walk(value)

    return result


def extract_operator(event):
    values = recursive_find(
        event,
        {
            "username_hash",
            "user_hash",
            "username",
        },
    )

    for value in values:
        if isinstance(value, (str, int, float)):
            value = str(value).strip()
            if value:
                return value

    return ""


def discover_event_files():
    files = []

    for dataset_b in DATA_ROOT.rglob("dataset_b"):
        if dataset_b.is_dir():
            files.extend(
                dataset_b.rglob(
                    "events.jsonl"
                )
            )

    return sorted(
        set(files),
        key=str,
    )


def load_segments():
    if not FINAL_SEGMENTS.exists():
        raise FileNotFoundError(
            f"Final Step-1 file not found:\n{FINAL_SEGMENTS}"
        )

    records = []

    with FINAL_SEGMENTS.open(
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
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSON at line {line_number}: {exc}"
                )

            records.append(obj)

    df = pd.DataFrame(
        records
    )

    required = {
        "session_id",
        "start",
        "end",
        "label",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "segments_v2.jsonl missing fields: "
            + ", ".join(sorted(missing))
        )

    df["start"] = pd.to_datetime(
        df["start"],
        utc=True,
    )
    df["end"] = pd.to_datetime(
        df["end"],
        utc=True,
    )

    df["segment_index"] = (
        df.groupby(
            "session_id"
        )
        .cumcount()
        + 1
    )

    df["segment_key"] = (
        df["session_id"].astype(str)
        + "::"
        + df["segment_index"].astype(str)
    )

    return df


def load_all_events():
    event_files = discover_event_files()

    if len(event_files) != 20:
        raise RuntimeError(
            f"Expected 20 Dataset-B events.jsonl files; found {len(event_files)}"
        )

    rows = []
    parse_errors = 0

    for i, event_file in enumerate(
        event_files,
        start=1,
    ):
        session_id = (
            event_file.parent.parent.name
        )
        chunk_id = (
            event_file.parent.name
        )

        print(
            f"Loading events "
            f"{i}/{len(event_files)}: "
            f"{session_id} / {chunk_id}"
        )

        with event_file.open(
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
                    event = json.loads(
                        line
                    )
                except Exception:
                    parse_errors += 1
                    continue

                ts = event_timestamp(
                    event
                )

                if ts is None:
                    continue

                rows.append(
                    {
                        "event_uid": (
                            f"{session_id}::"
                            f"{chunk_id}::"
                            f"{line_number}"
                        ),
                        "session_id":
                            session_id,
                        "chunk_id":
                            chunk_id,
                        "timestamp":
                            ts,
                        "event_type":
                            str(
                                event.get(
                                    "event_type",
                                    "",
                                )
                            ),
                        "layer":
                            str(
                                event.get(
                                    "layer",
                                    "",
                                )
                            ),
                        "operator_hash":
                            extract_operator(
                                event
                            ),
                        "apps":
                            extract_apps(
                                event
                            ),
                        "windows":
                            extract_windows(
                                event
                            ),
                        "urls":
                            extract_urls(
                                event
                            ),
                        "dom_values":
                            extract_dom_values(
                                event
                            ),
                    }
                )

    events = pd.DataFrame(
        rows
    )

    if parse_errors:
        print(
            f"WARNING: JSON parse errors: {parse_errors}"
        )

    return events, parse_errors


def main():
    print("=" * 78)
    print(
        "STEP 2 — LAYER 3 AUTHORITATIVE SEGMENT EVIDENCE"
    )
    print("=" * 78)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    segments = load_segments()
    events, parse_errors = (
        load_all_events()
    )

    print()
    print(
        f"segments_v2.jsonl rows: {len(segments):,}"
    )
    print(
        f"events loaded: {len(events):,}"
    )

    # Basic segment checks.
    duplicate_segment_keys = int(
        segments[
            "segment_key"
        ].duplicated().sum()
    )

    invalid_times = int(
        (
            segments["end"]
            <= segments["start"]
        ).sum()
    )

    session_count = segments[
        "session_id"
    ].nunique()

    expected_sessions = set(
        events[
            "session_id"
        ].unique()
    )

    segment_sessions = set(
        segments[
            "session_id"
        ].unique()
    )

    missing_event_sessions = (
        segment_sessions
        - expected_sessions
    )

    unknown_event_sessions = (
        expected_sessions
        - segment_sessions
    )

    # Build per-session sorted event lists.
    events_by_session = {}

    for session_id, group in events.groupby(
        "session_id"
    ):
        events_by_session[
            session_id
        ] = group.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    evidence_rows = []
    assigned_event_uids = set()
    overlap_event_uids = set()
    out_of_session_events = 0

    for i, segment in segments.iterrows():

        sid = str(
            segment[
                "session_id"
            ]
        )

        session_events = (
            events_by_session.get(
                sid,
                pd.DataFrame(),
            )
        )

        start = segment[
            "start"
        ]
        end = segment[
            "end"
        ]

        matched = session_events[
            (
                session_events[
                    "timestamp"
                ] >= start
            )
            & (
                session_events[
                    "timestamp"
                ] <= end
            )
        ].copy()

        uids = list(
            matched[
                "event_uid"
            ]
        )

        duplicate_assignment = (
            set(uids)
            & assigned_event_uids
        )

        if duplicate_assignment:
            overlap_event_uids.update(
                duplicate_assignment
            )

        assigned_event_uids.update(
            uids
        )

        event_types = Counter(
            matched[
                "event_type"
            ]
        )

        layers = Counter(
            matched[
                "layer"
            ]
        )

        operators = sorted(
            {
                str(x)
                for x in matched[
                    "operator_hash"
                ]
                if str(x).strip()
            }
        )

        apps = set()
        windows = set()
        urls = set()
        dom_values = set()

        for value in matched[
            "apps"
        ]:
            apps.update(value)

        for value in matched[
            "windows"
        ]:
            windows.update(value)

        for value in matched[
            "urls"
        ]:
            urls.update(value)

        for value in matched[
            "dom_values"
        ]:
            dom_values.update(value)

        browser_event_types = {
            "browser_click",
            "browser_form_input",
            "browser_navigation",
            "browser_error",
            "browser_alert",
            "browser_tab_event",
            "extension_connected",
            "extension_disconnected",
        }

        desktop_event_types = {
            "keystroke",
            "mouse_click",
            "mouse_scroll",
            "shortcut",
            "mouse_double_click",
            "mouse_drag_drop",
        }

        row = {
            "segment_key":
                segment[
                    "segment_key"
                ],
            "session_id":
                sid,
            "segment_index":
                int(
                    segment[
                        "segment_index"
                    ]
                ),
            "label":
                str(
                    segment[
                        "label"
                    ]
                ),
            "start":
                start,
            "end":
                end,
            "duration_seconds":
                (
                    end - start
                ).total_seconds(),
            "event_count":
                len(matched),
            "browser_event_count":
                int(
                    sum(
                        event_types[x]
                        for x
                        in browser_event_types
                    )
                ),
            "keystroke_count":
                int(
                    event_types[
                        "keystroke"
                    ]
                ),
            "mouse_click_count":
                int(
                    event_types[
                        "mouse_click"
                    ]
                ),
            "mouse_scroll_count":
                int(
                    event_types[
                        "mouse_scroll"
                    ]
                ),
            "shortcut_count":
                int(
                    event_types[
                        "shortcut"
                    ]
                ),
            "app_switch_count":
                int(
                    event_types[
                        "app_switch"
                    ]
                ),
            "window_title_change_count":
                int(
                    event_types[
                        "window_title_change"
                    ]
                ),
            "clipboard_change_count":
                int(
                    event_types[
                        "clipboard_change"
                    ]
                ),
            "browser_form_input_count":
                int(
                    event_types[
                        "browser_form_input"
                    ]
                ),
            "browser_click_count":
                int(
                    event_types[
                        "browser_click"
                    ]
                ),
            "browser_navigation_count":
                int(
                    event_types[
                        "browser_navigation"
                    ]
                ),
            "screenshot_count":
                int(
                    event_types[
                        "screenshot_smart"
                    ]
                ),
            "text_input_complete_count":
                int(
                    event_types[
                        "text_input_complete"
                    ]
                ),
            "apps":
                " | ".join(
                    sorted(apps)
                ),
            "app_count":
                len(apps),
            "windows":
                " | ".join(
                    sorted(windows)
                ),
            "window_count":
                len(windows),
            "urls":
                " | ".join(
                    sorted(urls)
                ),
            "url_count":
                len(urls),
            "dom_values":
                " | ".join(
                    sorted(dom_values)
                ),
            "dom_value_count":
                len(dom_values),
            "operators":
                " | ".join(
                    operators
                ),
            "operator_count":
                len(operators),
            "layers":
                " | ".join(
                    f"{k}:{v}"
                    for k, v
                    in sorted(
                        layers.items()
                    )
                ),
            "event_types":
                " | ".join(
                    f"{k}:{v}"
                    for k, v
                    in sorted(
                        event_types.items()
                    )
                ),
            "matched_event_uids":
                " | ".join(
                    uids
                ),
            "events_unique_within_segment":
                int(
                    len(uids)
                    == len(
                        set(uids)
                    )
                ),
            "session_match":
                int(
                    all(
                        matched[
                            "session_id"
                        ]
                        == sid
                    )
                    if not matched.empty
                    else True
                ),
        }

        evidence_rows.append(
            row
        )

    evidence = pd.DataFrame(
        evidence_rows
    )

    # Detect raw events not assigned to any segment.
    all_event_uids = set(
        events[
            "event_uid"
        ]
    )

    unassigned_uids = (
        all_event_uids
        - assigned_event_uids
    )

    # For contiguous segments, every in-session event should ideally belong
    # to exactly one segment. We keep this as a diagnostic because the
    # segmenter may start/end inside the session and therefore leave some
    # events outside the predicted partition.
    assigned_event_count = len(
        assigned_event_uids
    )

    unassigned_event_count = len(
        unassigned_uids
    )

    # Check segment contiguity by session.
    contiguity_failures = []

    for sid, group in segments.groupby(
        "session_id"
    ):
        group = group.sort_values(
            "segment_index"
        )

        expected_index = list(
            range(
                1,
                len(group) + 1,
            )
        )

        actual_index = (
            group[
                "segment_index"
            ]
            .tolist()
        )

        if actual_index != expected_index:
            contiguity_failures.append(
                sid
            )

        # No overlap between consecutive intervals.
        if len(group) > 1:
            previous_end = group[
                "end"
            ].iloc[:-1].reset_index(
                drop=True
            )
            next_start = group[
                "start"
            ].iloc[1:].reset_index(
                drop=True
            )

            if (
                next_start
                < previous_end
            ).any():
                contiguity_failures.append(
                    sid
                )

    # Process-family coverage from raw DOM values.
    prefix_counts = Counter()

    prefix_patterns = {
        "pi":
            re.compile(
                r"(?<![A-Za-z0-9])pi-(?:note|ok)(?![A-Za-z0-9])",
                re.I,
            ),
        "la":
            re.compile(
                r"(?<![A-Za-z0-9])la-(?:note|ok)(?![A-Za-z0-9])",
                re.I,
            ),
        "ob":
            re.compile(
                r"(?<![A-Za-z0-9])ob-(?:note|ok)(?![A-Za-z0-9])",
                re.I,
            ),
        "si":
            re.compile(
                r"(?<![A-Za-z0-9])si-(?:note|ok)(?![A-Za-z0-9])",
                re.I,
            ),
        "rt":
            re.compile(
                r"(?<![A-Za-z0-9])rt-(?:note|ok)(?![A-Za-z0-9])",
                re.I,
            ),
    }

    for value in evidence[
        "dom_values"
    ]:
        text = str(value)

        for prefix, pattern in prefix_patterns.items():
            if pattern.search(text):
                prefix_counts[
                    prefix
                ] += 1

    evidence.to_csv(
        EVIDENCE_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [
            [
                "expected_segments",
                EXPECTED_SEGMENTS,
            ],
            [
                "actual_segments",
                len(segments),
            ],
            [
                "expected_events",
                EXPECTED_EVENTS,
            ],
            [
                "actual_events_loaded",
                len(events),
            ],
            [
                "events_assigned_to_segments",
                assigned_event_count,
            ],
            [
                "events_not_assigned_to_any_segment",
                unassigned_event_count,
            ],
            [
                "duplicate_event_assignments",
                len(overlap_event_uids),
            ],
            [
                "parse_errors",
                parse_errors,
            ],
            [
                "duplicate_segment_keys",
                duplicate_segment_keys,
            ],
            [
                "invalid_segment_times",
                invalid_times,
            ],
            [
                "segment_sessions",
                session_count,
            ],
            [
                "event_sessions_missing_from_segments",
                len(missing_event_sessions),
            ],
            [
                "segment_sessions_missing_from_events",
                len(unknown_event_sessions),
            ],
            [
                "session_contiguity_failures",
                len(
                    set(
                        contiguity_failures
                    )
                ),
            ],
            [
                "segments_with_zero_events",
                int(
                    (
                        evidence[
                            "event_count"
                        ]
                        == 0
                    ).sum()
                ),
            ],
            [
                "segments_with_cross_session_events",
                int(
                    (
                        evidence[
                            "session_match"
                        ]
                        == 0
                    ).sum()
                ),
            ],
            [
                "pi_raw_dom_value_occurrence_segments",
                prefix_counts[
                    "pi"
                ],
            ],
            [
                "la_raw_dom_value_occurrence_segments",
                prefix_counts[
                    "la"
                ],
            ],
            [
                "ob_raw_dom_value_occurrence_segments",
                prefix_counts[
                    "ob"
                ],
            ],
            [
                "si_raw_dom_value_occurrence_segments",
                prefix_counts[
                    "si"
                ],
            ],
            [
                "rt_raw_dom_value_occurrence_segments",
                prefix_counts[
                    "rt"
                ],
            ],
        ],
        columns=[
            "metric",
            "value",
        ],
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print(
        "LAYER 3 RESULT"
    )
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "EVENT COUNT BY FINAL SEGMENT LABEL:"
    )

    print(
        evidence.groupby(
            "label"
        ).agg(
            segments=(
                "label",
                "size",
            ),
            total_events=(
                "event_count",
                "sum",
            ),
            median_events=(
                "event_count",
                "median",
            ),
            total_duration_seconds=(
                "duration_seconds",
                "sum",
            ),
        ).to_string()
    )

    passed = (
        len(segments)
        == EXPECTED_SEGMENTS
        and len(events)
        == EXPECTED_EVENTS
        and parse_errors == 0
        and duplicate_segment_keys == 0
        and invalid_times == 0
        and not missing_event_sessions
        and not unknown_event_sessions
        and not overlap_event_uids
        and (
            evidence[
                "session_match"
            ].eq(1).all()
        )
        and not contiguity_failures
    )

    print()
    if passed:
        print(
            "STATUS: PASS"
        )
        print(
            "540 final segments successfully reconciled "
            "against the complete Dataset-B event stream."
        )
    else:
        print(
            "STATUS: REVIEW REQUIRED"
        )

    print()
    print(
        "Authoritative evidence:"
    )
    print(
        EVIDENCE_OUT
    )
    print(
        "Integrity summary:"
    )
    print(
        SUMMARY_OUT
    )


if __name__ == "__main__":
    main()
