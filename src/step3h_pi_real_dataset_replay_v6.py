
"""
STEP 3H — PI REAL DATA REPLAY / FEASIBILITY TEST V6

FINAL CORRECTED REAL-DATA REPLAY

Core correction
---------------
Dataset-B raw events do NOT carry segment_index. The original project loader
groups events by session_id and orders them using timestamp_ms followed by
correlation.sequence_number.

Therefore real events are attached to canonical pi segments by:
    session_id + timestamp interval

The canonical segments.jsonl has:
    label, session_id, start, end

Step-3A-1 provides the validated pi segment table with:
    session_id, segment_index, start/end
and previously established:
    209 pi segments
    7,325 raw event rows attached to pi

This V6:
    1. Loads ALL Dataset-B events from the exact project data root.
    2. Reconstructs all 15 sessions and all chunks.
    3. Maps the 209 canonical pi intervals to A1 segment_index.
    4. Assigns every raw Dataset-B event to a pi segment by timestamp.
    5. Validates the resulting pi-attached raw-event count.
    6. Uses Step-3A-3 PI_Semantic_Profiles as the authoritative semantic OCR
       population (77 direct OCR-covered segments).
    7. Uses Step-3A-4 chronological sequences.
    8. Produces a per-segment offline replay classification.

No live production application is accessed.

Important distinction
---------------------
This is a REAL-DATA OFFLINE REPLAY:
    actual Dataset-B events + actual OCR evidence + actual 209 pi segments.

It is NOT:
    live production automation,
    production accuracy,
    production savings,
    or a statistical estimate of unsupported semantic subtypes.

Output
------
outputs/step3h_pi_real_dataset_replay_v6.xlsx
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
SEGMENTS_PATH = PROJECT_ROOT / "segments.jsonl"
OUT = PROJECT_ROOT / "outputs"

A1_SEG = OUT / "step3a1_pi_authoritative_segment_evidence_v2.csv"
A1_EVT = OUT / "step3a1_pi_authoritative_event_detail_v2.csv"

A3_FILES = sorted(
    OUT.glob("step3a3_pi_ocr_semantic_evidence*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

A4_FILES = sorted(
    OUT.glob("step3a4_pi_workflow_map_v3*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

OUTPUT = OUT / "step3h_pi_real_dataset_replay_v6.xlsx"


def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def read_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ============================================================================
# Original project's exact Dataset-B loader model
# ============================================================================

def discover_event_files(root: Path) -> list[Path]:
    return sorted(
        root.rglob("events.jsonl")
    )


def identify_dataset(path: Path) -> str | None:
    for part in path.parts:
        if part.lower() == "dataset_a":
            return "A"

        if part.lower() == "dataset_b":
            return "B"

    return None


def load_all_events(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    """
    Exact loader architecture used at project start:
        discover all events.jsonl
        identify dataset by path
        group by session_id
        retain original event dictionaries
    """
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    event_files = discover_event_files(root)

    if not event_files:
        raise RuntimeError(
            f"No events.jsonl files found below {root}"
        )

    for path in event_files:
        if identify_dataset(path) != "B":
            continue

        for event in read_jsonl(path):
            event["_dataset"] = "B"
            event["_source_file"] = str(path)

            session_id = event.get("session_id")

            if session_id:
                sessions[str(session_id)].append(event)

    return dict(sessions), [
        p for p in event_files
        if identify_dataset(p) == "B"
    ]


def sort_session_events(
    events: list[dict[str, Any]],
):
    return sorted(
        events,
        key=lambda e: (
            e.get(
                "timestamp_ms",
                float("inf")
            ),
            e.get(
                "correlation",
                {}
            ).get(
                "sequence_number",
                float("inf")
            ),
        ),
    )


def event_timestamp_ms(event):
    value = event.get("timestamp_ms")

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        try:
            return int(float(value))
        except Exception:
            return None

    return None


def event_chunk_id(event):
    value = (
        event.get("correlation", {})
        .get("chunk_id")
    )

    return (
        value
        if isinstance(value, str)
        else None
    )


def extract_app(event):
    context = event.get("context") or {}
    active = context.get("active_app") or {}

    return norm(
        active.get("app_name")
        or event.get("active_app")
        or event.get("application")
        or event.get("app")
    )


def extract_routes(event):
    routes = (
        event.get("routes")
        or event.get("route")
        or event.get("url")
        or event.get("path")
    )

    return norm(routes)


def extract_dom(event):
    return norm(
        event.get("dom_ui_values")
        or event.get("dom")
        or event.get("ui_values")
    )


def extract_documents(event):
    return norm(
        event.get("document_values")
        or event.get("document")
        or event.get("documents")
    )


# ============================================================================
# Canonical segments and A1 mapping
# ============================================================================

def parse_canonical_timestamp(value):
    ts = pd.to_datetime(
        norm(value),
        errors="coerce",
        utc=True,
    )
    return ts


def load_canonical_pi():
    if not SEGMENTS_PATH.exists():
        raise FileNotFoundError(SEGMENTS_PATH)

    rows = []

    for row_number, obj in enumerate(
        read_jsonl(SEGMENTS_PATH)
    ):
        if norm(
            obj.get("label")
        ).lower() != "pi":
            continue

        session = obj.get("session_id")
        start = obj.get("start")
        end = obj.get("end")

        if not session or start is None or end is None:
            continue

        rows.append({
            "canonical_row":
                row_number,
            "session_id":
                str(session),
            "canonical_start":
                norm(start),
            "canonical_end":
                norm(end),
            "start_ts":
                parse_canonical_timestamp(start),
            "end_ts":
                parse_canonical_timestamp(end),
        })

    df = pd.DataFrame(rows)

    if len(df) != 209:
        raise RuntimeError(
            f"Expected 209 canonical pi segments; found {len(df)}."
        )

    return df


def load_a1_segments():
    if not A1_SEG.exists():
        raise FileNotFoundError(A1_SEG)

    a1 = pd.read_csv(A1_SEG)

    required = {
        "session_id",
        "segment_index",
    }

    missing = required - set(a1.columns)

    if missing:
        raise RuntimeError(
            f"A1 segment evidence missing {sorted(missing)}."
        )

    a1["session_id"] = (
        a1["session_id"].astype(str)
    )

    a1["segment_index"] = pd.to_numeric(
        a1["segment_index"],
        errors="coerce",
    ).astype("Int64")

    # Discover actual A1 segment boundary columns.
    start_col = next(
        (
            c for c in (
                "start",
                "segment_start",
                "start_time",
                "start_timestamp",
            )
            if c in a1.columns
        ),
        None,
    )

    end_col = next(
        (
            c for c in (
                "end",
                "segment_end",
                "end_time",
                "end_timestamp",
            )
            if c in a1.columns
        ),
        None,
    )

    if not start_col or not end_col:
        raise RuntimeError(
            "A1 segment evidence has no usable start/end columns. "
            f"Columns={list(a1.columns)}"
        )

    a1["_a1_start_ts"] = (
        a1[start_col].map(
            parse_canonical_timestamp
        )
    )

    a1["_a1_end_ts"] = (
        a1[end_col].map(
            parse_canonical_timestamp
        )
    )

    return a1


def overlap_seconds(
    a_start,
    a_end,
    b_start,
    b_end,
):
    if any(
        pd.isna(x)
        for x in (
            a_start,
            a_end,
            b_start,
            b_end,
        )
    ):
        return -1.0

    latest_start = max(
        a_start,
        b_start,
    )

    earliest_end = min(
        a_end,
        b_end,
    )

    return max(
        0.0,
        (
            earliest_end
            - latest_start
        ).total_seconds(),
    )


def map_canonical_to_a1(
    canonical,
    a1,
):
    by_session = {
        sid: g.copy()
        for sid, g in a1.groupby(
            "session_id"
        )
    }

    used = set()
    rows = []

    for _, c in canonical.iterrows():
        session = c["session_id"]

        candidates = by_session.get(
            session,
            pd.DataFrame(
                columns=a1.columns
            ),
        )

        if candidates.empty:
            continue

        candidates = candidates[
            ~candidates["segment_index"].apply(
                lambda x:
                    (
                        session,
                        int(x)
                    ) in used
            )
        ].copy()

        exact = candidates[
            candidates["_a1_start_ts"].eq(
                c["start_ts"]
            )
            & candidates["_a1_end_ts"].eq(
                c["end_ts"]
            )
        ]

        if len(exact) == 1:
            r = exact.iloc[0]

            key = (
                session,
                int(r["segment_index"])
            )

            used.add(key)

            rows.append({
                "canonical_row":
                    int(c["canonical_row"]),
                "session_id":
                    session,
                "segment_index":
                    int(r["segment_index"]),
                "match_method":
                    "EXACT_TIMESTAMP",
                "overlap_seconds":
                    overlap_seconds(
                        c["start_ts"],
                        c["end_ts"],
                        r["_a1_start_ts"],
                        r["_a1_end_ts"],
                    ),
            })

            continue

        scores = []

        for _, r in candidates.iterrows():
            ov = overlap_seconds(
                c["start_ts"],
                c["end_ts"],
                r["_a1_start_ts"],
                r["_a1_end_ts"],
            )

            if ov < 0:
                continue

            c_dur = max(
                0.001,
                (
                    c["end_ts"]
                    - c["start_ts"]
                ).total_seconds(),
            )

            r_dur = max(
                0.001,
                (
                    r["_a1_end_ts"]
                    - r["_a1_start_ts"]
                ).total_seconds(),
            )

            union = (
                c_dur
                + r_dur
                - ov
            )

            ratio = (
                ov / union
                if union > 0
                else 0
            )

            scores.append(
                (
                    ratio,
                    ov,
                    r,
                )
            )

        if not scores:
            continue

        scores.sort(
            key=lambda x: (
                x[0],
                x[1],
            ),
            reverse=True,
        )

        best = scores[0]

        if len(scores) > 1:
            second = scores[1]

            if (
                best[0] < 0.90
                and (
                    best[0]
                    - second[0]
                    < 0.10
                )
            ):
                raise RuntimeError(
                    "Ambiguous canonical→A1 mapping encountered. "
                    f"session={session}, "
                    f"canonical_row={c['canonical_row']}"
                )

        r = best[2]

        key = (
            session,
            int(r["segment_index"])
        )

        used.add(key)

        rows.append({
            "canonical_row":
                int(c["canonical_row"]),
            "session_id":
                session,
            "segment_index":
                int(r["segment_index"]),
            "match_method":
                "TIMESTAMP_OVERLAP",
            "overlap_seconds":
                best[1],
            "overlap_ratio":
                best[0],
        })

    mapping = pd.DataFrame(rows)

    if len(mapping) != 209:
        raise RuntimeError(
            "Canonical→A1 mapping failed: "
            f"{len(mapping)}/209."
        )

    if mapping[
        ["session_id", "segment_index"]
    ].drop_duplicates().shape[0] != 209:
        raise RuntimeError(
            "Canonical→A1 mapping is not one-to-one."
        )

    return mapping


# ============================================================================
# Real event → pi segment assignment
# ============================================================================

def assign_real_events_to_pi(
    sessions,
    mapping,
    a1,
):
    """
    Assign raw Dataset-B events by:
        session_id + timestamp_ms ∈ A1 segment [start,end]

    This is the critical corrected join.
    """
    # Build authoritative segment intervals only for mapped pi segments.
    mapped_keys = set(
        zip(
            mapping["session_id"].astype(str),
            mapping["segment_index"].astype(int),
        )
    )

    a1_lookup = {}

    for _, r in a1.iterrows():
        if pd.isna(r["segment_index"]):
            continue

        key = (
            str(r["session_id"]),
            int(r["segment_index"]),
        )

        if key not in mapped_keys:
            continue

        start_ts = r["_a1_start_ts"]
        end_ts = r["_a1_end_ts"]

        if pd.isna(start_ts) or pd.isna(end_ts):
            continue

        a1_lookup.setdefault(
            str(r["session_id"]),
            []
        ).append({
            "segment_index":
                int(r["segment_index"]),
            "start_ts":
                start_ts,
            "end_ts":
                end_ts,
        })

    for sid in a1_lookup:
        a1_lookup[sid].sort(
            key=lambda r: r["start_ts"]
        )

    assigned = defaultdict(list)
    unassigned = []

    total_events = 0
    timestamp_missing = 0
    multi_match = 0

    for sid, events in sessions.items():
        intervals = a1_lookup.get(
            str(sid),
            []
        )

        if not intervals:
            continue

        ordered = sort_session_events(
            events
        )

        for event in ordered:
            total_events += 1

            ts_ms = event_timestamp_ms(
                event
            )

            if ts_ms is None:
                timestamp_missing += 1
                unassigned.append({
                    "session_id": sid,
                    "reason":
                        "MISSING_TIMESTAMP_MS",
                    "event_type":
                        norm(
                            event.get("event_type")
                        ),
                    "source_file":
                        event.get(
                            "_source_file",
                            ""
                        ),
                })
                continue

            ts = pd.to_datetime(
                ts_ms,
                unit="ms",
                utc=True,
            )

            matches = [
                interval
                for interval in intervals
                if (
                    interval["start_ts"]
                    <= ts
                    <= interval["end_ts"]
                )
            ]

            if len(matches) == 1:
                idx = matches[0]["segment_index"]

                assigned[
                    (
                        str(sid),
                        int(idx)
                    )
                ].append(
                    event
                )

            elif len(matches) == 0:
                unassigned.append({
                    "session_id": sid,
                    "timestamp_ms": ts_ms,
                    "reason":
                        "OUTSIDE_PI_INTERVALS",
                    "event_type":
                        norm(
                            event.get("event_type")
                        ),
                    "source_file":
                        event.get(
                            "_source_file",
                            ""
                        ),
                })

            else:
                multi_match += 1

                # Do not silently double-assign a boundary event.
                # Pick the interval with maximum interior proximity; if tied,
                # stop rather than guess.
                distances = []

                for interval in matches:
                    left = abs(
                        (
                            ts
                            - interval["start_ts"]
                        ).total_seconds()
                    )
                    right = abs(
                        (
                            interval["end_ts"]
                            - ts
                        ).total_seconds()
                    )

                    distances.append(
                        (
                            min(
                                left,
                                right,
                            ),
                            interval,
                        )
                    )

                distances.sort(
                    key=lambda x: x[0]
                )

                if (
                    len(distances) > 1
                    and abs(
                        distances[0][0]
                        - distances[1][0]
                    ) < 1e-9
                ):
                    raise RuntimeError(
                        "Raw event lies on an ambiguous pi boundary. "
                        f"session={sid}, timestamp_ms={ts_ms}"
                    )

                idx = distances[0][1][
                    "segment_index"
                ]

                assigned[
                    (
                        str(sid),
                        int(idx)
                    )
                ].append(
                    event
                )

    return (
        assigned,
        unassigned,
        {
            "total_events_seen":
                total_events,
            "timestamp_missing":
                timestamp_missing,
            "multi_interval_events":
                multi_match,
        }
    )


# ============================================================================
# OCR — use Step-3A-3 semantic profiles as authoritative
# ============================================================================

def load_authoritative_ocr():
    if not A3_FILES:
        raise FileNotFoundError(
            "Step-3A-3 semantic workbook not found."
        )

    path = A3_FILES[0]

    sem = pd.read_excel(
        path,
        sheet_name="PI_Semantic_Profiles",
    )

    required = {
        "session_id",
        "segment_index",
    }

    if not required.issubset(
        sem.columns
    ):
        raise RuntimeError(
            "A3 semantic profiles do not contain "
            "session_id + segment_index."
        )

    sem["session_id"] = (
        sem["session_id"].astype(str)
    )

    sem["segment_index"] = pd.to_numeric(
        sem["segment_index"],
        errors="coerce",
    ).astype("Int64")

    sem = sem[
        sem["segment_index"].notna()
    ].copy()

    sem = sem.drop_duplicates(
        ["session_id", "segment_index"]
    )

    # Established Step-3A-3 population is 77.
    if len(sem) != 77:
        raise RuntimeError(
            "Authoritative A3 semantic population changed: "
            f"expected 77, found {len(sem)}."
        )

    return sem, path


# ============================================================================
# A4 workflow sequence
# ============================================================================

def load_a4_sequences():
    if not A4_FILES:
        raise FileNotFoundError(
            "Step-3A-4 workflow map workbook not found."
        )

    path = A4_FILES[0]

    seq = pd.read_excel(
        path,
        sheet_name="Segment_Workflow_Sequences",
    )

    if {
        "session_id",
        "segment_index",
    }.issubset(seq.columns):

        seq["session_id"] = (
            seq["session_id"].astype(str)
        )

        seq["segment_index"] = pd.to_numeric(
            seq["segment_index"],
            errors="coerce",
        ).astype("Int64")

    return seq, path


# ============================================================================
# Replay classification
# ============================================================================

def build_replay_rows(
    mapping,
    assigned_events,
    ocr,
    a4,
):
    raw_by_key = assigned_events

    ocr_map = {}

    for _, r in ocr.iterrows():
        ocr_map[
            (
                str(r["session_id"]),
                int(r["segment_index"])
            )
        ] = r.to_dict()

    a4_map = {}

    if not a4.empty:
        for key, g in a4.groupby(
            ["session_id", "segment_index"]
        ):
            a4_map[
                (
                    str(key[0]),
                    int(key[1]),
                )
            ] = g.iloc[0].to_dict()

    rows = []

    for _, m in mapping.iterrows():
        sid = str(m["session_id"])
        idx = int(m["segment_index"])

        events = raw_by_key.get(
            (sid, idx),
            []
        )

        sem = ocr_map.get(
            (sid, idx)
        )

        a4r = a4_map.get(
            (sid, idx)
        )

        counts = Counter(
            norm(
                e.get("event_type")
            )
            for e in events
            if norm(
                e.get("event_type")
            )
        )

        routes = sorted({
            route
            for e in events
            for route in (
                "/payroll-items",
                "/onboarding",
                "/social-insurance",
                "/leave-applications",
                "/resident-tax",
            )
            if route in extract_routes(e).lower()
        })

        apps = sorted({
            extract_app(e)
            for e in events
            if extract_app(e)
        })

        phase_sequence = (
            norm(
                a4r.get(
                    "clean_phase_sequence"
                )
            )
            if a4r
            else ""
        )

        browser_navigation = (
            counts.get(
                "browser_navigation",
                0
            )
        )

        form_input = (
            counts.get(
                "browser_form_input",
                0
            )
        )

        browser_click = (
            counts.get(
                "browser_click",
                0
            )
        )

        clipboard = (
            counts.get(
                "clipboard_change",
                0
            )
        )

        app_switch = (
            counts.get(
                "app_switch",
                0
            )
        )

        has_browser = bool(
            routes
            or browser_navigation
            or browser_click
            or form_input
        )

        has_form = bool(
            form_input
            or "FORM_ENTRY" in phase_sequence
        )

        has_transfer = bool(
            clipboard
            or "DATA_TRANSFER" in phase_sequence
        )

        source_evidence = bool(
            sem is not None
            and any(
                norm(
                    sem.get(c)
                )
                for c in (
                    "business_object",
                    "request_type",
                    "field_names",
                )
            )
        )

        target_evidence = bool(
            has_browser
            and (
                has_form
                or "/payroll-items"
                in routes
                or "FORM_ENTRY"
                in phase_sequence
            )
        )

        transform_evidence = bool(
            has_transfer
            or (
                source_evidence
                and (
                    norm(
                        sem.get(
                            "field_names"
                        )
                    )
                    or norm(
                        sem.get(
                            "request_type"
                        )
                    )
                )
            )
        )

        human_text = ""

        if sem is not None:
            human_text = " | ".join(
                norm(
                    sem.get(c)
                )
                for c in (
                    "action",
                    "decision_options",
                    "ocr_english",
                    "ocr_japanese",
                )
                if norm(
                    sem.get(c)
                )
            )

        human_gate = bool(
            re.search(
                r"登録確定|保留|承認|却下|"
                r"\bapprove\b|\bapproval\b|\bhold\b|"
                r"\breject\b|\bconfirm\b|\bfinalize\b",
                human_text,
                re.I,
            )
        )

        exception_text = ""

        if sem is not None:
            exception_text = " | ".join(
                norm(
                    sem.get(c)
                )
                for c in (
                    "status",
                    "action",
                    "visible_instruction",
                    "ocr_english",
                )
                if norm(
                    sem.get(c)
                )
            )

        explicit_exception = bool(
            re.search(
                r"\b(error|invalid|missing|exception|not found)\b|"
                r"エラー|未入力|不備",
                exception_text,
                re.I,
            )
        )

        has_document = bool(
            any(
                extract_documents(e)
                for e in events
            )
            or (
                sem is not None
                and norm(
                    sem.get("document")
                )
            )
            or (
                "DOCUMENT_WORK"
                in phase_sequence
            )
        )

        variant_flags = []

        if has_document:
            variant_flags.append(
                "DOCUMENT"
            )

        if len(apps) > 1:
            variant_flags.append(
                "MULTI_APPLICATION"
            )

        if app_switch:
            variant_flags.append(
                "APPLICATION_SWITCH"
            )

        if (
            "DOCUMENT_WORK"
            in phase_sequence
        ):
            variant_flags.append(
                "DOCUMENT_PHASE"
            )

        if explicit_exception:
            replay_class = "EXCEPTION"
        elif human_gate and target_evidence:
            replay_class = "HUMAN_GATE"
        elif not events:
            replay_class = "INSUFFICIENT_RAW_EVIDENCE"
        elif not source_evidence:
            replay_class = "STRUCTURAL_ONLY"
        elif not target_evidence:
            replay_class = "INSUFFICIENT_TARGET_EVIDENCE"
        elif not transform_evidence:
            replay_class = "TRANSFORM_REVIEW"
        elif variant_flags:
            replay_class = "REPLAYABLE_WITH_VARIANT"
        else:
            replay_class = "REPLAYABLE"

        timestamps = [
            event_timestamp_ms(e)
            for e in events
        ]
        timestamps = [
            t for t in timestamps
            if t is not None
        ]

        duration = (
            (
                max(timestamps)
                - min(timestamps)
            ) / 1000.0
            if timestamps
            else None
        )

        rows.append({
            "session_id": sid,
            "segment_index": idx,
            "replay_class": replay_class,
            "raw_event_count": len(events),
            "duration_seconds_observed":
                duration,
            "direct_ocr_present":
                sem is not None,
            "case_type":
                (
                    semantic_case_type(
                        sem
                    )
                    if sem is not None
                    else "SEMANTICALLY_UNRESOLVED"
                ),
            "routes_observed":
                " | ".join(routes),
            "applications_observed":
                " | ".join(apps),
            "phase_sequence":
                phase_sequence,
            "browser_navigation_events":
                browser_navigation,
            "browser_form_input_events":
                form_input,
            "browser_click_events":
                browser_click,
            "clipboard_change_events":
                clipboard,
            "app_switch_events":
                app_switch,
            "keyboard_events":
                counts.get(
                    "keystroke",
                    0
                ),
            "shortcut_events":
                counts.get(
                    "shortcut",
                    0
                ),
            "source_evidence":
                source_evidence,
            "target_evidence":
                target_evidence,
            "transformation_evidence":
                transform_evidence,
            "core_preparation_supported":
                bool(
                    source_evidence
                    and target_evidence
                    and transform_evidence
                ),
            "human_gate_evidence":
                human_gate,
            "exception_signal":
                (
                    "SEMANTIC_EXCEPTION_SIGNAL"
                    if explicit_exception
                    else ""
                ),
            "variant_flags":
                " | ".join(
                    sorted(
                        set(
                            variant_flags
                        )
                    )
                ),
            "business_object":
                norm(
                    sem.get(
                        "business_object"
                    )
                )
                if sem is not None
                else "",
            "request_type":
                norm(
                    sem.get(
                        "request_type"
                    )
                )
                if sem is not None
                else "",
            "fields":
                norm(
                    sem.get(
                        "field_names"
                    )
                )
                if sem is not None
                else "",
            "status":
                norm(
                    sem.get(
                        "status"
                    )
                )
                if sem is not None
                else "",
            "action":
                norm(
                    sem.get(
                        "action"
                    )
                )
                if sem is not None
                else "",
            "document":
                norm(
                    sem.get(
                        "document"
                    )
                )
                if sem is not None
                else "",
            "decision_options":
                norm(
                    sem.get(
                        "decision_options"
                    )
                )
                if sem is not None
                else "",
            "ocr_english":
                norm(
                    sem.get(
                        "ocr_english"
                    )
                )
                if sem is not None
                else "",
            "mapping_method":
                norm(
                    m.get(
                        "match_method"
                    )
                ),
            "mapping_overlap_ratio":
                m.get(
                    "overlap_ratio"
                ),
            "notes":
                (
                    "Direct A3 semantic OCR available."
                    if sem is not None
                    else
                    "No direct A3 semantic OCR; structural/raw evidence only."
                ),
        })

    return pd.DataFrame(rows)


def semantic_case_type(sem):
    if not sem:
        return "SEMANTICALLY_UNRESOLVED"

    text = " | ".join(
        norm(
            sem.get(c)
        )
        for c in (
            "business_object",
            "request_type",
            "action",
            "ocr_english",
            "ocr_japanese",
        )
        if norm(
            sem.get(c)
        )
    )

    for pattern, label in [
        (
            r"expense reimbursement|reimbursement|経費精算|経費",
            "EXPENSE_REIMBURSEMENT",
        ),
        (
            r"payroll change|給与変更|payroll",
            "PAYROLL_CHANGE",
        ),
        (
            r"dependent deduction|扶養控除",
            "DEPENDENT_DEDUCTION_CHANGE",
        ),
        (
            r"transportation expense|交通費",
            "TRANSPORTATION_EXPENSE",
        ),
    ]:
        if re.search(
            pattern,
            text,
            re.I
        ):
            return label

    return "OTHER_OR_UNRESOLVED"


def build_workflow_coverage(replay):
    total = len(replay)

    rows = []

    for dimension, mask in [
        (
            "raw_event_evidence",
            replay["raw_event_count"].gt(0),
        ),
        (
            "browser_target_evidence",
            replay["target_evidence"],
        ),
        (
            "browser_navigation_observed",
            replay[
                "browser_navigation_events"
            ].gt(0),
        ),
        (
            "form_input_observed",
            replay[
                "browser_form_input_events"
            ].gt(0),
        ),
        (
            "clipboard_transfer_observed",
            replay[
                "clipboard_change_events"
            ].gt(0),
        ),
        (
            "application_switch_observed",
            replay[
                "app_switch_events"
            ].gt(0),
        ),
        (
            "direct_ocr_semantics",
            replay[
                "direct_ocr_present"
            ],
        ),
        (
            "source_semantics",
            replay["source_evidence"],
        ),
        (
            "transformation_evidence",
            replay[
                "transformation_evidence"
            ],
        ),
        (
            "core_preparation_supported",
            replay[
                "core_preparation_supported"
            ],
        ),
        (
            "human_gate_evidence",
            replay[
                "human_gate_evidence"
            ],
        ),
    ]:
        rows.append({
            "dimension": dimension,
            "segments":
                int(mask.sum()),
            "coverage_pct":
                round(
                    100 * mask.sum() / total,
                    2
                )
                if total
                else 0,
        })

    return pd.DataFrame(rows)


def style_workbook(path):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"

        if (
            ws.max_row > 1
            and ws.max_column > 1
        ):
            ws.auto_filter.ref = (
                ws.dimensions
            )

        for cell in ws[1]:
            cell.font = Font(
                bold=True,
                color="FFFFFF",
            )
            cell.fill = PatternFill(
                "solid",
                fgColor="1F4E78",
            )
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

        for i, cells in enumerate(
            ws.iter_cols(
                min_row=1,
                max_row=min(
                    ws.max_row,
                    120
                ),
            ),
            1,
        ):
            width = max(
                [
                    len(
                        str(c.value or "")
                    )
                    for c in cells
                ]
                + [12]
            )

            ws.column_dimensions[
                get_column_letter(i)
            ].width = min(
                width + 2,
                52,
            )


def main():
    print("=" * 80)
    print("STEP 3H — PI REAL DATA REPLAY / FEASIBILITY TEST V6")
    print("=" * 80)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset-B root not found: {DATA_ROOT}"
        )

    canonical = load_canonical_pi()

    a1 = load_a1_segments()

    mapping = map_canonical_to_a1(
        canonical,
        a1,
    )

    sessions, event_files = (
        load_all_events(
            DATA_ROOT
        )
    )

    if len(sessions) != 15:
        raise RuntimeError(
            f"Expected 15 Dataset-B sessions; found {len(sessions)}."
        )

    total_events = sum(
        len(events)
        for events in sessions.values()
    )

    if total_events != 20477:
        raise RuntimeError(
            f"Expected 20,477 Dataset-B events; found {total_events}."
        )

    assigned, unassigned, assignment_meta = (
        assign_real_events_to_pi(
            sessions,
            mapping,
            a1,
        )
    )

    assigned_count = sum(
        len(v)
        for v in assigned.values()
    )

    # Step-3A-1 already established 7,325 pi-attached raw rows.
    # Require a close exact match; here we require exact equality.
    if assigned_count != 7325:
        raise RuntimeError(
            "Real-event → pi interval assignment mismatch: "
            f"expected 7,325, found {assigned_count}."
        )

    ocr, a3_path = (
        load_authoritative_ocr()
    )

    a4, a4_path = (
        load_a4_sequences()
    )

    replay = build_replay_rows(
        mapping,
        assigned,
        ocr,
        a4,
    )

    if len(
        replay[
            [
                "session_id",
                "segment_index",
            ]
        ].drop_duplicates()
    ) != 209:
        raise RuntimeError(
            "Replay output does not contain 209 unique pi segments."
        )

    class_counts = (
        replay["replay_class"]
        .value_counts()
        .rename_axis("replay_class")
        .reset_index(
            name="segments"
        )
    )

    class_counts["segment_pct"] = (
        100
        * class_counts["segments"]
        / 209
    ).round(2)

    workflow_coverage = (
        build_workflow_coverage(
            replay
        )
    )

    direct_ocr_count = int(
        replay["direct_ocr_present"].sum()
    )

    if direct_ocr_count != 77:
        raise RuntimeError(
            "Replay OCR population mismatch: "
            f"expected 77, found {direct_ocr_count}."
        )

    # Case-type summary strictly over direct OCR-covered segments.
    direct = replay[
        replay["direct_ocr_present"]
    ]

    ocr_summary = (
        direct
        .groupby("case_type")
        .agg(
            direct_ocr_segments=(
                "segment_index",
                "count",
            ),
            sessions=(
                "session_id",
                "nunique",
            ),
            core_preparation_supported=(
                "core_preparation_supported",
                "sum",
            ),
            human_gate=(
                "human_gate_evidence",
                "sum",
            ),
        )
        .reset_index()
    )

    variant_summary = (
        replay[
            replay["variant_flags"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ]
        .assign(
            variant=replay[
                replay["variant_flags"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
            ]["variant_flags"].str.split(
                " | "
            )
        )
        .explode("variant")
        .groupby("variant")
        .agg(
            segments=(
                "segment_index",
                "count",
            ),
            sessions=(
                "session_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    exception_summary = (
        replay[
            replay["replay_class"].isin(
                [
                    "EXCEPTION",
                    "INSUFFICIENT_RAW_EVIDENCE",
                    "INSUFFICIENT_TARGET_EVIDENCE",
                    "TRANSFORM_REVIEW",
                    "STRUCTURAL_ONLY",
                ]
            )
        ]
        .groupby("replay_class")
        .agg(
            segments=(
                "segment_index",
                "count",
            ),
            sessions=(
                "session_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    human_summary = pd.DataFrame([
        {
            "category":
                "DIRECT_HUMAN_GATE_EVIDENCE",
            "segments":
                int(
                    replay[
                        "human_gate_evidence"
                    ].sum()
                ),
            "interpretation":
                "Direct A3 OCR evidence indicates a business decision boundary; decision remains human.",
        }
    ])

    session_df = pd.DataFrame([
        {
            "session_id": sid,
            "event_count": len(events),
            "chunk_count": len({
                event_chunk_id(e)
                for e in events
                if event_chunk_id(e)
            }),
            "chunk_ids":
                " | ".join(
                    sorted({
                        event_chunk_id(e)
                        for e in events
                        if event_chunk_id(e)
                    })
                ),
            "first_timestamp_ms":
                min(
                    [
                        event_timestamp_ms(e)
                        for e in events
                        if event_timestamp_ms(e) is not None
                    ],
                    default=None,
                ),
            "last_timestamp_ms":
                max(
                    [
                        event_timestamp_ms(e)
                        for e in events
                        if event_timestamp_ms(e) is not None
                    ],
                    default=None,
                ),
        }
        for sid, events in sorted(
            sessions.items()
        )
    ])

    final = pd.DataFrame([
        {
            "area": "Canonical population",
            "result":
                "209/209 canonical pi segments mapped uniquely to A1.",
            "evidence_type":
                "OBSERVED_DATASET",
        },
        {
            "area": "Dataset-B reconstruction",
            "result":
                f"15 sessions, {len(event_files)} event files, and {total_events:,} real events reconstructed.",
            "evidence_type":
                "OBSERVED_DATASET",
        },
        {
            "area": "Real pi raw evidence",
            "result":
                "7,325 real Dataset-B event rows were assigned to the 209 pi intervals.",
            "evidence_type":
                "OBSERVED_DATASET",
        },
        {
            "area": "Direct OCR semantics",
            "result":
                "77 pi segments use the authoritative Step-3A-3 semantic profiles.",
            "evidence_type":
                "OBSERVED_DATASET",
        },
        {
            "area": "Offline replay",
            "result":
                "All 209 real pi segments were evaluated using real raw-event evidence plus directly covered OCR semantics.",
            "evidence_type":
                "OFFLINE_REPLAY",
        },
        {
            "area": "Production automation",
            "result":
                "Not tested.",
            "evidence_type":
                "LIMITATION",
        },
        {
            "area": "Production time savings",
            "result":
                "Not estimated.",
            "evidence_type":
                "LIMITATION",
        },
    ])

    validation = pd.DataFrame([
        {
            "check":
                "canonical_pi_population",
            "expected": 209,
            "observed":
                len(canonical),
            "status":
                "PASS"
                if len(canonical) == 209
                else "FAIL",
        },
        {
            "check":
                "dataset_b_sessions",
            "expected": 15,
            "observed":
                len(sessions),
            "status":
                "PASS"
                if len(sessions) == 15
                else "FAIL",
        },
        {
            "check":
                "dataset_b_event_files",
            "expected": ">0",
            "observed":
                len(event_files),
            "status":
                "PASS"
                if event_files
                else "FAIL",
        },
        {
            "check":
                "dataset_b_events",
            "expected": 20477,
            "observed":
                total_events,
            "status":
                "PASS"
                if total_events == 20477
                else "FAIL",
        },
        {
            "check":
                "canonical_a1_mapping",
            "expected": 209,
            "observed":
                len(mapping),
            "status":
                "PASS"
                if len(mapping) == 209
                else "FAIL",
        },
        {
            "check":
                "real_pi_attached_events",
            "expected": 7325,
            "observed":
                assigned_count,
            "status":
                "PASS"
                if assigned_count == 7325
                else "FAIL",
        },
        {
            "check":
                "pi_replay_rows",
            "expected": 209,
            "observed":
                len(
                    replay[
                        [
                            "session_id",
                            "segment_index",
                        ]
                    ].drop_duplicates()
                ),
            "status":
                "PASS"
                if len(
                    replay[
                        [
                            "session_id",
                            "segment_index",
                        ]
                    ].drop_duplicates()
                ) == 209
                else "FAIL",
        },
        {
            "check":
                "direct_ocr_segments",
            "expected": 77,
            "observed":
                direct_ocr_count,
            "status":
                "PASS"
                if direct_ocr_count == 77
                else "FAIL",
        },
        {
            "check":
                "ocr_extrapolation",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check":
                "production_accessed",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check":
                "human_decision_automated",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check":
                "segments_jsonl_modified",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
    ])

    readme = pd.DataFrame([
        [
            "Purpose",
            "Replay the evidence-derived pi preparation logic against all 209 real canonical pi segments.",
        ],
        [
            "Dataset-B source",
            str(DATA_ROOT),
        ],
        [
            "Loader",
            "All recursively discovered events.jsonl files under Dataset-B; events grouped by session_id and chronologically ordered by timestamp_ms + correlation.sequence_number.",
        ],
        [
            "Raw-to-segment join",
            "Raw events do not contain segment_index; they are assigned to A1 pi intervals using session_id + timestamp_ms.",
        ],
        [
            "Expected real-event count",
            "7,325 raw event rows were previously established as attached to the 209 pi segments in Step-3A-1 and are required as a consistency check here.",
        ],
        [
            "OCR",
            "Step-3A-3 PI_Semantic_Profiles is treated as the authoritative direct semantic OCR population: 77 segments.",
        ],
        [
            "Uncovered OCR",
            "The remaining pi segments are not assigned OCR-derived business semantics.",
        ],
        [
            "Interpretation",
            "Replay classification is offline evidence-supported feasibility, not production execution accuracy.",
        ],
    ], columns=["item", "value"])

    with pd.ExcelWriter(
        OUTPUT,
        engine="openpyxl"
    ) as writer:
        readme.to_excel(
            writer,
            sheet_name="README",
            index=False,
        )
        final.to_excel(
            writer,
            sheet_name="Final_Interpretation",
            index=False,
        )
        class_counts.to_excel(
            writer,
            sheet_name="Replay_Class_Counts",
            index=False,
        )
        workflow_coverage.to_excel(
            writer,
            sheet_name="Workflow_Coverage",
            index=False,
        )
        replay.to_excel(
            writer,
            sheet_name="209_Segment_Replay",
            index=False,
        )
        ocr_summary.to_excel(
            writer,
            sheet_name="OCR_Semantic_Summary",
            index=False,
        )
        variant_summary.to_excel(
            writer,
            sheet_name="Variant_Summary",
            index=False,
        )
        exception_summary.to_excel(
            writer,
            sheet_name="Exception_Summary",
            index=False,
        )
        human_summary.to_excel(
            writer,
            sheet_name="Human_Gate_Summary",
            index=False,
        )
        session_df.to_excel(
            writer,
            sheet_name="DatasetB_Sessions",
            index=False,
        )
        mapping.to_excel(
            writer,
            sheet_name="Canonical_A1_Mapping",
            index=False,
        )
        pd.DataFrame(
            unassigned[:5000]
        ).to_excel(
            writer,
            sheet_name="Unassigned_Real_Events",
            index=False,
        )
        validation.to_excel(
            writer,
            sheet_name="Validation",
            index=False,
        )

    style_workbook(OUTPUT)

    print(f"Dataset-B sessions reconstructed: {len(sessions)}")
    print(f"Dataset-B event files discovered: {len(event_files)}")
    print(f"Dataset-B events loaded: {total_events}")
    print(f"Canonical pi segments: {len(canonical)}")
    print(f"Canonical-to-A1 mappings: {len(mapping)}")
    print(
        f"Real raw events attached to pi: {assigned_count}"
    )
    print(
        f"Real raw events outside pi intervals: {len(unassigned)}"
    )
    print(
        f"Direct OCR semantic segments: {direct_ocr_count}"
    )

    print("\n3H RESULT")
    print(final.to_string(index=False))

    print("\nREPLAY CLASS COUNTS")
    print(class_counts.to_string(index=False))

    print("\nWORKFLOW COVERAGE")
    print(workflow_coverage.to_string(index=False))

    print("\nOCR SEMANTIC SUMMARY")
    print(
        ocr_summary.to_string(index=False)
        if not ocr_summary.empty
        else "No direct OCR semantic summary."
    )

    print("\nVALIDATION")
    print(validation.to_string(index=False))

    failed = validation[
        validation["status"] == "FAIL"
    ]

    if not failed.empty:
        raise RuntimeError(
            "3H validation failed."
        )

    print("\nOUTPUT")
    print(OUTPUT)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3H V6 COMPLETE.")


if __name__ == "__main__":
    main()
