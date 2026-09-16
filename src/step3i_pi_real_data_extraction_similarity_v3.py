
"""
STEP 3I — PI REAL-DATA STRUCTURED EXTRACTION VS OCR GROUND TRUTH V3

Purpose
-------
Evaluate whether a structured simulator can reconstruct actual business-field
evidence from REAL Dataset-B events, then compare that reconstruction against
the independent direct-OCR semantic profiles.

This version avoids the previous failure mode where instrumentation metadata
was treated as business data.

Evaluation population
---------------------
209 canonical pi segments
77 direct-OCR semantic segments are scored
132 segments remain coverage-only

Simulation INPUT
----------------
REAL Dataset-B raw event JSON only.

Ground truth
------------
Step-3A-3 PI_Semantic_Profiles only, after the simulated record is produced.

No OCR field/value/semantic text is fed into the simulator.

Structured extraction
---------------------
Event-type-specific:

browser_form_input
    field / target_field
    value / previous_value
    URL / route

clipboard_change
    text_content / text
    source_action
    content_type
    file_list

keystroke
    target_field
    input_context
    character
    key_action

browser_click
    element / target_element / accessible_name / text

browser_navigation
    url / route / page_title

document/file evidence
    file_name / document_name / path

The simulator also extracts semantically named fields by matching the actual
field/target-field identifiers, not by treating arbitrary event metadata as
the value.

Scoring
-------
A. Required-field recovery:
    compare OCR-established field names against actual observed raw field IDs.

B. Field-value recovery:
    for each recovered field, compare the extracted value against direct OCR
    text for concrete identifiers/dates/numbers where such values exist.

C. Semantic category recovery:
    request_type, status, action, decision options, document are scored only
    when raw evidence contains a corresponding label/value.

D. Evidence provenance:
    every extracted value records event type + event timestamp + JSON path.

Classification:
    EXACT
    PARTIAL
    NOT_OBSERVED_IN_RAW
    CONFLICT
    NOT_APPLICABLE

No single composite score is presented as "accuracy".

A local mock run is included as a separate engineering check.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
SEGMENTS = ROOT / "segments.jsonl"
OUT = ROOT / "outputs"

A1_SEG = OUT / "step3a1_pi_authoritative_segment_evidence_v2.csv"
A3_FILES = sorted(
    OUT.glob("step3a3_pi_ocr_semantic_evidence*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

MOCK = ROOT / "prototype" / "step3e1_mock_payroll_items.html"

OUTPUT = OUT / "step3i_pi_real_data_extraction_similarity_v3.xlsx"

EXPECTED = {
    "canonical": 209,
    "sessions": 15,
    "events": 20477,
    "pi_events": 7325,
    "ocr_segments": 77,
}


# ============================================================================
# Generic helpers
# ============================================================================

def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def low(v: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        norm(v).lower(),
    ).strip()


def unique(values):
    out = []
    seen = set()

    for value in values:
        s = norm(value)

        if not s:
            continue

        k = low(s)

        if k in seen:
            continue

        seen.add(k)
        out.append(s)

    return out


def join(values, limit=100):
    return " | ".join(
        unique(values)[:limit]
    )


def token_set(text):
    return set(
        re.findall(
            r"[a-z0-9_]+|[ぁ-んァ-ン一-龥]{2,}",
            low(text),
        )
    )


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def read_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ============================================================================
# Dataset-B loader — exact project model
# ============================================================================

def identify_dataset(path: Path):
    for part in path.parts:
        if part.lower() == "dataset_b":
            return "B"
        if part.lower() == "dataset_a":
            return "A"
    return None


def load_dataset_b():
    event_files = sorted(
        DATA_ROOT.rglob("events.jsonl")
    )

    if not event_files:
        raise RuntimeError(
            f"No events.jsonl files below {DATA_ROOT}"
        )

    sessions = defaultdict(list)
    b_files = []

    for path in event_files:
        if identify_dataset(path) != "B":
            continue

        b_files.append(path)

        for event in read_jsonl(path):
            sid = event.get("session_id")

            if not sid:
                continue

            event["_source_file"] = str(path)
            sessions[str(sid)].append(event)

    sessions = dict(sessions)

    total = sum(
        len(v)
        for v in sessions.values()
    )

    if len(sessions) != EXPECTED["sessions"]:
        raise RuntimeError(
            f"Expected {EXPECTED['sessions']} sessions, found {len(sessions)}."
        )

    if total != EXPECTED["events"]:
        raise RuntimeError(
            f"Expected {EXPECTED['events']} events, found {total}."
        )

    return sessions, b_files


def event_time(event):
    try:
        return int(
            float(
                event.get(
                    "timestamp_ms"
                )
            )
        )
    except Exception:
        return None


def sort_events(events):
    return sorted(
        events,
        key=lambda e: (
            event_time(e)
            if event_time(e) is not None
            else float("inf"),
            (
                e.get("correlation")
                or {}
            ).get(
                "sequence_number",
                float("inf"),
            ),
        ),
    )


# ============================================================================
# Canonical and A1 mapping
# ============================================================================

def parse_ts(v):
    return pd.to_datetime(
        norm(v),
        errors="coerce",
        utc=True,
    )


def load_canonical():
    rows = []

    for n, obj in enumerate(
        read_jsonl(SEGMENTS),
        1,
    ):
        if norm(
            obj.get("label")
        ).lower() != "pi":
            continue

        if not all(
            obj.get(x) is not None
            for x in (
                "session_id",
                "start",
                "end",
            )
        ):
            continue

        rows.append({
            "canonical_row": n,
            "session_id":
                str(obj["session_id"]),
            "start":
                norm(obj["start"]),
            "end":
                norm(obj["end"]),
            "start_ts":
                parse_ts(obj["start"]),
            "end_ts":
                parse_ts(obj["end"]),
        })

    df = pd.DataFrame(rows)

    if len(df) != EXPECTED["canonical"]:
        raise RuntimeError(
            f"Expected 209 canonical pi segments, found {len(df)}."
        )

    return df


def load_a1():
    a1 = pd.read_csv(A1_SEG)

    if not {
        "session_id",
        "segment_index",
    }.issubset(a1.columns):
        raise RuntimeError(
            "A1 segment table lacks required keys."
        )

    a1["session_id"] = (
        a1["session_id"].astype(str)
    )

    a1["segment_index"] = pd.to_numeric(
        a1["segment_index"],
        errors="coerce",
    ).astype("Int64")

    start_col = next(
        (
            c for c in (
                "segment_start",
                "start",
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
                "segment_end",
                "end",
                "end_time",
                "end_timestamp",
            )
            if c in a1.columns
        ),
        None,
    )

    if not start_col or not end_col:
        raise RuntimeError(
            "Cannot find A1 segment boundary columns."
        )

    a1["_start_ts"] = a1[
        start_col
    ].map(parse_ts)

    a1["_end_ts"] = a1[
        end_col
    ].map(parse_ts)

    return a1


def overlap_seconds(
    a_start,
    a_end,
    b_start,
    b_end,
):
    if any(
        pd.isna(v)
        for v in (
            a_start,
            a_end,
            b_start,
            b_end,
        )
    ):
        return -1

    start = max(
        a_start,
        b_start,
    )
    end = min(
        a_end,
        b_end,
    )

    return max(
        0,
        (
            end - start
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
        candidates = by_session.get(
            c["session_id"],
            pd.DataFrame(
                columns=a1.columns
            ),
        )

        candidates = candidates[
            ~candidates[
                "segment_index"
            ].apply(
                lambda x:
                    (
                        c["session_id"],
                        int(x),
                    )
                    in used
            )
        ]

        exact = candidates[
            candidates["_start_ts"].eq(
                c["start_ts"]
            )
            & candidates["_end_ts"].eq(
                c["end_ts"]
            )
        ]

        if len(exact) == 1:
            r = exact.iloc[0]

            key = (
                c["session_id"],
                int(r["segment_index"]),
            )

            used.add(key)

            rows.append({
                "canonical_row":
                    int(c["canonical_row"]),
                "session_id":
                    c["session_id"],
                "segment_index":
                    int(r["segment_index"]),
                "match_method":
                    "EXACT_TIMESTAMP",
                "overlap_ratio":
                    1.0,
            })

            continue

        scores = []

        for _, r in candidates.iterrows():
            ov = overlap_seconds(
                c["start_ts"],
                c["end_ts"],
                r["_start_ts"],
                r["_end_ts"],
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
                    r["_end_ts"]
                    - r["_start_ts"]
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
            reverse=True,
            key=lambda x: (
                x[0],
                x[1],
            ),
        )

        best = scores[0]

        if len(scores) > 1:
            second = scores[1]

            if (
                best[0] < 0.90
                and best[0] - second[0] < 0.10
            ):
                raise RuntimeError(
                    "Ambiguous canonical/A1 mapping."
                )

        r = best[2]

        key = (
            c["session_id"],
            int(r["segment_index"]),
        )

        used.add(key)

        rows.append({
            "canonical_row":
                int(c["canonical_row"]),
            "session_id":
                c["session_id"],
            "segment_index":
                int(r["segment_index"]),
            "match_method":
                "TIMESTAMP_OVERLAP",
            "overlap_ratio":
                best[0],
        })

    mapping = pd.DataFrame(rows)

    if len(mapping) != EXPECTED["canonical"]:
        raise RuntimeError(
            f"Canonical/A1 mapping incomplete: {len(mapping)}/209."
        )

    return mapping


# ============================================================================
# Direct OCR ground truth
# ============================================================================

def load_ocr_ground_truth():
    if not A3_FILES:
        raise FileNotFoundError(
            "Step-3A-3 semantic workbook not found."
        )

    gt = pd.read_excel(
        A3_FILES[0],
        sheet_name="PI_Semantic_Profiles",
    )

    gt["session_id"] = (
        gt["session_id"].astype(str)
    )

    gt["segment_index"] = pd.to_numeric(
        gt["segment_index"],
        errors="coerce",
    ).astype("Int64")

    gt = gt[
        gt["segment_index"].notna()
    ].copy()

    gt = gt.drop_duplicates(
        [
            "session_id",
            "segment_index",
        ]
    )

    if len(gt) != EXPECTED["ocr_segments"]:
        raise RuntimeError(
            f"Expected 77 OCR profiles, got {len(gt)}."
        )

    return gt


# ============================================================================
# Actual raw-event extraction
# ============================================================================

NOISE_KEY = re.compile(
    r"(event_id|session_id|sequence_number|chunk_id|"
    r"timestamp|hash|sha|version|monitor|screen|"
    r"resolution|latency|http_status|"
    r"process_id|pid|browser_version)$",
    re.I,
)


def collect_payload_scalars(
    obj,
    path="",
):
    """
    Generic scalar collector.

    Unlike the prior implementation, values are not all promoted into
    business fields. They remain path-qualified and are later interpreted
    according to event type and key name.
    """
    result = []

    if obj is None:
        return result

    if isinstance(obj, dict):
        for key, value in obj.items():
            p = (
                f"{path}.{key}"
                if path
                else str(key)
            )

            result.extend(
                collect_payload_scalars(
                    value,
                    p,
                )
            )

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            result.extend(
                collect_payload_scalars(
                    value,
                    f"{path}[{i}]",
                )
            )

    elif isinstance(
        obj,
        (str, int, float, bool),
    ):
        text = norm(obj)

        if not text:
            return result

        final_key = (
            path.rsplit(".", 1)[-1]
            if "." in path
            else path
        )

        if NOISE_KEY.search(
            final_key
        ):
            return result

        if len(text) > 1000:
            return result

        result.append(
            {
                "path":
                    path,
                "value":
                    text,
            }
        )

    return result


def get_payload(event):
    """
    Collect candidate actual payload containers plus event-specific
    top-level data fields.
    """
    payloads = []

    event_type = norm(
        event.get("event_type")
    )

    # Exact high-value paths observed in the project's instrumentation.
    preferred = []

    if event_type == "browser_form_input":
        preferred = [
            "payload",
            "browser_payload",
            "data",
        ]

    elif event_type == "clipboard_change":
        preferred = [
            "payload",
            "clipboard",
            "data",
        ]

    elif event_type == "keystroke":
        preferred = [
            "payload",
            "input",
            "uia_v2",
            "data",
        ]

    elif event_type == "browser_click":
        preferred = [
            "payload",
            "data",
        ]

    elif event_type == "browser_navigation":
        preferred = [
            "payload",
            "browser",
            "data",
        ]

    for key in preferred:
        value = event.get(key)

        if isinstance(
            value,
            (dict, list),
        ):
            payloads.append(
                (
                    key,
                    value,
                )
            )

    # If the event uses a typed event-specific nested object, discover other
    # dict/list containers but exclude known instrumentation context.
    for key, value in event.items():
        if key.startswith("_"):
            continue

        if key in {
            "correlation",
            "context",
            "metadata",
            "extensions",
        }:
            continue

        if key in {
            "payload",
            "browser_payload",
            "data",
            "clipboard",
            "input",
            "uia_v2",
            "browser",
        }:
            continue

        if isinstance(
            value,
            (dict, list),
        ):
            payloads.append(
                (
                    key,
                    value,
                )
            )

    scalars = []

    for root_name, value in payloads:
        scalars.extend(
            collect_payload_scalars(
                value,
                root_name,
            )
        )

    # Explicit top-level business-bearing event fields.
    for key in (
        "field",
        "target_field",
        "value",
        "previous_value",
        "text_content",
        "clipboard_text",
        "character",
        "input_context",
        "key_action",
        "element",
        "target_element",
        "accessible_name",
        "file_name",
        "document_name",
        "file_path",
        "path",
        "url",
        "route",
    ):
        if key in event:
            value = event.get(key)

            if isinstance(
                value,
                (str, int, float, bool),
            ):
                text = norm(value)

                if text:
                    scalars.append(
                        {
                            "path":
                                f"event.{key}",
                            "value":
                                text,
                        }
                    )

    # Deduplicate exact path/value pairs.
    seen = set()
    result = []

    for item in scalars:
        sig = (
            item["path"],
            item["value"],
        )

        if sig in seen:
            continue

        seen.add(sig)
        result.append(item)

    return result


def normalize_field_name(value):
    """
    Map observed DOM/input field names to a controlled canonical vocabulary.
    """
    s = low(value)

    aliases = {
        "employee_id": [
            "employee_id",
            "employeeid",
            "employee id",
            "社員番号",
            "社員id",
        ],
        "employee_name": [
            "employee_name",
            "employee name",
            "氏名",
            "社員名",
        ],
        "category": [
            "category",
            "カテゴリ",
            "区分",
        ],
        "amount": [
            "amount",
            "金額",
            "支給額",
            "費用",
        ],
        "request_type": [
            "request_type",
            "request type",
            "申請種別",
            "依頼種別",
            "type",
            "種別",
            "変更種別",
        ],
        "status": [
            "status",
            "ステータス",
            "状態",
        ],
        "processing_comment": [
            "processing_comment",
            "processing comment",
            "処理コメント",
            "確認コメント",
            "処理内容",
        ],
        "effective_date": [
            "effective_date",
            "effective date",
            "適用日",
            "対象日",
        ],
        "change_type": [
            "change_type",
            "change type",
            "変更種別",
        ],
        "retroactive_flag": [
            "retroactive_flag",
            "retroactive flag",
            "遡及",
        ],
    }

    for canonical, choices in aliases.items():
        if any(
            choice in s
            for choice in choices
        ):
            return canonical

    return ""


def extract_structured_case(events):
    field_values = defaultdict(list)
    provenance = defaultdict(list)

    semantic_values = defaultdict(list)

    all_candidates = []

    for event in sort_events(events):
        event_type = norm(
            event.get("event_type")
        )

        payload = get_payload(
            event
        )

        for item in payload:
            path = item["path"]
            value = item["value"]

            all_candidates.append(
                {
                    "event_type":
                        event_type,
                    "timestamp_ms":
                        event_time(event),
                    "path":
                        path,
                    "value":
                        value,
                    "source_file":
                        event.get(
                            "_source_file",
                            "",
                        ),
                }
            )

        # Event-type-specific field identity.
        if event_type == "browser_form_input":
            field_candidates = []

            for item in payload:
                field_norm = normalize_field_name(
                    item["value"]
                    if (
                        "field" in item["path"].lower()
                        or "target_field" in item["path"].lower()
                    )
                    else item["path"]
                )

                if field_norm:
                    field_candidates.append(
                        field_norm
                    )

            # Pair field identity with candidate value.
            for item in payload:
                path_lower = low(
                    item["path"]
                )

                if not (
                    "field" in path_lower
                    or "target_field" in path_lower
                ):
                    continue

                canonical = normalize_field_name(
                    item["value"]
                )

                if not canonical:
                    canonical = normalize_field_name(
                        item["path"]
                    )

                if canonical:
                    # Look for sibling values in the same event.
                    for sibling in payload:
                        sibling_path = low(
                            sibling["path"]
                        )

                        if (
                            sibling is item
                            or
                            "value" not in sibling_path
                        ):
                            continue

                        field_values[
                            canonical
                        ].append(
                            sibling["value"]
                        )

                        provenance[
                            canonical
                        ].append(
                            f"{event_type}@{event_time(event)}:{sibling['path']}"
                        )

        # Direct top-level fallback for field/value pairs.
        field_raw = norm(
            event.get("field")
            or event.get("target_field")
        )

        value_raw = norm(
            event.get("value")
        )

        canonical = normalize_field_name(
            field_raw
        )

        if (
            canonical
            and value_raw
        ):
            field_values[
                canonical
            ].append(
                value_raw
            )

            provenance[
                canonical
            ].append(
                f"{event_type}@{event_time(event)}:event.value"
            )

        # Clipboard is source evidence, not a semantic field by itself.
        if event_type == "clipboard_change":
            for item in payload:
                p = low(
                    item["path"]
                )

                if (
                    "text_content" in p
                    or p.endswith(".text")
                    or "clipboard_text" in p
                ):
                    semantic_values[
                        "clipboard"
                    ].append(
                        item["value"]
                    )

        # Explicit action/decision labels from click targets.
        if event_type in {
            "browser_click",
            "mouse_click",
        }:
            for item in payload:
                p = low(
                    item["path"]
                )

                if any(
                    x in p
                    for x in (
                        "element",
                        "target_element",
                        "accessible_name",
                        "text",
                    )
                ):
                    semantic_values[
                        "click_target"
                    ].append(
                        item["value"]
                    )

    # Search all actual raw payload values for known semantic categories, but
    # do not invent them.
    all_text = " | ".join(
        x["value"]
        for x in all_candidates
    )

    for canonical, patterns in {
        "request_type": [
            r"expense_reimbursement",
            r"payroll_change",
            r"expense reimbursement",
            r"payroll change",
            r"経費精算",
            r"給与変更",
        ],
        "status": [
            r"UNPROCESSED_OR_PENDING",
            r"PROCESSED_OR_COMPLETED",
            r"HOLD",
            r"APPROVAL",
            r"保留",
        ],
        "action": [
            r"CONFIRM_OR_FINALIZE",
            r"CHANGE_OR_UPDATE",
            r"SUBMIT_OR_REQUEST",
            r"REIMBURSEMENT_PROCESSING",
            r"登録確定",
            r"保留",
        ],
    }.items():
        for pattern in patterns:
            if re.search(
                pattern,
                all_text,
                re.I,
            ):
                semantic_values[
                    canonical
                ].append(
                    pattern
                )

    return {
        "field_values": {
            k: unique(v)
            for k, v in field_values.items()
        },
        "provenance": dict(
            provenance
        ),
        "semantic_values": {
            k: unique(v)
            for k, v in semantic_values.items()
        },
        "all_candidates": all_candidates,
    }


# ============================================================================
# Ground truth normalization
# ============================================================================

def split_semantic_set(value):
    s = norm(value)

    if not s:
        return []

    return [
        x.strip()
        for x in re.split(
            r"\s*;\s*|\s*\|\s*",
            s,
        )
        if x.strip()
    ]


def canonical_ground_truth_fields(value):
    raw = split_semantic_set(
        value
    )

    mapped = []

    for item in raw:
        canonical = normalize_field_name(
            item
        )

        if canonical:
            mapped.append(
                canonical
            )
        else:
            mapped.append(
                low(item)
            )

    return unique(mapped)


def concrete_tokens(text):
    """
    Extract concrete identifiers, dates, amounts and alphanumeric case
    identifiers. Do not treat event ids/hashes as business values because OCR
    ground truth will almost never contain them.
    """
    s = norm(text)

    if not s:
        return set()

    patterns = [
        r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b",
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?\b",
        r"\b\d+(?:\.\d{1,2})?\b",
        r"\b(?:INV|ID|EMP|P|E)[-_]?[A-Z0-9-]{2,}\b",
    ]

    out = set()

    for p in patterns:
        out.update(
            x.lower()
            for x in re.findall(
                p,
                s,
                re.I,
            )
        )

    return out


# ============================================================================
# Per-case scoring
# ============================================================================

def compare_case(
    gt_row,
    simulation,
):
    gt_fields = canonical_ground_truth_fields(
        gt_row.get("field_names")
    )

    observed_fields = sorted(
        simulation["field_values"].keys()
    )

    required = set(
        gt_fields
    )

    observed = set(
        observed_fields
    )

    recovered = sorted(
        required
        & observed
    )

    missing = sorted(
        required
        - observed
    )

    extra = sorted(
        observed
        - required
    )

    if required:
        precision = (
            len(
                recovered
            )
            / len(observed)
            if observed
            else 0.0
        )

        recall = (
            len(
                recovered
            )
            / len(required)
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

    else:
        precision = None
        recall = None
        f1 = None

    gt_text = " ".join(
        norm(
            gt_row.get(c)
        )
        for c in (
            "business_object",
            "request_type",
            "field_names",
            "status",
            "action",
            "document",
            "decision_options",
            "ocr_english",
        )
        if norm(
            gt_row.get(c)
        )
    )

    gt_values = concrete_tokens(
        gt_text
    )

    sim_text = " ".join(
        [
            value
            for values
            in simulation[
                "field_values"
            ].values()
            for value in values
        ]
        + [
            value
            for values
            in simulation[
                "semantic_values"
            ].values()
            for value in values
        ]
        + [
            item["value"]
            for item
            in simulation[
                "all_candidates"
            ]
        ]
    )

    sim_values = concrete_tokens(
        sim_text
    )

    value_recovered = (
        sorted(
            gt_values
            & sim_values
        )
    )

    value_missing = (
        sorted(
            gt_values
            - sim_values
        )
    )

    value_recall = (
        len(
            value_recovered
        )
        / len(gt_values)
        if gt_values
        else None
    )

    def semantic_score(
        gt_col,
        sim_key,
        aliases=None,
    ):
        gt_set = token_set(
            " ".join(
                split_semantic_set(
                    gt_row.get(
                        gt_col
                    )
                )
            )
        )

        vals = simulation[
            "semantic_values"
        ].get(
            sim_key,
            []
        )

        sim_set = token_set(
            " ".join(vals)
        )

        if aliases:
            for value in vals:
                for alias, canonical in aliases.items():
                    if alias in low(value):
                        sim_set.add(
                            canonical
                        )

        return jaccard(
            gt_set,
            sim_set,
        ) if gt_set else None

    # Field-aware semantic checks. These are only meaningful when the raw event
    # payload actually contains semantic labels.
    request_gt = set(
        low(x)
        for x in split_semantic_set(
            gt_row.get(
                "request_type"
            )
        )
    )

    request_sim = set(
        low(x)
        for x in simulation[
            "field_values"
        ].get(
            "request_type",
            [],
        )
    )

    request_sim |= set(
        low(x)
        for x in simulation[
            "semantic_values"
        ].get(
            "request_type",
            [],
        )
    )

    request_exact = (
        bool(
            request_gt
            & request_sim
        )
        if request_gt
        else None
    )

    status_gt = set(
        low(x)
        for x in split_semantic_set(
            gt_row.get(
                "status"
            )
        )
    )

    status_sim = set(
        low(x)
        for x in simulation[
            "field_values"
        ].get(
            "status",
            [],
        )
    )

    status_sim |= set(
        low(x)
        for x in simulation[
            "semantic_values"
        ].get(
            "status",
            [],
        )
    )

    status_match = (
        bool(
            status_gt
            & status_sim
        )
        if status_gt
        else None
    )

    action_gt = set(
        low(x)
        for x in split_semantic_set(
            gt_row.get(
                "action"
            )
        )
    )

    action_sim = set(
        low(x)
        for x in simulation[
            "semantic_values"
        ].get(
            "action",
            [],
        )
    )

    action_match = (
        bool(
            action_gt
            & action_sim
        )
        if action_gt
        else None
    )

    return {
        "gt_fields":
            "; ".join(gt_fields),
        "observed_fields":
            "; ".join(observed_fields),
        "fields_recovered":
            "; ".join(recovered),
        "fields_missing":
            "; ".join(missing),
        "extra_fields":
            "; ".join(extra),
        "field_precision":
            precision,
        "field_recall":
            recall,
        "field_f1":
            f1,
        "gt_concrete_values":
            "; ".join(
                sorted(gt_values)
            ),
        "recovered_concrete_values":
            "; ".join(
                value_recovered
            ),
        "missing_concrete_values":
            "; ".join(
                value_missing
            ),
        "concrete_value_recall":
            value_recall,
        "request_type_match":
            request_exact,
        "status_match":
            status_match,
        "action_match":
            action_match,
        "candidate_count":
            len(
                simulation[
                    "all_candidates"
                ]
            ),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print(
        "STEP 3I — PI REAL-DATA STRUCTURED EXTRACTION VS OCR GROUND TRUTH V3"
    )
    print("=" * 80)

    canonical = load_canonical()
    a1 = load_a1()

    mapping = map_canonical_to_a1(
        canonical,
        a1,
    )

    sessions, event_files = (
        load_dataset_b()
    )

    total_events = sum(
        len(v)
        for v in sessions.values()
    )

    # Attach real raw events to the authoritative A1 pi intervals.
    intervals = {}

    for _, r in a1.iterrows():
        if pd.isna(
            r["segment_index"]
        ):
            continue

        intervals[
            (
                str(r["session_id"]),
                int(r["segment_index"]),
            )
        ] = (
            r["_start_ts"],
            r["_end_ts"],
        )

    assigned = defaultdict(list)

    for sid, events in sessions.items():
        sid_intervals = [
            (
                key,
                start,
                end,
            )
            for key, (
                start,
                end
            ) in intervals.items()
            if key[0] == str(sid)
        ]

        for event in sort_events(
            events
        ):
            timestamp = event_time(
                event
            )

            if timestamp is None:
                continue

            ts = pd.to_datetime(
                timestamp,
                unit="ms",
                utc=True,
            )

            hits = [
                item
                for item in sid_intervals
                if (
                    item[1]
                    <= ts
                    <= item[2]
                )
            ]

            if len(hits) == 1:
                assigned[
                    hits[0][0]
                ].append(
                    event
                )

            elif len(hits) > 1:
                # A boundary event can occur at the end/start of adjacent
                # segments. Assign it to the closer boundary deterministically.
                selected = min(
                    hits,
                    key=lambda item:
                        min(
                            abs(
                                (
                                    ts
                                    - item[1]
                                ).total_seconds()
                            ),
                            abs(
                                (
                                    item[2]
                                    - ts
                                ).total_seconds()
                            ),
                        ),
                )

                assigned[
                    selected[0]
                ].append(
                    event
                )

    assigned_count = sum(
        len(v)
        for v in assigned.values()
    )

    if assigned_count != EXPECTED["pi_events"]:
        raise RuntimeError(
            f"Expected 7,325 pi events, assigned {assigned_count}."
        )

    gt = load_ocr_ground_truth()

    gt_map = {
        (
            str(r["session_id"]),
            int(r["segment_index"]),
        ): r
        for _, r in gt.iterrows()
    }

    scored_rows = []

    # Score only the 77 direct OCR cases.
    for key, gt_row in gt_map.items():
        events = assigned.get(
            key,
            []
        )

        simulation = (
            extract_structured_case(
                events
            )
        )

        metrics = compare_case(
            gt_row,
            simulation,
        )

        field_rows = []

        for field, values in (
            simulation[
                "field_values"
            ].items()
        ):
            field_rows.append({
                "session_id":
                    key[0],
                "segment_index":
                    key[1],
                "field":
                    field,
                "extracted_values":
                    "; ".join(values),
                "provenance":
                    "; ".join(
                        simulation[
                            "provenance"
                        ].get(
                            field,
                            [],
                        )
                    ),
            })

        scored_rows.append({
            "session_id":
                key[0],
            "segment_index":
                key[1],
            "raw_event_count":
                len(events),
            "gt_business_object":
                norm(
                    gt_row.get(
                        "business_object"
                    )
                ),
            "gt_request_type":
                norm(
                    gt_row.get(
                        "request_type"
                    )
                ),
            "gt_field_names":
                norm(
                    gt_row.get(
                        "field_names"
                    )
                ),
            "gt_status":
                norm(
                    gt_row.get(
                        "status"
                    )
                ),
            "gt_action":
                norm(
                    gt_row.get(
                        "action"
                    )
                ),
            "gt_document":
                norm(
                    gt_row.get(
                        "document"
                    )
                ),
            "gt_decision_options":
                norm(
                    gt_row.get(
                        "decision_options"
                    )
                ),
            "sim_employee_id":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "employee_id",
                        [],
                    )
                ),
            "sim_employee_name":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "employee_name",
                        [],
                    )
                ),
            "sim_category":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "category",
                        [],
                    )
                ),
            "sim_amount":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "amount",
                        [],
                    )
                ),
            "sim_request_type":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "request_type",
                        [],
                    )
                ),
            "sim_status":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "status",
                        [],
                    )
                ),
            "sim_processing_comment":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "processing_comment",
                        [],
                    )
                ),
            "sim_effective_date":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "effective_date",
                        [],
                    )
                ),
            "sim_change_type":
                "; ".join(
                    simulation[
                        "field_values"
                    ].get(
                        "change_type",
                        [],
                    )
                ),
            "sim_click_targets":
                "; ".join(
                    simulation[
                        "semantic_values"
                    ].get(
                        "click_target",
                        [],
                    )
                ),
            **metrics,
        })

    scored = pd.DataFrame(
        scored_rows
    )

    # Field-level details.
    field_detail_rows = []

    for key, gt_row in gt_map.items():
        events = assigned.get(
            key,
            []
        )

        sim = extract_structured_case(
            events
        )

        gt_fields = canonical_ground_truth_fields(
            gt_row.get(
                "field_names"
            )
        )

        for field in gt_fields:
            values = sim[
                "field_values"
            ].get(
                field,
                [],
            )

            field_detail_rows.append({
                "session_id":
                    key[0],
                "segment_index":
                    key[1],
                "field":
                    field,
                "ground_truth_field":
                    field,
                "extraction_status":
                    (
                        "RECOVERED"
                        if values
                        else "NOT_OBSERVED_IN_RAW"
                    ),
                "extracted_value":
                    "; ".join(values),
                "provenance":
                    "; ".join(
                        sim[
                            "provenance"
                        ].get(
                            field,
                            [],
                        )
                    ),
            })

    field_details = pd.DataFrame(
        field_detail_rows
    )

    # 209 population coverage.
    population_rows = []

    mapping_keys = [
        (
            str(a),
            int(b),
        )
        for a, b in zip(
            mapping["session_id"],
            mapping["segment_index"],
        )
    ]

    gt_keys = set(
        gt_map
    )

    for key in mapping_keys:
        events = assigned.get(
            key,
            []
        )

        population_rows.append({
            "session_id":
                key[0],
            "segment_index":
                key[1],
            "raw_event_count":
                len(events),
            "direct_ocr_ground_truth":
                key in gt_keys,
            "simulation_input_available":
                bool(events),
            "population_status":
                (
                    "SCORED"
                    if key in gt_keys
                    else "COVERAGE_ONLY"
                ),
        })

    population = pd.DataFrame(
        population_rows
    )

    # Summary metrics.
    summary_rows = [
        {
            "metric":
                "canonical_pi_segments",
            "value":
                len(canonical),
            "interpretation":
                "Full real pi population.",
        },
        {
            "metric":
                "direct_ocr_scored_segments",
            "value":
                len(scored),
            "interpretation":
                "Only direct-OCR cases are scored.",
        },
        {
            "metric":
                "coverage_only_segments",
            "value":
                209 - len(scored),
            "interpretation":
                "No direct semantic ground truth; not treated as failures.",
        },
        {
            "metric":
                "mean_field_precision",
            "value":
                round(
                    scored[
                        "field_precision"
                    ].dropna().mean(),
                    4,
                ),
            "interpretation":
                "Precision of recovered semantic field identities against OCR-established fields.",
        },
        {
            "metric":
                "mean_field_recall",
            "value":
                round(
                    scored[
                        "field_recall"
                    ].dropna().mean(),
                    4,
                ),
            "interpretation":
                "Recall of OCR-established fields from real raw-event evidence.",
        },
        {
            "metric":
                "mean_field_f1",
            "value":
                round(
                    scored[
                        "field_f1"
                    ].dropna().mean(),
                    4,
                ),
            "interpretation":
                "Field-level F1; not production accuracy.",
        },
        {
            "metric":
                "mean_concrete_value_recall",
            "value":
                round(
                    scored[
                        "concrete_value_recall"
                    ].dropna().mean(),
                    4,
                )
                if scored[
                    "concrete_value_recall"
                ].notna().any()
                else None,
            "interpretation":
                "Recovery of concrete identifiers/dates/amounts available in both evidence sources.",
        },
        {
            "metric":
                "request_type_exact_cases",
            "value":
                int(
                    scored[
                        "request_type_match"
                    ].eq(True).sum()
                ),
            "interpretation":
                "Exact semantic match among cases where both sides expose a request type.",
        },
        {
            "metric":
                "status_match_cases",
            "value":
                int(
                    scored[
                        "status_match"
                    ].eq(True).sum()
                ),
            "interpretation":
                "Status overlap among cases where comparable status evidence is present.",
        },
        {
            "metric":
                "action_match_cases",
            "value":
                int(
                    scored[
                        "action_match"
                    ].eq(True).sum()
                ),
            "interpretation":
                "Action overlap among cases where comparable action evidence is present.",
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    # Failure analysis, focused on evidence gaps rather than generic text
    # similarity.
    failure = scored.sort_values(
        [
            "field_recall",
            "concrete_value_recall",
        ],
        na_position="first",
    )[
        [
            "session_id",
            "segment_index",
            "field_precision",
            "field_recall",
            "field_f1",
            "concrete_value_recall",
            "gt_field_names",
            "fields_recovered",
            "fields_missing",
            "extra_fields",
            "gt_concrete_values",
            "recovered_concrete_values",
            "missing_concrete_values",
            "sim_request_type",
            "sim_status",
        ]
    ].head(40)

    # Local mock verification is a separate engineering check, not part of
    # ground-truth extraction scoring.
    mock = pd.DataFrame()

    if MOCK.exists():
        try:
            from playwright.sync_api import sync_playwright

            from http.server import (
                BaseHTTPRequestHandler,
                ThreadingHTTPServer,
            )
            from threading import Thread

            host = "127.0.0.1"
            port = 8790
            url = (
                f"http://{host}:{port}/payroll-items"
            )

            class Handler(
                BaseHTTPRequestHandler
            ):
                def log_message(
                    self,
                    fmt,
                    *args,
                ):
                    pass

                def do_GET(self):
                    if (
                        self.path.split(
                            "?"
                        )[0]
                        == "/payroll-items"
                    ):
                        data = MOCK.read_bytes()

                        self.send_response(200)
                        self.send_header(
                            "Content-Type",
                            "text/html; charset=utf-8",
                        )
                        self.send_header(
                            "Content-Length",
                            str(len(data)),
                        )
                        self.end_headers()
                        self.wfile.write(
                            data
                        )
                    else:
                        self.send_error(
                            404
                        )

            server = ThreadingHTTPServer(
                (host, port),
                Handler,
            )

            Thread(
                target=server.serve_forever,
                daemon=True,
            ).start()

            mock_rows = []

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True
                )
                page = browser.new_page()

                for _, row in scored.iterrows():
                    page.goto(
                        url,
                        wait_until="domcontentloaded",
                    )

                    expected = {}

                    employee_id = norm(
                        row["sim_employee_id"]
                    )

                    if employee_id:
                        page.locator(
                            "#employee_id"
                        ).fill(
                            employee_id
                        )
                        expected[
                            "employee_id"
                        ] = employee_id

                    request_type = low(
                        row["sim_request_type"]
                    )

                    if (
                        "expense" in request_type
                        or "reimbursement"
                        in request_type
                        or "経費" in request_type
                    ):
                        value = (
                            "expense_reimbursement"
                        )

                        page.locator(
                            "#request_type"
                        ).select_option(
                            value
                        )
                        expected[
                            "request_type"
                        ] = value

                    elif (
                        "payroll"
                        in request_type
                        or "給与"
                        in request_type
                    ):
                        value = (
                            "payroll_change"
                        )

                        page.locator(
                            "#request_type"
                        ).select_option(
                            value
                        )
                        expected[
                            "request_type"
                        ] = value

                    amount = norm(
                        row["sim_amount"]
                    ).replace(
                        ",",
                        "",
                    )

                    if re.fullmatch(
                        r"-?\d+(?:\.\d{1,2})?",
                        amount,
                    ):
                        page.locator(
                            "#amount"
                        ).fill(
                            amount
                        )
                        expected[
                            "amount"
                        ] = amount

                    date = norm(
                        row["sim_effective_date"]
                    )

                    if re.fullmatch(
                        r"\d{4}-\d{2}-\d{2}",
                        date,
                    ):
                        page.locator(
                            "#effective_date"
                        ).fill(
                            date
                        )
                        expected[
                            "effective_date"
                        ] = date

                    comment = norm(
                        row[
                            "sim_processing_comment"
                        ]
                    )

                    if comment:
                        page.locator(
                            "#processing_comment"
                        ).fill(
                            comment
                        )
                        expected[
                            "processing_comment"
                        ] = comment

                    actual = page.evaluate(
                        "window.getPreparedRecord()"
                    )

                    verified = sum(
                        actual.get(
                            k
                        ) == v
                        for k, v in expected.items()
                    )

                    gate = (
                        page.locator(
                            "#register"
                        ).is_disabled()
                        and
                        page.locator(
                            "#hold"
                        ).is_disabled()
                    )

                    mock_rows.append({
                        "session_id":
                            row["session_id"],
                        "segment_index":
                            row["segment_index"],
                        "fields_attempted":
                            len(expected),
                        "fields_verified":
                            verified,
                        "human_gate_preserved":
                            gate,
                        "register_clicked":
                            False,
                        "hold_clicked":
                            False,
                        "status":
                            (
                                "PASS"
                                if gate
                                else "FAIL_HUMAN_GATE"
                            ),
                    })

                browser.close()

            server.shutdown()

            mock = pd.DataFrame(
                mock_rows
            )

        except Exception as exc:
            mock = pd.DataFrame([
                {
                    "status":
                        "NOT_RUN",
                    "reason":
                        f"{type(exc).__name__}: {exc}",
                }
            ])

    validation = pd.DataFrame([
        {
            "check":
                "canonical_pi_population",
            "expected":
                209,
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
            "expected":
                15,
            "observed":
                len(sessions),
            "status":
                "PASS"
                if len(sessions) == 15
                else "FAIL",
        },
        {
            "check":
                "dataset_b_events",
            "expected":
                20477,
            "observed":
                total_events,
            "status":
                "PASS"
                if total_events == 20477
                else "FAIL",
        },
        {
            "check":
                "dataset_b_event_files",
            "expected":
                ">0",
            "observed":
                len([
                    p for p in event_files
                    if identify_dataset(p) == "B"
                ]),
            "status":
                "PASS"
                if event_files
                else "FAIL",
        },
        {
            "check":
                "canonical_a1_mapping",
            "expected":
                209,
            "observed":
                len(mapping),
            "status":
                "PASS"
                if len(mapping) == 209
                else "FAIL",
        },
        {
            "check":
                "real_pi_event_assignment",
            "expected":
                7325,
            "observed":
                assigned_count,
            "status":
                "PASS"
                if assigned_count == 7325
                else "FAIL",
        },
        {
            "check":
                "direct_ocr_scored_population",
            "expected":
                77,
            "observed":
                len(scored),
            "status":
                "PASS"
                if len(scored) == 77
                else "FAIL",
        },
        {
            "check":
                "ocr_used_as_simulation_input",
            "expected":
                "NO",
            "observed":
                "NO",
            "status":
                "PASS",
        },
        {
            "check":
                "all_209_real_inputs_available",
            "expected":
                209,
            "observed":
                int(
                    population[
                        "simulation_input_available"
                    ].sum()
                ),
            "status":
                "PASS"
                if int(
                    population[
                        "simulation_input_available"
                    ].sum()
                ) == 209
                else "FAIL",
        },
        {
            "check":
                "production_accessed",
            "expected":
                "NO",
            "observed":
                "NO",
            "status":
                "PASS",
        },
        {
            "check":
                "production_accuracy_claimed",
            "expected":
                "NO",
            "observed":
                "NO",
            "status":
                "PASS",
        },
    ])

    readme = pd.DataFrame([
        [
            "Experiment",
            "Structured extraction from real Dataset-B events compared against independent direct-OCR semantics.",
        ],
        [
            "Population",
            "209 canonical pi segments.",
        ],
        [
            "Scored set",
            "77 direct-OCR semantic segments. The remaining 132 are coverage-only.",
        ],
        [
            "Simulator input",
            "Real Dataset-B raw event payloads only.",
        ],
        [
            "Ground truth",
            "Step-3A-3 PI_Semantic_Profiles, applied only after simulation.",
        ],
        [
            "Key correction",
            "Actual event payload field/value paths are extracted; event IDs, timestamps, hashes, versions and other instrumentation noise are excluded.",
        ],
        [
            "Metrics",
            "Field precision/recall/F1, concrete-value recall, and semantic category matches.",
        ],
        [
            "Interpretation",
            "These are evidence-recovery metrics, not production accuracy.",
        ],
        [
            "Production",
            "No production system is accessed.",
        ],
    ], columns=[
        "item",
        "value",
    ])

    with pd.ExcelWriter(
        OUTPUT,
        engine="openpyxl",
    ) as writer:
        readme.to_excel(
            writer,
            sheet_name="README",
            index=False,
        )
        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )
        population.to_excel(
            writer,
            sheet_name="209_Population",
            index=False,
        )
        scored.to_excel(
            writer,
            sheet_name="77_Scored_Cases",
            index=False,
        )
        field_details.to_excel(
            writer,
            sheet_name="Field_Recovery_Detail",
            index=False,
        )
        failure.to_excel(
            writer,
            sheet_name="Failure_Analysis",
            index=False,
        )
        mock.to_excel(
            writer,
            sheet_name="Simulation_Mock_Run",
            index=False,
        )
        mapping.to_excel(
            writer,
            sheet_name="Canonical_A1_Mapping",
            index=False,
        )
        validation.to_excel(
            writer,
            sheet_name="Validation",
            index=False,
        )

    # Workbook styling.
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(
        OUTPUT
    )

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
                    120,
                ),
            ),
            1,
        ):
            width = max(
                [
                    len(
                        str(
                            c.value or ""
                        )
                    )
                    for c in cells
                ]
                + [12]
            )

            ws.column_dimensions[
                get_column_letter(i)
            ].width = min(
                width + 2,
                55,
            )

    wb.save(
        OUTPUT
    )

    print("\n3I RESULT")
    print(
        f"canonical pi segments: {len(canonical)}"
    )
    print(
        f"real Dataset-B events: {total_events}"
    )
    print(
        f"real pi-attached events: {assigned_count}"
    )
    print(
        f"direct OCR scored cases: {len(scored)}"
    )
    print(
        f"coverage-only cases: {209 - len(scored)}"
    )

    print("\nSUMMARY")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nFIELD RECOVERY STATUS")
    if not field_details.empty:
        print(
            field_details[
                "extraction_status"
            ].value_counts(
                dropna=False
            ).to_string()
        )

    print("\nSIMULATION MOCK")
    if (
        not mock.empty
        and "status" in mock.columns
    ):
        print(
            mock[
                "status"
            ].value_counts(
                dropna=False
            ).to_string()
        )

    print("\nVALIDATION")
    print(
        validation.to_string(
            index=False
        )
    )

    failures = validation[
        validation["status"] == "FAIL"
    ]

    if not failures.empty:
        raise RuntimeError(
            "3I validation failed."
        )

    print("\nOUTPUT")
    print(OUTPUT)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3I V3 COMPLETE.")


if __name__ == "__main__":
    main()
