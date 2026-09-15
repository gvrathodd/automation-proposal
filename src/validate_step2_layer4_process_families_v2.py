
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
FINAL_SEGMENTS = PROJECT_ROOT / "segments_v2.jsonl"
OUT = PROJECT_ROOT / "outputs"

EVENT_FILE_AUDIT = OUT / "step2_layer1_dataset_b_chunk_audit.csv"
SEGMENT_EVIDENCE = OUT / "step2_layer3_authoritative_segment_evidence.csv"

SUMMARY_OUT = OUT / "step2_layer4_process_family_summary_v2.csv"
TOKEN_OUT = OUT / "step2_layer4_prefix_token_evidence_v2.csv"
SEGMENT_OUT = OUT / "step2_layer4_segment_prefix_evidence_v2.csv"
VARIANT_OUT = OUT / "step2_layer4_prefix_by_variant_v2.csv"

PREFIXES = ["pi", "la", "ob", "si", "rt"]

PREFIX_PATTERNS = {
    prefix: re.compile(
        rf"(?<![A-Za-z0-9]){prefix}-(?:note|ok)(?![A-Za-z0-9])",
        re.I,
    )
    for prefix in PREFIXES
}


def recursive_find(obj, keys):
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                found.append(value)
            found.extend(recursive_find(value, keys))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(recursive_find(value, keys))

    return found


def flatten_strings(obj):
    values = []

    if isinstance(obj, str):
        value = obj.strip()
        if value:
            values.append(value)
    elif isinstance(obj, dict):
        for value in obj.values():
            values.extend(flatten_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(flatten_strings(value))

    return values


def extract_operator(event):
    vals = recursive_find(
        event,
        {"username_hash", "user_hash", "username"},
    )

    for value in vals:
        if isinstance(value, (str, int, float)):
            value = str(value).strip()
            if value:
                return value
    return ""


def extract_dom_values(event):
    vals = recursive_find(
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

    values = set()
    for value in vals:
        values.update(flatten_strings(value))

    return values


def extract_apps(event):
    vals = recursive_find(
        event,
        {
            "app_name",
            "application",
            "process_name",
            "new_app",
            "previous_app",
        },
    )

    values = set()
    for value in vals:
        values.update(flatten_strings(value))

    return values


def extract_windows(event):
    vals = recursive_find(
        event,
        {"window_title", "title"},
    )

    values = set()
    for value in vals:
        values.update(flatten_strings(value))

    return values


def extract_urls(event):
    vals = recursive_find(
        event,
        {"url", "href", "current_url", "browser_url"},
    )

    values = set()
    for value in vals:
        for item in flatten_strings(value):
            if (
                item.startswith("http://")
                or item.startswith("https://")
                or item.startswith("file://")
                or "/#" in item
            ):
                values.add(item)

    return values


def detect_prefix_from_text(text):
    hits = set()
    for prefix, pattern in PREFIX_PATTERNS.items():
        if pattern.search(text):
            hits.add(prefix)
    return hits


def infer_variant(apps):
    text = " | ".join(sorted(x.lower() for x in apps))
    if "word" in text:
        return "word_assisted"
    if "notepad" in text:
        return "notepad_assisted"
    if "excel" in text:
        return "excel_assisted"
    return "browser_only"


def load_segment_evidence():
    if not SEGMENT_EVIDENCE.exists():
        raise FileNotFoundError(SEGMENT_EVIDENCE)

    df = pd.read_csv(
        SEGMENT_EVIDENCE,
        keep_default_na=False,
    )

    df["start"] = pd.to_datetime(
        df["start"],
        utc=True,
        errors="raise",
    )
    df["end"] = pd.to_datetime(
        df["end"],
        utc=True,
        errors="raise",
    )

    df["segment_key"] = (
        df["session_id"].astype(str)
        + "::"
        + df["segment_index"].astype(str)
    )

    return df


def discover_event_files():
    files = []

    if EVENT_FILE_AUDIT.exists():
        audit = pd.read_csv(
            EVENT_FILE_AUDIT,
            keep_default_na=False,
        )

        if "events_file" in audit.columns:
            for value in audit["events_file"]:
                path = Path(str(value))
                if path.exists():
                    files.append(path)

    if files:
        return sorted(set(files), key=str)

    for dataset_b in DATA_ROOT.rglob("dataset_b"):
        if dataset_b.is_dir():
            files.extend(dataset_b.rglob("events.jsonl"))

    return sorted(set(files), key=str)


def parse_event_timestamp(event):
    for key in (
        "timestamp",
        "timestamp_ms",
        "timestamp_iso",
        "ts",
        "time",
    ):
        if key not in event:
            continue

        value = event[key]

        try:
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

            return pd.to_datetime(
                value,
                utc=True,
            )
        except Exception:
            continue

    return None


def build_event_records(segments):
    event_files = discover_event_files()

    if len(event_files) != 20:
        raise RuntimeError(
            f"Expected 20 event files; found {len(event_files)}"
        )

    segments_by_session = {}

    for sid, group in segments.groupby("session_id"):
        # Explicitly normalize timestamps before comparisons.
        group = group.copy()
        group["start"] = pd.to_datetime(
            group["start"],
            utc=True,
            errors="raise",
        )
        group["end"] = pd.to_datetime(
            group["end"],
            utc=True,
            errors="raise",
        )
        segments_by_session[sid] = group.sort_values(
            "start"
        ).reset_index(drop=True)

    assigned_rows = []

    for i, event_file in enumerate(event_files, start=1):
        session_id = event_file.parent.parent.name
        chunk_id = event_file.parent.name

        print(
            f"Scanning {i}/{len(event_files)}: "
            f"{session_id} / {chunk_id}"
        )

        session_segments = segments_by_session.get(
            session_id,
            pd.DataFrame(),
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
                    event = json.loads(line)
                except Exception:
                    continue

                ts = parse_event_timestamp(event)
                if ts is None or session_segments.empty:
                    continue

                matched_idx = None

                # Deterministic interval ownership.
                for j, seg in session_segments.iterrows():
                    is_last = (
                        j == len(session_segments) - 1
                    )

                    if (
                        seg["start"] <= ts < seg["end"]
                    ) or (
                        is_last
                        and seg["start"] <= ts <= seg["end"]
                    ):
                        matched_idx = j
                        break

                if matched_idx is None:
                    continue

                seg = session_segments.iloc[matched_idx]

                dom_values = extract_dom_values(event)
                prefix_hits = set()

                for value in dom_values:
                    prefix_hits.update(
                        detect_prefix_from_text(value)
                    )

                apps = extract_apps(event)

                assigned_rows.append(
                    {
                        "segment_key": seg["segment_key"],
                        "session_id": session_id,
                        "segment_index": int(
                            seg["segment_index"]
                        ),
                        "event_type": str(
                            event.get("event_type", "")
                        ),
                        "layer": str(
                            event.get("layer", "")
                        ),
                        "operator_hash": extract_operator(event),
                        "apps": apps,
                        "windows": extract_windows(event),
                        "urls": extract_urls(event),
                        "dom_values": dom_values,
                        "prefix_hits": prefix_hits,
                    }
                )

    return assigned_rows


def main():
    print("=" * 78)
    print(
        "STEP 2 — LAYER 4 PROCESS-FAMILY VALIDATION V2"
    )
    print("=" * 78)

    OUT.mkdir(parents=True, exist_ok=True)

    segments = load_segment_evidence()

    if len(segments) != 540:
        raise RuntimeError(
            f"Expected 540 segment evidence rows; found {len(segments)}"
        )

    print(f"Segments: {len(segments):,}")

    rows = build_event_records(segments)

    print()
    print(f"Assigned event records: {len(rows):,}")

    prefix_token_events = Counter()
    prefix_token_segments = defaultdict(set)
    prefix_by_event_type = defaultdict(Counter)
    prefix_by_layer = defaultdict(Counter)
    prefix_sessions = defaultdict(set)
    prefix_operators = defaultdict(set)
    prefix_apps = defaultdict(Counter)
    prefix_variants = defaultdict(Counter)

    seg_prefix_event_count = Counter()
    seg_prefix_dom_values = defaultdict(set)

    for row in rows:
        prefixes = row["prefix_hits"]
        if not prefixes:
            continue

        variant = infer_variant(row["apps"])

        for prefix in prefixes:
            prefix_token_events[prefix] += 1
            prefix_token_segments[prefix].add(
                row["segment_key"]
            )
            prefix_by_event_type[prefix][
                row["event_type"]
            ] += 1
            prefix_by_layer[prefix][
                row["layer"]
            ] += 1
            prefix_sessions[prefix].add(
                row["session_id"]
            )

            operator = row["operator_hash"]
            if operator:
                prefix_operators[prefix].add(
                    operator
                )

            for app in row["apps"]:
                prefix_apps[prefix][app] += 1

            prefix_variants[prefix][variant] += 1

            seg_key = row["segment_key"]

            seg_prefix_event_count[
                (seg_key, prefix)
            ] += 1

            seg_prefix_dom_values[
                (seg_key, prefix)
            ].update(
                row["dom_values"]
            )

    token_counter = Counter()

    for row in rows:
        for value in row["dom_values"]:
            for prefix in detect_prefix_from_text(value):
                token_counter[
                    (prefix, value)
                ] += 1

    token_rows = []

    for (prefix, token), count in token_counter.most_common():
        token_rows.append(
            {
                "process_prefix": prefix,
                "dom_value": token,
                "event_occurrences": count,
                "segments_with_value": len(
                    {
                        row["segment_key"]
                        for row in rows
                        if token in row["dom_values"]
                        and prefix in row["prefix_hits"]
                    }
                ),
            }
        )

    pd.DataFrame(token_rows).to_csv(
        TOKEN_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    segment_rows = []

    for _, segment in segments.iterrows():
        key = segment["segment_key"]

        hits = {
            prefix: seg_prefix_event_count[
                (key, prefix)
            ]
            for prefix in PREFIXES
            if seg_prefix_event_count[
                (key, prefix)
            ] > 0
        }

        primary = (
            max(hits, key=hits.get)
            if hits
            else ""
        )

        segment_rows.append(
            {
                "segment_key": key,
                "session_id": segment["session_id"],
                "segment_index": int(
                    segment["segment_index"]
                ),
                "final_label": segment["label"],
                "detected_prefixes": "|".join(
                    sorted(hits)
                ),
                "primary_raw_prefix": primary,
                "prefix_count": len(hits),
                "pi_evidence_events": hits.get("pi", 0),
                "la_evidence_events": hits.get("la", 0),
                "ob_evidence_events": hits.get("ob", 0),
                "si_evidence_events": hits.get("si", 0),
                "rt_evidence_events": hits.get("rt", 0),
                "raw_dom_values": " | ".join(
                    sorted(
                        {
                            value
                            for prefix in hits
                            for value in seg_prefix_dom_values[
                                (key, prefix)
                            ]
                        }
                    )
                ),
            }
        )

    seg_out = pd.DataFrame(segment_rows)

    seg_out.to_csv(
        SEGMENT_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    variant_rows = []

    for prefix in PREFIXES:
        counts = prefix_variants[prefix]
        total = sum(counts.values())

        for variant, count in sorted(counts.items()):
            variant_rows.append(
                {
                    "process_prefix": prefix,
                    "application_variant": variant,
                    "raw_evidence_events": count,
                    "fraction_within_prefix_pct": (
                        count / total * 100
                        if total
                        else 0
                    ),
                }
            )

    pd.DataFrame(variant_rows).to_csv(
        VARIANT_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []

    for prefix in PREFIXES:
        type_counts = prefix_by_event_type[prefix]
        layer_counts = prefix_by_layer[prefix]

        summary_rows.append(
            {
                "process_prefix": prefix,
                "segments_with_raw_prefix": len(
                    prefix_token_segments[prefix]
                ),
                "fraction_of_540_segments_pct": (
                    len(prefix_token_segments[prefix])
                    / len(segments)
                    * 100
                ),
                "raw_prefix_event_occurrences": prefix_token_events[prefix],
                "sessions_with_prefix": len(
                    prefix_sessions[prefix]
                ),
                "distinct_operators_with_prefix": len(
                    prefix_operators[prefix]
                ),
                "event_types_with_prefix": len(
                    type_counts
                ),
                "layers_with_prefix": len(
                    layer_counts
                ),
                "top_event_types": json.dumps(
                    dict(
                        type_counts.most_common(10)
                    ),
                    ensure_ascii=False,
                ),
                "layers": json.dumps(
                    dict(layer_counts),
                    ensure_ascii=False,
                ),
                "top_applications": json.dumps(
                    dict(
                        prefix_apps[prefix].most_common(
                            10
                        )
                    ),
                    ensure_ascii=False,
                ),
                "application_variants": json.dumps(
                    dict(prefix_variants[prefix]),
                    ensure_ascii=False,
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("LAYER 4 PROCESS-FAMILY SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))

    print()
    print("PREFIX × FINAL SEGMENT LABEL")

    crosstab = pd.crosstab(
        seg_out["final_label"],
        seg_out["primary_raw_prefix"],
    )

    print(crosstab.to_string())

    print()
    print("SEGMENTS WITH MULTIPLE RAW PREFIXES:")

    multi = seg_out[
        seg_out["prefix_count"] > 1
    ]

    print(f"{len(multi)} / {len(seg_out)}")

    if not multi.empty:
        print(
            multi[
                [
                    "session_id",
                    "segment_index",
                    "final_label",
                    "detected_prefixes",
                    "pi_evidence_events",
                    "la_evidence_events",
                    "ob_evidence_events",
                    "si_evidence_events",
                    "rt_evidence_events",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    exact_label_agreement = (
        (
            seg_out["primary_raw_prefix"]
            == seg_out["final_label"]
        )
        | (
            (seg_out["final_label"] == "other")
            & (
                seg_out["primary_raw_prefix"]
                == ""
            )
        )
    ).mean()

    print()
    print(
        "Final-label vs raw-prefix agreement: "
        f"{exact_label_agreement * 100:.2f}%"
    )

    print()
    print("Outputs:")
    for path in (
        SUMMARY_OUT,
        TOKEN_OUT,
        SEGMENT_OUT,
        VARIANT_OUT,
    ):
        print(path)

    print()
    print(
        "Layer 4 complete."
    )
    print(
        "Conclusion at this layer: pi/la/ob/si/rt are validated as candidate stable workflow families."
    )
    print(
        "No business-process splitting or automation ranking performed."
    )


if __name__ == "__main__":
    main()
