
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SEGMENTS_FILE = ROOT / "segments_v2.jsonl"
SEGMENT_EVIDENCE = OUT / "step2_layer3_authoritative_segment_evidence.csv"
L5_TAXONOMY = OUT / "step2_layer5_conservative_process_taxonomy_v1.csv"
L5_EVIDENCE = OUT / "step2_layer5_conservative_process_evidence_v1.csv"
L7_FAMILY = OUT / "step2_layer7_all540_family_quantification.csv"
L8_FAMILY = OUT / "step2_layer8_family_work_decomposition.csv"
L2_DOM = OUT / "step2_layer2_dom_identifier_inventory.csv"
L2_URL = OUT / "step2_layer2_url_inventory.csv"

OUT_MATRIX = OUT / "step2_layer9_feasibility_risk_matrix_v2.csv"
OUT_EVIDENCE = OUT / "step2_layer9_feasibility_evidence_v2.csv"
OUT_SEGAPP = OUT / "step2_layer9_segment_application_evidence_v2.csv"


FAMILIES = ["pi", "la", "ob", "si", "rt", "other"]

DATASET_B_HINTS = [
    "dataset_b",
    "dataset-b",
]


def norm(x):
    return re.sub(
        r"\s+",
        " ",
        str(x or ""),
    ).strip()


def split_values(value):
    """
    Split the compact pipe/semicolon representation used by the evidence
    tables. Empty values are ignored.
    """
    text = str(value or "").strip()

    if not text:
        return []

    return [
        item.strip()
        for item in re.split(
            r"\s*\|\s*|\s*;\s*",
            text,
        )
        if item.strip()
    ]


def find_dataset_b_root():
    """
    Discover Dataset B under the repository's raw_data tree.

    This deliberately scans only the project root, rather than making any
    assumptions about a machine-specific absolute path.
    """
    candidates = []

    for base in [
        ROOT,
        ROOT / "raw_data",
    ]:
        if not base.exists():
            continue

        for p in base.rglob("events.jsonl"):
            parts = {
                part.lower()
                for part in p.parts
            }

            if any(
                hint in p.as_posix().lower()
                for hint in DATASET_B_HINTS
            ):
                candidates.append(
                    p.parent
                )

    # Unique event-file parent directories.
    candidates = sorted(
        set(candidates)
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not discover Dataset-B events.jsonl files under project root."
        )

    return candidates


def load_segments():
    rows = []

    with SEGMENTS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(
                line
            )

            rows.append(
                {
                    "session_id":
                        str(
                            obj["session_id"]
                        ),
                    "start":
                        pd.to_datetime(
                            obj["start"],
                            utc=True,
                        ),
                    "end":
                        pd.to_datetime(
                            obj["end"],
                            utc=True,
                        ),
                    "label":
                        str(
                            obj["label"]
                        ),
                }
            )

    df = pd.DataFrame(
        rows
    )

    if len(df) != 540:
        raise RuntimeError(
            f"Expected 540 segments; found {len(df)}"
        )

    df["segment_index"] = (
        df.groupby(
            "session_id"
        ).cumcount()
        + 1
    )

    df["segment_key"] = (
        df["session_id"].astype(str)
        + "::"
        + df["segment_index"].astype(str)
    )

    return df


def load_segment_evidence():
    df = pd.read_csv(
        SEGMENT_EVIDENCE,
        keep_default_na=False,
    )

    df["segment_key"] = (
        df["session_id"].astype(str)
        + "::"
        + df["segment_index"].astype(str)
    )

    return df


def normalize_app_name(value):
    """
    Exact app normalization. No substring matching.

    Only event.app is used here. We do not treat visible/open-app context as
    active application involvement.
    """
    text = norm(value).lower()

    if not text:
        return ""

    exact = {
        "microsoft edge": "Microsoft Edge",
        "msedge.exe": "Microsoft Edge",
        "msedge": "Microsoft Edge",

        "microsoft word": "Microsoft Word",
        "winword.exe": "Microsoft Word",
        "winword": "Microsoft Word",

        "microsoft excel": "Microsoft Excel",
        "excel.exe": "Microsoft Excel",
        "excel": "Microsoft Excel",

        "notepad": "Notepad",
        "notepad.exe": "Notepad",

        "microsoft teams": "Microsoft Teams",
        "ms-teams": "Microsoft Teams",
        "ms-teams.exe": "Microsoft Teams",

        "windows explorer": "Windows Explorer",
        "explorer.exe": "Windows Explorer",
        "explorer": "Windows Explorer",

        "windowsterminal": "Windows Terminal",
        "windowsterminal.exe": "Windows Terminal",

        "powershell": "PowerShell",
        "powershell.exe": "PowerShell",

        "procmine-desktop-agent": "ProcMine Agent",
        "procmine-desktop-agent.exe": "ProcMine Agent",
    }

    return exact.get(
        text,
        norm(value),
    )


def raw_prefix(line, limit=500):
    text = str(line).replace("\r", " ").replace("\n", " ")
    return text[:limit]


def parse_event_line(line):
    """
    Robust Dataset-B event parser.

    Supports both:
      - normal JSON event objects
      - the flattened key/value representation visible in some Dataset-B
        records, e.g.
          line=66 timestamp_ms=1782924300467
          [["event.source.username_hash", "..."],
           ["event.context.active_app.process_name", "WINWORD.EXE"],
           ["event.context.active_app.app_name", "Microsoft Word"], ...]

    The fallback deliberately extracts only timestamp, event type, active app,
    and URL: exactly the fields needed by Layer 9.
    """
    raw = line.strip()

    # --------------------------------------------------------------
    # Attempt 1: normal JSON.
    # --------------------------------------------------------------
    try:
        obj = json.loads(raw)

        if isinstance(obj, dict):
            ts = (
                obj.get("timestamp_ms")
                or obj.get("timestamp")
                or obj.get("ts")
            )

            if isinstance(ts, str):
                try:
                    parsed = pd.to_datetime(
                        ts,
                        utc=True,
                    )
                    timestamp_ms = int(
                        parsed.timestamp() * 1000
                    )
                except Exception:
                    timestamp_ms = None
            else:
                try:
                    timestamp_ms = int(ts)
                except Exception:
                    timestamp_ms = None

            event_type = (
                obj.get("event_type")
                or obj.get("type")
                or obj.get("name")
                or ""
            )

            app = (
                obj.get("app")
                or obj.get("application")
                or (
                    obj.get("context", {})
                    .get("active_app", {})
                    .get("app_name", "")
                )
                or (
                    obj.get("event", {})
                    .get("context", {})
                    .get("active_app", {})
                    .get("app_name", "")
                )
            )

            process_name = (
                obj.get("context", {})
                .get("active_app", {})
                .get("process_name", "")
            )

            if process_name and not app:
                app = process_name

            url = (
                obj.get("url")
                or obj.get("payload", {}).get("url", "")
                or obj.get("event", {}).get("payload", {}).get("url", "")
            )

            return {
                "timestamp_ms": timestamp_ms,
                "event_type": str(event_type or ""),
                "app": normalize_app_name(app),
                "url": norm(url),
            }

    except Exception:
        pass

    # --------------------------------------------------------------
    # Attempt 2: flattened key/value event representation.
    # --------------------------------------------------------------
    # Timestamp can appear as:
    #   timestamp_ms=1782924300467
    #   "timestamp_ms": 1782924300467
    #   ["timestamp_ms", 1782924300467]
    timestamp_ms = None

    timestamp_patterns = [
        r'\btimestamp_ms\s*=\s*(\d{10,})',
        r'"timestamp_ms"\s*:\s*(\d{10,})',
        r'\[\s*"timestamp_ms"\s*,\s*(\d{10,})\s*\]',
    ]

    for pattern in timestamp_patterns:
        m = re.search(
            pattern,
            raw,
            flags=re.I,
        )
        if m:
            timestamp_ms = int(
                m.group(1)
            )
            break

    # Event type can be represented in either flattened or ordinary form.
    event_type = ""

    event_patterns = [
        r'\[\s*"event_type"\s*,\s*"([^"]+)"\s*\]',
        r'"event_type"\s*:\s*"([^"]+)"',
        r'\[\s*"event\.event_type"\s*,\s*"([^"]+)"\s*\]',
        r'\[\s*"event\.type"\s*,\s*"([^"]+)"\s*\]',
    ]

    for pattern in event_patterns:
        m = re.search(
            pattern,
            raw,
            flags=re.I,
        )
        if m:
            event_type = m.group(1)
            break

    # Active app: prefer active_app.app_name, then active_app.process_name.
    app = ""

    app_patterns = [
        r'\[\s*"event\.context\.active_app\.app_name"\s*,\s*"([^"]*)"\s*\]',
        r'\[\s*"context\.active_app\.app_name"\s*,\s*"([^"]*)"\s*\]',
        r'"active_app"\s*:\s*\{.*?"app_name"\s*:\s*"([^"]*)"',
    ]

    for pattern in app_patterns:
        m = re.search(
            pattern,
            raw,
            flags=re.I,
        )
        if m:
            app = m.group(1)
            break

    if not app:
        process_patterns = [
            r'\[\s*"event\.context\.active_app\.process_name"\s*,\s*"([^"]*)"\s*\]',
            r'\[\s*"context\.active_app\.process_name"\s*,\s*"([^"]*)"\s*\]',
            r'"active_app"\s*:\s*\{.*?"process_name"\s*:\s*"([^"]*)"',
        ]

        for pattern in process_patterns:
            m = re.search(
                pattern,
                raw,
                flags=re.I,
            )
            if m:
                app = m.group(1)
                break

    # URL: support flattened payload/url keys.
    url = ""

    url_patterns = [
        r'\[\s*"event\.payload\.url"\s*,\s*"([^"]*)"\s*\]',
        r'\[\s*"payload\.url"\s*,\s*"([^"]*)"\s*\]',
        r'"url"\s*:\s*"([^"]*)"',
    ]

    for pattern in url_patterns:
        m = re.search(
            pattern,
            raw,
            flags=re.I,
        )
        if m:
            url = m.group(1)
            break

    # Some event files expose the event type only as a filename-style field.
    if not event_type:
        filename_event = re.search(
            r'\b(event_type|type|name)\s*=\s*([A-Za-z0-9_.-]+)',
            raw,
            flags=re.I,
        )
        if filename_event:
            event_type = filename_event.group(2)

    if timestamp_ms is None:
        raise ValueError(
            "Unable to extract event timestamp"
        )

    return {
        "timestamp_ms":
            timestamp_ms,
        "event_type":
            str(event_type or ""),
        "app":
            normalize_app_name(
                app
            ),
        "url":
            norm(
                url
            ),
    }



def load_raw_segment_app_evidence(
    segment_df,
):
    """
    Rebuild exact application involvement from raw Dataset-B events.

    Uses [start, end) intervals and records reconciliation diagnostics instead
    of silently dropping events.
    """
    session_segments = defaultdict(list)

    for _, row in segment_df.iterrows():
        session_segments[
            row["session_id"]
        ].append(
            (
                int(row["segment_index"]),
                row["segment_key"],
                row["start"],
                row["end"],
            )
        )

    records = []

    event_files = find_dataset_b_root()

    print(
        f"Raw Dataset-B event-file directories discovered: {len(event_files)}"
    )

    event_file_count = 0
    total_events = 0
    parse_failures = 0
    parse_failure_samples = []
    missing_timestamp = 0
    outside_segment = 0
    boundary_candidates = 0

    for root in event_files:
        event_file = root / "events.jsonl"

        session_id = None
        for part in root.parts:
            if part.startswith("ses_"):
                session_id = part
                break

        if not session_id or session_id not in session_segments:
            continue

        event_file_count += 1

        with event_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_number, line in enumerate(
                f,
                start=1,
            ):
                if not line.strip():
                    continue

                total_events += 1

                try:
                    event = parse_event_line(line)
                except Exception as exc:
                    parse_failures += 1
                    if len(parse_failure_samples) < 10:
                        parse_failure_samples.append(
                            {
                                "file": str(event_file),
                                "line_number": line_number,
                                "error": str(exc),
                                "raw_prefix": raw_prefix(line),
                            }
                        )
                    continue

                if event["timestamp_ms"] is None:
                    missing_timestamp += 1
                    continue

                event_time = pd.to_datetime(
                    event["timestamp_ms"],
                    unit="ms",
                    utc=True,
                )

                matched = False

                # Primary policy: [start, end)
                for (
                    segment_index,
                    segment_key,
                    start_ts,
                    end_ts,
                ) in session_segments[session_id]:
                    if (
                        start_ts
                        <= event_time
                        < end_ts
                    ):
                        records.append(
                            {
                                "segment_key":
                                    segment_key,
                                "session_id":
                                    session_id,
                                "segment_index":
                                    segment_index,
                                "event_type":
                                    event["event_type"],
                                "active_app":
                                    event["app"],
                                "url":
                                    event["url"],
                                "assignment_policy":
                                    "standard_interval",
                            }
                        )
                        matched = True
                        break

                # Explicit endpoint policy: if an event lands exactly on a
                # segment end and no following segment starts at that same
                # instant, attach it to the preceding segment. This is an
                # endpoint reconciliation rule, not a new segmentation rule.
                if not matched:
                    end_matches = [
                        (
                            segment_index,
                            segment_key,
                            start_ts,
                            end_ts,
                        )
                        for (
                            segment_index,
                            segment_key,
                            start_ts,
                            end_ts,
                        ) in session_segments[
                            session_id
                        ]
                        if event_time == end_ts
                    ]

                    if end_matches:
                        (
                            segment_index,
                            segment_key,
                            start_ts,
                            end_ts,
                        ) = end_matches[-1]

                        records.append(
                            {
                                "segment_key":
                                    segment_key,
                                "session_id":
                                    session_id,
                                "segment_index":
                                    segment_index,
                                "event_type":
                                    event["event_type"],
                                "active_app":
                                    event["app"],
                                "url":
                                    event["url"],
                                "assignment_policy":
                                    "exact_end_boundary",
                            }
                        )
                        boundary_candidates += 1
                        matched = True

                if not matched:
                    outside_segment += 1

                    # Record exact boundary diagnostics even if a gap remains.
                    for (
                        _segment_index,
                        _segment_key,
                        start_ts,
                        end_ts,
                    ) in session_segments[
                        session_id
                    ]:
                        if (
                            event_time == start_ts
                            or event_time == end_ts
                        ):
                            boundary_candidates += 1
                            break

    raw_df = pd.DataFrame(records)

    print(
        f"Raw event files used: {event_file_count}"
    )
    print(
        f"Raw events scanned: {total_events:,}"
    )
    print(
        f"Raw event records attached to segments: {len(raw_df):,}"
    )
    print(
        f"Unassigned events: {outside_segment:,}"
    )
    print(
        f"Boundary-timestamp candidates among unassigned: "
        f"{boundary_candidates:,}"
    )
    print(
        f"Parse failures: {parse_failures:,}"
    )
    print(
        f"Missing timestamps: {missing_timestamp:,}"
    )

    expected = total_events
    reconciled = (
        len(raw_df)
        + outside_segment
        + parse_failures
        + missing_timestamp
    )

    print(
        f"Reconciliation check: "
        f"{len(raw_df):,} attached + "
        f"{outside_segment:,} unassigned + "
        f"{parse_failures:,} parse failures + "
        f"{missing_timestamp:,} missing timestamps = "
        f"{reconciled:,}"
    )

    if parse_failure_samples:
        print()
        print(
            "FIRST PARSE-FAILURE SAMPLES:"
        )
        for sample in parse_failure_samples:
            print(
                f"  {sample['file']}:{sample['line_number']} | "
                f"{sample['error']} | "
                f"{sample['raw_prefix']}"
            )

    if reconciled != expected:
        raise RuntimeError(
            "Raw-event reconciliation failed: "
            f"components sum to {reconciled}, expected {expected}."
        )

    return raw_df



def aggregate_raw_app_evidence(raw_df):
    rows = []

    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "segment_key",
                "exact_active_apps",
                "active_app_count",
                "word_active",
                "excel_active",
                "notepad_active",
                "browser_active",
                "teams_active",
                "other_active_apps",
                "raw_browser_navigation_events",
                "raw_browser_click_events",
                "raw_form_input_events",
            ]
        )

    for segment_key, group in raw_df.groupby(
        "segment_key"
    ):
        apps = sorted(
            {
                x
                for x in group[
                    "active_app"
                ]
                if x
            }
        )

        app_set = set(
            apps
        )

        browser = (
            "Microsoft Edge"
            in app_set
        )

        browser_nav = int(
            group[
                "event_type"
            ]
            .str.lower()
            .eq(
                "browser_navigation"
            )
            .sum()
        )

        browser_click = int(
            group[
                "event_type"
            ]
            .str.lower()
            .eq(
                "browser_click"
            )
            .sum()
        )

        form_input = int(
            group[
                "event_type"
            ]
            .str.lower()
            .eq(
                "browser_form_input"
            )
            .sum()
        )

        known = {
            "Microsoft Edge",
            "Microsoft Word",
            "Microsoft Excel",
            "Notepad",
            "Microsoft Teams",
            "Windows Explorer",
            "Windows Terminal",
            "PowerShell",
            "ProcMine Agent",
        }

        other_apps = sorted(
            app_set - known
        )

        rows.append(
            {
                "segment_key":
                    segment_key,
                "exact_active_apps":
                    " | ".join(
                        apps
                    ),
                "active_app_count":
                    len(apps),
                "word_active":
                    int(
                        "Microsoft Word"
                        in app_set
                    ),
                "excel_active":
                    int(
                        "Microsoft Excel"
                        in app_set
                    ),
                "notepad_active":
                    int(
                        "Notepad"
                        in app_set
                    ),
                "browser_active":
                    int(
                        browser
                    ),
                "teams_active":
                    int(
                        "Microsoft Teams"
                        in app_set
                    ),
                "other_active_apps":
                    " | ".join(
                        other_apps
                    ),
                "raw_browser_navigation_events":
                    browser_nav,
                "raw_browser_click_events":
                    browser_click,
                "raw_form_input_events":
                    form_input,
            }
        )

    return pd.DataFrame(
        rows
    )


def load_optional_family_data():
    l7 = pd.read_csv(
        L7_FAMILY,
        keep_default_na=False,
    )

    l8 = pd.read_csv(
        L8_FAMILY,
        keep_default_na=False,
    )

    taxonomy = pd.read_csv(
        L5_TAXONOMY,
        keep_default_na=False,
    )

    return (
        l7,
        l8,
        taxonomy,
    )


def main():
    print("=" * 80)
    print(
        "STEP 2 — LAYER 9 FEASIBILITY + RISK ASSESSMENT V2"
    )
    print("=" * 80)

    segments = load_segments()
    evidence = load_segment_evidence()

    # Preserve all 540 segment rows.
    base = segments.merge(
        evidence,
        on=[
            "segment_key",
            "session_id",
            "segment_index",
            "label",
        ],
        how="left",
        validate="one_to_one",
    )

    raw = load_raw_segment_app_evidence(
        segments
    )

    raw_apps = aggregate_raw_app_evidence(
        raw
    )

    if len(raw) != 20477:
        raise RuntimeError(
            "Layer-9 raw-event attachment did not reproduce the known "
            f"Dataset-B total of 20,477 attached records; got {len(raw)}. "
            "Do not use the feasibility matrix until this is reconciled."
        )

    base = base.merge(
        raw_apps,
        on="segment_key",
        how="left",
        validate="one_to_one",
    )

    for col in [
        "word_active",
        "excel_active",
        "notepad_active",
        "browser_active",
        "teams_active",
        "active_app_count",
        "raw_browser_navigation_events",
        "raw_browser_click_events",
        "raw_form_input_events",
    ]:
        base[col] = (
            pd.to_numeric(
                base[col],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    base["exact_active_apps"] = (
        base["exact_active_apps"]
        .fillna("")
        .astype(str)
    )

    base["other_active_apps"] = (
        base["other_active_apps"]
        .fillna("")
        .astype(str)
    )

    l7, l8, taxonomy = (
        load_optional_family_data()
    )

    targets = []

    # Family-level rows are the authoritative feasibility targets.
    for family in FAMILIES:
        if (
            family
            not in set(
                l7[
                    "process_family"
                ]
            )
        ):
            continue

        targets.append(
            {
                "family": family,
                "candidate":
                    f"{family}_family_level_workflow",
                "candidate_type":
                    "FAMILY_LEVEL",
            }
        )

    # Add directly observed case types.
    if not taxonomy.empty:
        obs = taxonomy[
            taxonomy[
                "classification"
            ]
            == "OBSERVED_CASE_TYPE"
        ]

        for _, row in obs.iterrows():
            targets.append(
                {
                    "family":
                        row[
                            "family"
                        ],
                    "candidate":
                        row[
                            "candidate_process"
                        ],
                    "candidate_type":
                        "OBSERVED_CASE_TYPE",
                }
            )

    l7_map = {
        row[
            "process_family"
        ]: row
        for _, row
        in l7.iterrows()
    }

    l8_map = {
        row[
            "family"
        ]: row
        for _, row
        in l8.iterrows()
    }

    output = []
    evidence_rows = []

    for target in targets:
        family = target[
            "family"
        ]
        candidate = target[
            "candidate"
        ]
        candidate_type = target[
            "candidate_type"
        ]

        fam = base[
            base[
                "label"
            ]
            == family
        ].copy()

        # Exact application metrics from raw events.
        app_metrics = {
            "word_active_pct":
                fam[
                    "word_active"
                ].mean()
                * 100,
            "excel_active_pct":
                fam[
                    "excel_active"
                ].mean()
                * 100,
            "notepad_active_pct":
                fam[
                    "notepad_active"
                ].mean()
                * 100,
            "browser_active_pct":
                fam[
                    "browser_active"
                ].mean()
                * 100,
            "mean_active_app_count":
                fam[
                    "active_app_count"
                ].mean(),
            "distinct_active_apps":
                sorted(
                    {
                        x
                        for value in fam[
                            "exact_active_apps"
                        ]
                        for x
                        in split_values(
                            value
                        )
                        if x
                    }
                ),
        }

        stats = l7_map.get(
            family,
            {}
        )

        l8stats = l8_map.get(
            family,
            {}
        )

        segments = int(
            stats.get(
                "execution_count_segments",
                len(fam),
            )
            or len(fam)
        )

        time_share = float(
            stats.get(
                "relative_time_share_pct",
                0,
            )
            or 0
        )

        mechanical = float(
            l8stats.get(
                "mechanical_evidence_pct",
                0,
            )
            or 0
        )

        decision = float(
            l8stats.get(
                "decision_point_evidence_pct",
                0,
            )
            or 0
        )

        clipboard = float(
            stats.get(
                "clipboard_segment_pct",
                0,
            )
            or 0
        )

        browser_interaction = float(
            stats.get(
                "browser_interaction_pct",
                0,
            )
            or 0
        )

        # Candidate-specific evidence.
        cand_row = taxonomy[
            (
                taxonomy[
                    "family"
                ]
                == family
            )
            & (
                taxonomy[
                    "candidate_process"
                ]
                == candidate
            )
        ]

        if not cand_row.empty:
            cand = cand_row.iloc[0]
        else:
            cand = {}

        route_evidence = (
            norm(
                cand.get(
                    "routes",
                    "",
                )
            )
            if isinstance(
                cand,
                dict,
            )
            else norm(
                cand.get(
                    "routes",
                    "",
                )
            )
        )

        field_evidence = (
            norm(
                cand.get(
                    "key_fields",
                    "",
                )
            )
            if isinstance(
                cand,
                dict,
            )
            else ""
        )

        action_evidence = (
            norm(
                cand.get(
                    "actions_status",
                    "",
                )
            )
            if isinstance(
                cand,
                dict,
            )
            else ""
        )

        document_evidence = (
            norm(
                cand.get(
                    "document_pattern",
                    "",
                )
            )
            if isinstance(
                cand,
                dict,
            )
            else ""
        )

        variants = (
            norm(
                cand.get(
                    "handling_variants",
                    "",
                )
            )
            if isinstance(
                cand,
                dict,
            )
            else ""
        )

        # Strong system/UI facts.
        expected_routes = {
            "pi":
                "/payroll-items",
            "la":
                "/leave-applications",
            "ob":
                "/onboarding",
            "si":
                "/social-insurance",
            "rt":
                "/resident-tax",
        }

        expected_route = (
            expected_routes.get(
                family,
                "",
            )
        )

        route_observed = (
            expected_route in str(
                evidence.get(
                    "urls",
                    pd.Series(
                        "",
                        index=evidence.index,
                    ),
                )
                .astype(str)
                .tolist()
            )
            if expected_route
            else False
        )

        route_answer = (
            "Observed in Dataset-B browser URLs"
            if route_observed
            else
            "Not independently confirmed from the URL inventory"
        )

        # DOM family signatures are known, but production stability is not.
        dom_answer = (
            f"Family-specific DOM identifiers are observed for {family}; "
            "production selector stability is UNKNOWN."
        )

        # System access cannot be proven from logs.
        system_access = (
            "Browser-based target interaction is observed. "
            "Production API availability, authentication method, "
            "permissions, and integration interfaces are UNKNOWN."
        )

        # Complexity/branch evidence.
        distinct_apps = app_metrics[
            "distinct_active_apps"
        ]

        if len(
            distinct_apps
        ) >= 3:
            branch_level = (
                "Multiple application-handling patterns observed"
            )
        elif len(
            distinct_apps
        ) == 2:
            branch_level = (
                "At least two application-handling patterns observed"
            )
        else:
            branch_level = (
                "Limited active-application variation observed"
            )

        if clipboard >= 80:
            transfer_pattern = (
                "Frequent clipboard activity; cross-system/data-transfer work is strongly evidenced."
            )
        elif clipboard >= 40:
            transfer_pattern = (
                "Moderate clipboard activity; data transfer is evidenced."
            )
        else:
            transfer_pattern = (
                "Clipboard activity is not dominant."
            )

        # Information gaps are evidence-based, not invented.
        missing = [
            "business-rule logic behind decisions",
            "production authentication/credential model",
            "production API/integration availability",
            "exception criteria",
            "real production wait times and SLAs",
        ]

        if candidate_type == "OBSERVED_CASE_TYPE":
            missing.append(
                "complete case-specific workflow outside OCR-observed segments"
            )

        failures = [
            "UI/selector changes",
            "expired sessions or authentication failures",
            "unexpected input/case structure",
            "upstream/downstream outage",
            "business-rule changes",
        ]

        if app_metrics[
            "word_active_pct"
        ] > 0:
            failures.append(
                "document/application state mismatch"
            )

        if clipboard >= 50:
            failures.append(
                "copied/transferred data mismatch"
            )

        # Governance: employee/tax/payroll/social-insurance domains are
        # inherently sensitive; the logs do not reveal policy, so state the
        # requirement rather than asserting a specific policy.
        if family in {
            "pi",
            "ob",
            "si",
            "rt",
        }:
            governance = (
                "Potentially sensitive employee/payroll/insurance/tax records "
                "are involved. Client validation is required for least privilege, "
                "credential storage, audit logs, retention, access review, and "
                "data-handling policy."
            )
            data_risk = "HIGH"
        else:
            governance = (
                "Business records are involved. Client validation is required "
                "for access control, credential storage, auditability, and retention."
            )
            data_risk = "MEDIUM"

        # Automation suitability is deliberately not a probability and is not
        # based only on a decision-button count.
        if (
            mechanical >= 90
            and decision < 20
            and clipboard >= 80
        ):
            fit = "STRONG_MECHANICAL_FIT"
        elif (
            mechanical >= 80
            and clipboard >= 50
        ):
            fit = "GOOD_MECHANICAL_FIT_WITH_HUMAN_STEP"
        else:
            fit = "PARTIAL_AUTOMATION_CANDIDATE"

        if decision >= 30:
            human_risk = "HIGH"
        elif decision >= 15:
            human_risk = "MEDIUM"
        else:
            human_risk = "LOW-MEDIUM"

        selector_risk = (
            "MEDIUM"
            if family != "other"
            else "HIGH / UNKNOWN"
        )

        cross_app_risk = (
            "HIGH"
            if (
                len(
                    distinct_apps
                ) >= 3
                or clipboard >= 80
            )
            else (
                "MEDIUM"
                if len(
                    distinct_apps
                ) >= 2
                else "LOW-MEDIUM"
            )
        )

        # Important: no score says "automation probability". This is simply
        # an auditable risk-level count to order diligence work.
        risk_points = (
            (2 if human_risk == "HIGH" else 1 if human_risk == "MEDIUM" else 0)
            + (2 if cross_app_risk == "HIGH" else 1 if cross_app_risk == "MEDIUM" else 0)
            + (2 if selector_risk.startswith("HIGH") else 1)
            + (2 if data_risk == "HIGH" else 1)
        )

        if risk_points >= 7:
            feasibility = "PILOT-FIRST"
        elif risk_points >= 5:
            feasibility = "FEASIBLE_WITH_VALIDATION"
        else:
            feasibility = "LOWER_RISK"

        if fit == "STRONG_MECHANICAL_FIT":
            recommendation = (
                "Prioritize for human-in-the-loop automation assessment"
            )
        elif fit == "GOOD_MECHANICAL_FIT_WITH_HUMAN_STEP":
            recommendation = (
                "Good candidate for partial automation; retain human decision step"
            )
        else:
            recommendation = (
                "Use targeted automation assistance rather than full automation"
            )

        if decision >= 30:
            recommendation += (
                "; do not automate the decision itself without business-rule evidence"
            )

        # Facts vs interpretation vs unknown.
        fact_evidence = [
            f"{segments} segments in family",
            f"{time_share:.1f}% of Dataset-B segment time",
            f"{mechanical:.1f}% of family segments show mechanical activity evidence",
            f"{decision:.1f}% show contextual decision-point evidence",
            f"{clipboard:.1f}% show clipboard activity",
            f"{app_metrics['browser_active_pct']:.1f}% contain Microsoft Edge as the exact active app",
            f"exact active apps observed: {', '.join(distinct_apps) if distinct_apps else 'none'}",
        ]

        interpretation_evidence = [
            transfer_pattern,
            branch_level,
            "High frequency of repetitive UI/data movement suggests an automation opportunity, "
            "but does not establish end-to-end automability.",
        ]

        unknown_evidence = [
            "API/integration availability",
            "production authentication model",
            "business decision rules",
            "exception policy",
            "production UI stability",
            "production governance requirements",
        ]

        output.append(
            {
                "family":
                    family,
                "candidate":
                    candidate,
                "candidate_type":
                    candidate_type,
                "all_540_family_segments":
                    segments,
                "family_time_share_pct":
                    time_share,
                "mechanical_evidence_pct":
                    mechanical,
                "decision_point_evidence_pct":
                    decision,
                "clipboard_segment_pct":
                    clipboard,
                "browser_interaction_pct":
                    browser_interaction,
                "exact_active_browser_pct":
                    app_metrics[
                        "browser_active_pct"
                    ],
                "exact_active_word_pct":
                    app_metrics[
                        "word_active_pct"
                    ],
                "exact_active_excel_pct":
                    app_metrics[
                        "excel_active_pct"
                    ],
                "exact_active_notepad_pct":
                    app_metrics[
                        "notepad_active_pct"
                    ],
                "distinct_exact_active_apps":
                    " | ".join(
                        distinct_apps
                    ),
                "mean_exact_active_app_count":
                    app_metrics[
                        "mean_active_app_count"
                    ],
                "system_access":
                    system_access,
                "ui_stability":
                    dom_answer,
                "systems_and_documents":
                    transfer_pattern,
                "observed_branch_variation":
                    branch_level,
                "missing_information":
                    " | ".join(
                        missing
                    ),
                "potential_production_failures":
                    " | ".join(
                        failures
                    ),
                "governance_security":
                    governance,
                "route_evidence":
                    route_evidence
                    or expected_route,
                "route_status":
                    route_answer,
                "dom_status":
                    dom_answer,
                "key_fields":
                    field_evidence,
                "actions_status":
                    action_evidence,
                "document_pattern":
                    document_evidence,
                "handling_variants":
                    variants,
                "automation_fit":
                    fit,
                "human_judgment_risk":
                    human_risk,
                "cross_application_risk":
                    cross_app_risk,
                "ui_selector_risk":
                    selector_risk,
                "data_governance_risk":
                    data_risk,
                "overall_feasibility":
                    feasibility,
                "risk_points":
                    risk_points,
                "recommendation":
                    recommendation,
                "observed_facts":
                    " | ".join(
                        fact_evidence
                    ),
                "interpretations":
                    " | ".join(
                        interpretation_evidence
                    ),
                "unknowns":
                    " | ".join(
                        unknown_evidence
                    ),
            }
        )

        evidence_rows.append(
            {
                "family":
                    family,
                "candidate":
                    candidate,
                "workload_evidence":
                    "Layer 7 all-540 family quantification",
                "mechanical_human_evidence":
                    "Layer 8 family decomposition",
                "semantic_evidence":
                    (
                        "Layer 5 direct OCR taxonomy"
                        if candidate_type
                        == "OBSERVED_CASE_TYPE"
                        else
                        "Layer 4 family validation + Layer 6 semantic evidence"
                    ),
                "application_evidence":
                    "Raw Dataset-B events.jsonl re-parsed per segment using exact event.app",
                "route_evidence":
                    "Layer 2 URL inventory + Layer 3 segment evidence",
                "dom_evidence":
                    "Layer 2 DOM inventory + Layer 4 family validation",
                "method_note":
                    "Application metrics in this version do not use substring matching "
                    "or visible/open-app context.",
            }
        )

    matrix = pd.DataFrame(
        output
    )
    evidence_df = pd.DataFrame(
        evidence_rows
    )

    matrix.to_csv(
        OUT_MATRIX,
        index=False,
        encoding="utf-8-sig",
    )

    evidence_df.to_csv(
        OUT_EVIDENCE,
        index=False,
        encoding="utf-8-sig",
    )

    # Save the exact raw active-app evidence as an audit trail.
    raw_apps.to_csv(
        OUT_SEGAPP,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print(
        "CORRECTED APPLICATION EVIDENCE"
    )
    print("=" * 80)

    for family in FAMILIES:
        fam = base[
            base["label"] == family
        ]

        if fam.empty:
            continue

        print(
            f"{family}: "
            f"browser={fam['browser_active'].mean()*100:.1f}% | "
            f"word={fam['word_active'].mean()*100:.1f}% | "
            f"excel={fam['excel_active'].mean()*100:.1f}% | "
            f"notepad={fam['notepad_active'].mean()*100:.1f}% | "
            f"mean active apps={fam['active_app_count'].mean():.2f}"
        )

    print()
    print(
        "Layer 9 V2 complete."
    )
    print(
        "No API availability, production selector stability, business rules, "
        "or governance policies were assumed."
    )
    print(
        "segments_v2.jsonl was not modified."
    )

    print()
    print("Outputs:")
    print(OUT_MATRIX)
    print(OUT_EVIDENCE)
    print(OUT_SEGAPP)


if __name__ == "__main__":
    main()
