
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
ENRICHED = ROOT / "outputs" / "labeler_gt_executions_with_signatures.csv"

CONTROL_SESSION = "ses_20260630-121953-LAPTOP-R36BQBTE"

# ============================================================
# DEBUG PURPOSE
# ============================================================
#
# The signature experiment found:
#   1,724 / 1,752 complete GT executions had event_count == 0.
#
# The raw event census proves Dataset A has rich app/window/browser
# data, so this script isolates the interval -> raw-event mapping.
#
# It does NOT:
#   - train anything
#   - change the segmenter
#   - build a new signature
#   - translate/OCR anything
#
# It prints:
#   1) exact resolved session folder
#   2) raw event count and timestamp range
#   3) GT execution count/range for the control session
#   4) event overlap for several GT executions
#   5) the sessions represented by the 28 non-zero rows
#
# ============================================================


def parse_ts(value):
    try:
        return pd.to_datetime(
            value,
            utc=True,
        )
    except Exception:
        return pd.NaT


def event_timestamp(event):
    if event.get("timestamp_ms") is not None:
        try:
            return pd.to_datetime(
                int(event["timestamp_ms"]),
                unit="ms",
                utc=True,
            )
        except Exception:
            pass

    for key in (
        "timestamp_iso",
        "timestamp",
        "ts_utc",
    ):
        if key in event:
            value = parse_ts(
                event[key]
            )
            if not pd.isna(value):
                return value

    return pd.NaT


def locate_session(session_id):
    direct = DATA_ROOT / session_id

    if direct.exists():
        return direct

    matches = [
        p
        for p in DATA_ROOT.rglob(session_id)
        if p.is_dir()
    ]

    if not matches:
        return None

    # Print all candidates. We will NOT silently choose among
    # multiple matches without showing them.
    print()
    print("Candidate session directories:")
    for p in matches:
        print("  ", p)

    return matches[0]


def load_raw_events(session_folder):
    rows = []

    files = sorted(
        session_folder.rglob(
            "events.jsonl"
        )
    )

    for event_file in files:

        try:
            with event_file.open(
                "r",
                encoding="utf-8",
            ) as handle:

                for line_no, line in enumerate(
                    handle,
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
                        continue

                    timestamp = event_timestamp(
                        event
                    )

                    if pd.isna(timestamp):
                        continue

                    rows.append(
                        {
                            "timestamp":
                                timestamp,
                            "event_type":
                                str(
                                    event.get(
                                        "event_type",
                                        event.get(
                                            "event",
                                            "",
                                        ),
                                    )
                                    or ""
                                ),
                            "layer":
                                str(
                                    event.get(
                                        "layer",
                                        "",
                                    )
                                    or ""
                                ),
                            "event_file":
                                str(
                                    event_file
                                ),
                            "source_line":
                                line_no,
                            "event":
                                event,
                        }
                    )

        except OSError as exc:
            print(
                f"WARNING reading {event_file}: {exc}"
            )

    rows.sort(
        key=lambda row: (
            row["timestamp"],
            row["event_file"],
            row["source_line"],
        )
    )

    return rows


def find_gt_manifest(session_folder):
    path = (
        session_folder
        / "gt_manifest.json"
    )

    if path.exists():
        return path

    return None


def extract_executions(manifest):
    rows = []

    for process in manifest.get(
        "processes",
        [],
    ):

        if not isinstance(
            process,
            dict,
        ):
            continue

        process_code = str(
            process.get(
                "code",
                "",
            )
            or ""
        )

        process_name = str(
            process.get(
                "family_name",
                "",
            )
            or ""
        )

        for execution in process.get(
            "executions",
            [],
        ):

            if not isinstance(
                execution,
                dict,
            ):
                continue

            start = parse_ts(
                execution.get(
                    "start_ts"
                )
            )

            end = parse_ts(
                execution.get(
                    "end_ts"
                )
            )

            if (
                pd.isna(start)
                or pd.isna(end)
            ):
                continue

            rows.append(
                {
                    "process_code":
                        str(
                            execution.get(
                                "code",
                                process_code,
                            )
                            or process_code
                        ),
                    "process_name":
                        process_name,
                    "case_id":
                        str(
                            execution.get(
                                "case_id",
                                "",
                            )
                            or ""
                        ),
                    "variant":
                        str(
                            execution.get(
                                "variant",
                                "",
                            )
                            or ""
                        ),
                    "start_ts":
                        start,
                    "end_ts":
                        end,
                    "exec_id":
                        str(
                            execution.get(
                                "exec_id",
                                "",
                            )
                            or ""
                        ),
                    "continues_from_prev":
                        bool(
                            execution.get(
                                "continues_from_prev",
                                False,
                            )
                        ),
                    "continues_to_next":
                        bool(
                            execution.get(
                                "continues_to_next",
                                False,
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def events_in_interval(
    events,
    start,
    end,
):
    return [
        row
        for row in events
        if (
            row["timestamp"] >= start
            and row["timestamp"] <= end
        )
    ]


def print_event_sample(
    rows,
    limit=10,
):
    for row in rows[:limit]:
        print(
            f"  {row['timestamp']} | "
            f"{row['event_type']} | "
            f"{row['layer']} | "
            f"{Path(row['event_file']).name}:{row['source_line']}"
        )


def inspect_control_session():
    print()
    print("=" * 70)
    print(
        "CONTROL SESSION"
    )
    print("=" * 70)

    session_folder = locate_session(
        CONTROL_SESSION
    )

    if session_folder is None:
        print(
            "CONTROL SESSION NOT FOUND"
        )
        return None, None

    print()
    print(
        "Resolved session folder:"
    )
    print(
        session_folder
    )

    event_files = sorted(
        session_folder.rglob(
            "events.jsonl"
        )
    )

    print()
    print(
        f"events.jsonl files: "
        f"{len(event_files)}"
    )

    for path in event_files:
        print(
            f"  {path}"
        )

    events = load_raw_events(
        session_folder
    )

    print()
    print(
        f"Raw events loaded: "
        f"{len(events):,}"
    )

    if events:
        print(
            f"First raw event: "
            f"{events[0]['timestamp']}"
        )

        print(
            f"Last raw event: "
            f"{events[-1]['timestamp']}"
        )

        print()
        print(
            "First 5 raw events:"
        )

        print_event_sample(
            events,
            limit=5,
        )

        print()
        print(
            "Last 5 raw events:"
        )

        print_event_sample(
            list(
                reversed(events)
            ),
            limit=5,
        )

    manifest_path = (
        find_gt_manifest(
            session_folder
        )
    )

    if manifest_path is None:
        print(
            "gt_manifest.json NOT FOUND "
            "inside resolved session."
        )
        return events, None

    print()
    print(
        f"Manifest: {manifest_path}"
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    gt = extract_executions(
        manifest
    )

    print()
    print(
        f"Complete GT executions in control session: "
        f"{len(gt):,}"
    )

    if gt.empty:
        return events, gt

    print()
    print(
        "GT time range:"
    )

    print(
        f"  first start: "
        f"{gt.start_ts.min()}"
    )

    print(
        f"  last end:    "
        f"{gt.end_ts.max()}"
    )

    print()
    print(
        "GT timestamp timezone:"
    )

    print(
        f"  start tz: "
        f"{gt.iloc[0]['start_ts'].tz}"
    )

    # Check several executions:
    # first, middle, last, plus any execution with matching events.
    indices = []

    for idx in [
        0,
        len(gt) // 2,
        len(gt) - 1,
    ]:
        if (
            idx >= 0
            and idx < len(gt)
            and idx not in indices
        ):
            indices.append(idx)

    print()
    print("=" * 70)
    print(
        "GT ↔ RAW EVENT OVERLAP CHECK"
    )
    print("=" * 70)

    for idx in indices:

        row = gt.iloc[idx]

        overlap = events_in_interval(
            events,
            row["start_ts"],
            row["end_ts"],
        )

        print()
        print(
            f"GT #{idx}"
        )

        print(
            f"  process_code: "
            f"{row['process_code']}"
        )

        print(
            f"  process_name: "
            f"{row['process_name']}"
        )

        print(
            f"  start: "
            f"{row['start_ts']}"
        )

        print(
            f"  end:   "
            f"{row['end_ts']}"
        )

        print(
            f"  events_inside: "
            f"{len(overlap)}"
        )

        if overlap:
            print_event_sample(
                overlap,
                limit=5,
            )

        else:
            before = [
                e
                for e in events
                if e["timestamp"]
                < row["start_ts"]
            ]

            after = [
                e
                for e in events
                if e["timestamp"]
                > row["end_ts"]
            ]

            print(
                "  nearest before:"
            )

            if before:
                b = before[-1]
                print(
                    f"    {b['timestamp']} "
                    f"delta="
                    f"{row['start_ts'] - b['timestamp']}"
                )
            else:
                print(
                    "    none"
                )

            print(
                "  nearest after:"
            )

            if after:
                a = after[0]
                print(
                    f"    {a['timestamp']} "
                    f"delta="
                    f"{a['timestamp'] - row['end_ts']}"
                )
            else:
                print(
                    "    none"
                )

    return events, gt


def inspect_nonzero_enrichment():
    print()
    print("=" * 70)
    print(
        "THE 28 NON-ZERO ENRICHED EXECUTIONS"
    )
    print("=" * 70)

    if not ENRICHED.exists():
        print(
            f"Missing:\n{ENRICHED}"
        )
        return

    df = pd.read_csv(
        ENRICHED,
        keep_default_na=False,
    )

    if "event_count" not in df.columns:
        print(
            "event_count column missing."
        )
        return

    counts = pd.to_numeric(
        df["event_count"],
        errors="coerce",
    ).fillna(0)

    nonzero = df[
        counts > 0
    ].copy()

    nonzero[
        "event_count"
    ] = counts[
        counts > 0
    ].astype(int).values

    print(
        f"Rows with event_count > 0: "
        f"{len(nonzero):,}"
    )

    if nonzero.empty:
        return

    print()
    print(
        "Sessions represented by non-zero rows:"
    )

    session_summary = (
        nonzero
        .groupby(
            "session_id"
        )
        .agg(
            executions=(
                "session_id",
                "size",
            ),
            total_events=(
                "event_count",
                "sum",
            ),
            max_events=(
                "event_count",
                "max",
            ),
        )
        .sort_values(
            [
                "executions",
                "total_events",
            ],
            ascending=False,
        )
    )

    print(
        session_summary.to_string()
    )

    print()
    print(
        "Non-zero execution examples:"
    )

    display_columns = [
        "session_id",
        "process_code",
        "process_name",
        "start_ts",
        "end_ts",
        "event_count",
        "url_paths",
        "apps",
        "window_titles",
    ]

    display_columns = [
        col
        for col in display_columns
        if col in nonzero.columns
    ]

    print(
        nonzero[
            display_columns
        ]
        .sort_values(
            "event_count",
            ascending=False,
        )
        .head(20)
        .to_string(
            index=False
        )
    )


def main():
    print("=" * 70)
    print(
        "GT ↔ RAW EVENT TIMESTAMP ALIGNMENT DEBUGGER"
    )
    print("=" * 70)

    print(
        "\nThis is a diagnosis-only script."
    )

    inspect_control_session()
    inspect_nonzero_enrichment()

    print()
    print("=" * 70)
    print(
        "DEBUG COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
