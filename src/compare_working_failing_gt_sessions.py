
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
ENRICHED = ROOT / "outputs" / "labeler_gt_executions_with_signatures.csv"


WORKING_SESSION = "ses_20260630-121953-LAPTOP-R36BQBTE"


def parse_ts(value):
    try:
        return pd.to_datetime(value, utc=True)
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
            value = parse_ts(event[key])
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

    return matches


def locate_manifest(session_folder):
    path = session_folder / "gt_manifest.json"

    return path if path.exists() else None


def read_manifest_executions(manifest_path):
    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    rows = []

    for process in manifest.get(
        "processes",
        [],
    ):

        if not isinstance(process, dict):
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

            if not isinstance(execution, dict):
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

                    "start":
                        start,

                    "end":
                        end,

                    "exec_id":
                        str(
                            execution.get(
                                "exec_id",
                                "",
                            )
                            or ""
                        ),
                }
            )

    return pd.DataFrame(rows)


def load_events(session_folder):
    rows = []

    files = sorted(
        session_folder.rglob(
            "events.jsonl"
        )
    )

    for file in files:

        try:
            with file.open(
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
                            "file":
                                str(file),
                            "line":
                                line_no,
                            "event":
                                event,
                        }
                    )

        except OSError:
            continue

    rows.sort(
        key=lambda x: (
            x["timestamp"],
            x["file"],
            x["line"],
        )
    )

    return rows


def inspect_session(
    session_id,
    label,
):
    print()
    print("=" * 70)
    print(
        f"{label}: {session_id}"
    )
    print("=" * 70)

    matches = locate_session(
        session_id
    )

    print(
        f"Session directory candidates: "
        f"{len(matches)}"
    )

    for match in matches:
        print(
            f"  {match}"
        )

    if not matches:
        print(
            "NO SESSION DIRECTORY FOUND"
        )
        return

    if len(matches) > 1:
        print(
            "WARNING: multiple session directories found."
        )

    session_folder = matches[0]

    print(
        f"\nUsing:"
        f"\n{session_folder}"
    )

    event_files = sorted(
        session_folder.rglob(
            "events.jsonl"
        )
    )

    print(
        f"\nevents.jsonl files: "
        f"{len(event_files)}"
    )

    for f in event_files:
        print(
            f"  {f}"
        )

    events = load_events(
        session_folder
    )

    print(
        f"\nRaw events: "
        f"{len(events):,}"
    )

    if events:
        print(
            f"Raw first timestamp: "
            f"{events[0]['timestamp']}"
        )
        print(
            f"Raw last timestamp:  "
            f"{events[-1]['timestamp']}"
        )

    manifest_path = locate_manifest(
        session_folder
    )

    print(
        f"\nManifest:"
    )

    print(
        manifest_path
        if manifest_path
        else "NOT FOUND"
    )

    if not manifest_path:
        return

    gt = read_manifest_executions(
        manifest_path
    )

    print(
        f"\nComplete GT executions: "
        f"{len(gt):,}"
    )

    if gt.empty:
        return

    print(
        f"GT first start: "
        f"{gt['start'].min()}"
    )

    print(
        f"GT last end:    "
        f"{gt['end'].max()}"
    )

    # Inspect first, middle and last GT execution.
    sample_indices = sorted(
        set(
            [
                0,
                len(gt) // 2,
                len(gt) - 1,
            ]
        )
    )

    for index in sample_indices:

        row = gt.iloc[index]

        inside = [
            e
            for e in events
            if (
                e["timestamp"]
                >= row["start"]
                and e["timestamp"]
                <= row["end"]
            )
        ]

        print()
        print(
            f"GT sample #{index}: "
            f"{row['process_code']} "
            f"{row['process_name']}"
        )

        print(
            f"  start = {row['start']}"
        )

        print(
            f"  end   = {row['end']}"
        )

        print(
            f"  events inside = "
            f"{len(inside)}"
        )

        if not inside:

            before = [
                e
                for e in events
                if e["timestamp"]
                < row["start"]
            ]

            after = [
                e
                for e in events
                if e["timestamp"]
                > row["end"]
            ]

            if before:
                b = before[-1]

                print(
                    f"  nearest before = "
                    f"{b['timestamp']} "
                    f"delta="
                    f"{row['start'] - b['timestamp']}"
                )
            else:
                print(
                    "  nearest before = none"
                )

            if after:
                a = after[0]

                print(
                    f"  nearest after = "
                    f"{a['timestamp']} "
                    f"delta="
                    f"{a['timestamp'] - row['end']}"
                )
            else:
                print(
                    "  nearest after = none"
                )

        else:

            for e in inside[:5]:
                print(
                    f"    {e['timestamp']} | "
                    f"{e['event_type']} | "
                    f"{e['layer']} | "
                    f"{Path(e['file']).name}:{e['line']}"
                )


def main():

    print("=" * 70)
    print(
        "WORKING vs FAILING DATASET-A GT/EVENT SESSION DIAGNOSTIC"
    )
    print("=" * 70)

    if not ENRICHED.exists():
        raise FileNotFoundError(
            ENRICHED
        )

    enriched = pd.read_csv(
        ENRICHED,
        keep_default_na=False,
    )

    event_counts = pd.to_numeric(
        enriched[
            "event_count"
        ],
        errors="coerce",
    ).fillna(0)

    nonzero_sessions = list(
        enriched.loc[
            event_counts > 0,
            "session_id",
        ].unique()
    )

    zero_sessions = list(
        enriched.loc[
            event_counts == 0,
            "session_id",
        ].unique()
    )

    print(
        f"\nNon-zero sessions: "
        f"{len(nonzero_sessions)}"
    )

    print(
        f"Zero-event sessions: "
        f"{len(zero_sessions)}"
    )

    print(
        "\nNon-zero sessions:"
    )

    for session_id in nonzero_sessions:
        print(
            f"  {session_id}"
        )

    # Working control.
    inspect_session(
        WORKING_SESSION,
        "WORKING CONTROL SESSION",
    )

    # First five failing sessions.
    for i, session_id in enumerate(
        zero_sessions[:5],
        start=1,
    ):
        inspect_session(
            session_id,
            f"FAILING SESSION {i}",
        )

    print()
    print("=" * 70)
    print(
        "COMPARISON COMPLETE"
    )
    print("=" * 70)

    print(
        "\nInterpretation:"
    )

    print(
        "Compare raw timestamp range vs GT timestamp range "
        "for the working and failing sessions."
    )


if __name__ == "__main__":
    main()
