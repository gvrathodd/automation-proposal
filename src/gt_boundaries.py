from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

    return records


def find_gt_files() -> list[Path]:
    return sorted(DATA_ROOT.glob("**/dataset_a/**/gt.jsonl"))


def extract_session_id(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("ses_"):
            return part
    return None


def build_boundary_rows(gt_path: Path) -> list[dict[str, Any]]:
    session_id = extract_session_id(gt_path)

    records = read_jsonl(gt_path)

    rows: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        event_type = record.get("event")

        if event_type not in {
            "process_started",
            "process_switched_out",
            "process_suspended",
            "process_resumed",
        }:
            continue

        rows.append(
            {
                "session_id": session_id,
                "source_file": str(gt_path),
                "source_line": index + 1,
                "ts": record.get("ts_utc"),
                "event": event_type,
                "process_code": record.get("process_code")
                or record.get("from")
                or record.get("current_process"),
                "current_process": record.get("current_process"),
                "process_variant": record.get("process_variant"),
                "process_name": record.get("process_name"),
                "case_id": record.get("case_id"),
                "from": record.get("from"),
                "to": record.get("to"),
                "split_id": record.get("split_id"),
                "phase": record.get("phase"),
            }
        )

    return rows


def main() -> None:
    gt_files = find_gt_files()

    print("=" * 72)
    print("GROUND TRUTH BOUNDARY EXTRACTION")
    print("=" * 72)

    print(f"GT files found: {len(gt_files)}")

    all_rows: list[dict[str, Any]] = []

    for path in gt_files:
        all_rows.extend(build_boundary_rows(path))

    if not all_rows:
        raise RuntimeError("No GT boundary events found.")

    df = pd.DataFrame(all_rows)

    df["ts"] = pd.to_datetime(
        df["ts"],
        utc=True,
        errors="coerce",
    )

    df = df.sort_values(
        ["session_id", "ts", "source_line"],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "gt_boundaries.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Boundary events extracted: {len(df)}")
    print(f"Sessions represented: {df['session_id'].nunique()}")
    print(f"Output: {output_path}")

    print()
    print("=" * 72)
    print("BOUNDARY EVENT COUNTS")
    print("=" * 72)

    counts = df["event"].value_counts()

    for event_type, count in counts.items():
        print(f"{event_type:24s} {count:>6}")

    print()
    print("=" * 72)
    print("PROCESS STARTS")
    print("=" * 72)

    starts = df[df["event"] == "process_started"]

    print(f"Process-start events: {len(starts)}")

    print()
    print("=" * 72)
    print("PROCESS SWITCH-OUTS")
    print("=" * 72)

    switches = df[
        df["event"] == "process_switched_out"
    ]

    print(f"Process-switch-out events: {len(switches)}")

    print()
    print("=" * 72)
    print("SUSPENSIONS / RESUMES")
    print("=" * 72)

    suspensions = df[
        df["event"] == "process_suspended"
    ]

    resumes = df[
        df["event"] == "process_resumed"
    ]

    print(f"Suspensions: {len(suspensions)}")
    print(f"Resumptions: {len(resumes)}")

    if len(suspensions):
        print()
        print("Examples:")
        print(
            suspensions[
                [
                    "session_id",
                    "ts",
                    "process_code",
                    "from",
                    "to",
                    "split_id",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()