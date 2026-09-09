from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_dataset_a_manifests() -> list[Path]:
    return sorted(
        DATA_ROOT.glob(
            "**/dataset_a/**/gt_manifest.json"
        )
    )


def extract_executions(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_json(manifest_path)

    session_id = None

    for part in manifest_path.parts:
        if part.startswith("ses_"):
            session_id = part
            break

    rows: list[dict[str, Any]] = []

    for process in manifest.get("processes", []):
        process_code = process.get("code")
        process_name = process.get("family_name")
        domain = process.get("domain")

        for execution in process.get("executions", []):
            rows.append(
                {
                    "session_id": session_id,
                    "process_code": process_code,
                    "process_name": process_name,
                    "domain": domain,
                    "variant": execution.get("variant"),
                    "case_id": execution.get("case_id"),
                    "start_ts": execution.get("start_ts"),
                    "end_ts": execution.get("end_ts"),
                    "phase": execution.get("phase"),
                    "exec_id": execution.get("exec_id"),
                    "seq": execution.get("seq"),
                    "split_id": execution.get("split_id"),
                    "continues_from_prev": execution.get(
                        "continues_from_prev"
                    ),
                    "continues_to_next": execution.get(
                        "continues_to_next"
                    ),
                }
            )

    return rows


def main() -> None:
    manifest_paths = find_dataset_a_manifests()

    print("=" * 70)
    print("GROUND TRUTH EXTRACTION")
    print("=" * 70)

    print(f"GT manifests found: {len(manifest_paths)}")

    all_rows: list[dict[str, Any]] = []

    for path in manifest_paths:
        rows = extract_executions(path)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No ground-truth executions found.")

    df = pd.DataFrame(all_rows)

    # Parse timestamps while preserving missing end timestamps.
    df["start_ts"] = pd.to_datetime(
        df["start_ts"],
        utc=True,
        errors="coerce",
    )

    df["end_ts"] = pd.to_datetime(
        df["end_ts"],
        utc=True,
        errors="coerce",
    )

    df["duration_seconds"] = (
        df["end_ts"] - df["start_ts"]
    ).dt.total_seconds()

    df = df.sort_values(
        ["session_id", "start_ts", "process_code", "seq"],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "gt_executions.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Executions extracted: {len(df)}")
    print(f"Sessions represented: {df['session_id'].nunique()}")
    print(f"Processes represented: {df['process_code'].nunique()}")
    print(f"Output: {output_path}")

    print()
    print("=" * 70)
    print("PROCESS COUNTS")
    print("=" * 70)

    counts = (
        df.groupby(
            ["process_code", "process_name", "domain"],
            dropna=False,
        )
        .size()
        .sort_values(ascending=False)
    )

    for index, count in counts.items():
        process_code, process_name, domain = index

        print(
            f"{process_code:>3} | "
            f"{str(process_name):30s} | "
            f"{str(domain):8s} | "
            f"{count:>3}"
        )

    print()
    print("=" * 70)
    print("MISSING END TIMESTAMPS")
    print("=" * 70)

    missing_end = df["end_ts"].isna()

    print(
        f"Executions without end_ts: "
        f"{missing_end.sum()}"
    )

    if missing_end.any():
        display_columns = [
            "session_id",
            "process_code",
            "process_name",
            "case_id",
            "start_ts",
            "end_ts",
            "phase",
            "exec_id",
            "split_id",
        ]

        print()
        print(
            df.loc[
                missing_end,
                display_columns,
            ].to_string(index=False)
        )

    print()
    print("=" * 70)
    print("VARIANTS")
    print("=" * 70)

    variant_counts = (
        df.groupby(
            ["process_code", "variant"],
            dropna=False,
        )
        .size()
        .sort_values(ascending=False)
    )

    for index, count in variant_counts.items():
        process_code, variant = index
        print(
            f"Process={process_code:>3} | "
            f"variant={str(variant):8s} | "
            f"{count:>3}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()