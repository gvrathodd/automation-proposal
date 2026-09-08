from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return its records."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARNING] Invalid JSON: {path}:{line_number}: {exc}")

    return records


def discover_event_files(root: Path) -> list[Path]:
    """Find every events.jsonl under the raw download directory."""
    return sorted(root.rglob("events.jsonl"))


def discover_gt_files(root: Path) -> list[Path]:
    """Find every gt.jsonl under the raw download directory."""
    return sorted(root.rglob("gt.jsonl"))


def discover_gt_manifest_files(root: Path) -> list[Path]:
    """Find every gt_manifest.json under the raw download directory."""
    return sorted(root.rglob("gt_manifest.json"))


def discover_screenshots(root: Path) -> dict[str, list[Path]]:
    """
    Build an index:
        screenshot filename -> list of physical files

    A list is used because duplicate filenames can theoretically exist
    in different downloaded packages.
    """
    index: dict[str, list[Path]] = defaultdict(list)

    for path in root.rglob("*.jpg"):
        index[path.name].append(path)

    return dict(index)


def summarize_event_file(path: Path) -> dict[str, Any]:
    """Extract basic information from one events.jsonl."""
    records = read_jsonl(path)

    session_ids = {
        record.get("session_id")
        for record in records
        if record.get("session_id")
    }

    chunk_ids = {
        record.get("correlation", {}).get("chunk_id")
        for record in records
        if record.get("correlation", {}).get("chunk_id")
    }

    event_types = Counter(
        record.get("event_type")
        for record in records
        if record.get("event_type")
    )

    timestamps = [
        record["timestamp_ms"]
        for record in records
        if isinstance(record.get("timestamp_ms"), (int, float))
    ]

    return {
        "path": path,
        "events": len(records),
        "session_ids": sorted(session_ids),
        "chunk_ids": sorted(chunk_ids),
        "event_types": event_types,
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }


def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Data directory not found:\n{DATA_ROOT}"
        )

    print("=" * 70)
    print("DATASET DISCOVERY")
    print("=" * 70)
    print(f"Data root: {DATA_ROOT}")
    print()

    event_files = discover_event_files(DATA_ROOT)
    gt_files = discover_gt_files(DATA_ROOT)
    gt_manifest_files = discover_gt_manifest_files(DATA_ROOT)
    screenshot_index = discover_screenshots(DATA_ROOT)

    print(f"events.jsonl files found:      {len(event_files)}")
    print(f"gt.jsonl files found:           {len(gt_files)}")
    print(f"gt_manifest.json files found:   {len(gt_manifest_files)}")
    print(f"unique screenshot filenames:    {len(screenshot_index)}")
    print()

    print("=" * 70)
    print("EVENT FILES")
    print("=" * 70)

    total_events = 0
    sessions: set[str] = set()

    for path in event_files:
        summary = summarize_event_file(path)

        total_events += summary["events"]
        sessions.update(summary["session_ids"])

        print(f"\nFile: {path.relative_to(DATA_ROOT)}")
        print(f"  events:      {summary['events']}")
        print(f"  sessions:    {summary['session_ids']}")
        print(f"  chunks:      {summary['chunk_ids']}")
        print(f"  first ts:    {summary['first_timestamp']}")
        print(f"  last ts:     {summary['last_timestamp']}")
        print(f"  event types: {dict(summary['event_types'])}")

    print()
    print("=" * 70)
    print("TOTALS")
    print("=" * 70)
    print(f"Total events discovered: {total_events}")
    print(f"Unique sessions discovered: {len(sessions)}")

    print()
    print("=" * 70)
    print("SCREENSHOT INDEX CHECK")
    print("=" * 70)

    duplicate_names = {
        name: paths
        for name, paths in screenshot_index.items()
        if len(paths) > 1
    }

    print(f"Duplicate screenshot filenames: {len(duplicate_names)}")

    if duplicate_names:
        print("\nExamples:")
        for name, paths in list(duplicate_names.items())[:5]:
            print(f"  {name}")
            for path in paths:
                print(f"    - {path.relative_to(DATA_ROOT)}")


if __name__ == "__main__":
    main()