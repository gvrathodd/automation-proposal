from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"


def read_jsonl(path: Path):
    """Read a JSONL file one record at a time."""
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARNING] Invalid JSON at "
                    f"{path}:{line_number}: {exc}"
                )


def discover_event_files(root: Path) -> list[Path]:
    """Find all events.jsonl files."""
    return sorted(root.rglob("events.jsonl"))


def identify_dataset(path: Path) -> str | None:
    """Return A, B, or None based on the path."""
    for part in path.parts:
        if part.lower() == "dataset_a":
            return "A"
        if part.lower() == "dataset_b":
            return "B"

    return None


def load_all_events(root: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Load all events and group them by session_id.

    Physical chunk/file layout is ignored for ordering.
    Events are later sorted using timestamp_ms.
    """
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    event_files = discover_event_files(root)

    for path in event_files:
        dataset = identify_dataset(path)

        for event in read_jsonl(path):
            event["_dataset"] = dataset
            event["_source_file"] = str(path)

            session_id = event.get("session_id")

            if session_id:
                sessions[session_id].append(event)

    return dict(sessions)


def sort_session_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Sort events chronologically.

    timestamp_ms is the primary ordering key.
    sequence_number is retained only as a secondary key.
    """
    return sorted(
        events,
        key=lambda e: (
            e.get("timestamp_ms", float("inf")),
            e.get("correlation", {}).get(
                "sequence_number",
                float("inf"),
            ),
        ),
    )


def get_timestamp_ms(event: dict[str, Any]) -> int | None:
    value = event.get("timestamp_ms")

    if isinstance(value, (int, float)):
        return int(value)

    return None


def get_chunk_id(event: dict[str, Any]) -> str | None:
    value = event.get("correlation", {}).get("chunk_id")

    return value if isinstance(value, str) else None


def build_session_summary(
    session_id: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact summary of one reconstructed session."""
    events = sort_session_events(events)

    timestamps = [
        get_timestamp_ms(event)
        for event in events
    ]

    timestamps = [
        ts for ts in timestamps
        if ts is not None
    ]

    chunk_ids = sorted(
        {
            chunk_id
            for event in events
            if (chunk_id := get_chunk_id(event))
        }
    )

    dataset_values = {
        event.get("_dataset")
        for event in events
        if event.get("_dataset")
    }

    event_types: dict[str, int] = defaultdict(int)

    for event in events:
        event_type = event.get("event_type")

        if isinstance(event_type, str):
            event_types[event_type] += 1

    return {
        "session_id": session_id,
        "dataset": sorted(dataset_values),
        "event_count": len(events),
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "first_timestamp_ms": min(timestamps) if timestamps else None,
        "last_timestamp_ms": max(timestamps) if timestamps else None,
        "event_types": dict(
            sorted(event_types.items())
        ),
    }


def format_timestamp(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return "N/A"

    from datetime import datetime, timezone

    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        timezone.utc,
    ).isoformat()


def print_session_preview(
    session_id: str,
    events: list[dict[str, Any]],
    limit: int = 20,
) -> None:
    """
    Print the first few events from a reconstructed
    chronological session.
    """
    events = sort_session_events(events)

    print()
    print("=" * 80)
    print(f"SESSION PREVIEW: {session_id}")
    print("=" * 80)

    for event in events[:limit]:
        timestamp = format_timestamp(
            get_timestamp_ms(event)
        )

        event_type = event.get(
            "event_type",
            "UNKNOWN",
        )

        layer = event.get(
            "layer",
            "UNKNOWN",
        )

        context = event.get("context") or {}
        active_app = context.get("active_app") or {}

        app = active_app.get("app_name")

        chunk = get_chunk_id(event)

        sequence_number = (
            event.get("correlation", {})
            .get("sequence_number")
        )

        print(
            f"{timestamp} | "
            f"{layer:6s} | "
            f"{event_type:24s} | "
            f"app={str(app):20s} | "
            f"chunk={str(chunk):38s} | "
            f"seq={sequence_number}"
        )


def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATA_ROOT}"
        )

    print("=" * 80)
    print("CANONICAL SESSION RECONSTRUCTION")
    print("=" * 80)
    print(f"Data root: {DATA_ROOT}")

    sessions = load_all_events(DATA_ROOT)

    print()
    print(f"Sessions reconstructed: {len(sessions)}")

    summaries = []

    for session_id, events in sessions.items():
        summaries.append(
            build_session_summary(
                session_id,
                events,
            )
        )

    summaries.sort(
        key=lambda x: (
            x["dataset"],
            x["session_id"],
        )
    )

    # Dataset-level summary
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for summary in summaries:
        for dataset in summary["dataset"]:
            by_dataset[dataset].append(summary)

    print()
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    for dataset in ("A", "B"):
        dataset_summaries = by_dataset.get(
            dataset,
            [],
        )

        total_events = sum(
            s["event_count"]
            for s in dataset_summaries
        )

        multi_chunk = sum(
            1
            for s in dataset_summaries
            if s["chunk_count"] > 1
        )

        print(
            f"Dataset {dataset}: "
            f"{len(dataset_summaries)} sessions, "
            f"{total_events:,} events, "
            f"{multi_chunk} multi-chunk sessions"
        )

    # Show one known multi-chunk Dataset A session
    candidate_sessions = [
        summary
        for summary in summaries
        if "A" in summary["dataset"]
        and summary["chunk_count"] > 1
    ]

    if candidate_sessions:
        example = candidate_sessions[0]

        session_id = example["session_id"]

        print()
        print("=" * 80)
        print("MULTI-CHUNK SESSION EXAMPLE")
        print("=" * 80)

        print(f"Session: {session_id}")
        print(
            f"Chunks: {example['chunk_count']}"
        )

        print("Chunk IDs:")

        for chunk_id in example["chunk_ids"]:
            print(f"  {chunk_id}")

        print(
            f"First event: "
            f"{format_timestamp(example['first_timestamp_ms'])}"
        )

        print(
            f"Last event:  "
            f"{format_timestamp(example['last_timestamp_ms'])}"
        )

        print_session_preview(
            session_id,
            sessions[session_id],
            limit=20,
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()