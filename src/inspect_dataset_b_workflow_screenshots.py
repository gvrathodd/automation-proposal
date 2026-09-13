from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json
import re

import pandas as pd

from loader import (
    load_all_events,
    sort_session_events,
    get_timestamp_ms,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
INSPECTION_FILE = PROJECT_ROOT / "outputs" / "dataset_b_workflow_inspection.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "dataset_b_screenshot_inspection.csv"

TARGETS = [
    {
        "session_id": "ses_20260701-175258-LAPTOP-76QMG9DE",
        "segment_index": 11,
        "label": "Edge > Notepad > Edge",
    },
    {
        "session_id": "ses_20260701-180923-NEELA9BAF",
        "segment_index": 23,
        "label": "Edge > Notepad > Edge",
    },
    {
        "session_id": "ses_20260701-183232-LAPTOP-76QMG9DE",
        "segment_index": 1,
        "label": "Edge > Notepad > Edge",
    },
]

MAX_SCREENSHOTS_PER_SEGMENT = 24
EVENT_MARGIN_SECONDS = 2
MAX_TEXT_LENGTH = 220


def get_event_type(event):
    value = event.get("event_type")
    return value if isinstance(value, str) else ""


def get_event_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}
    value = active_app.get("app_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "UNKNOWN"


def get_screenshot_filenames(event):
    payload = event.get("payload") or {}
    reference = payload.get("file_reference")
    if not reference:
        return []
    out = []
    if isinstance(reference, dict):
        value = reference.get("filename")
        if isinstance(value, str):
            out.append(value)
    elif isinstance(reference, list):
        for item in reference:
            if isinstance(item, dict):
                value = item.get("filename")
                if isinstance(value, str):
                    out.append(value)
    return out


def get_context_text(event):
    context = event.get("context") or {}
    value = context.get("extracted_text")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def get_browser_url(event):
    payload = event.get("payload") or {}
    context = event.get("context") or {}
    for value in [
        payload.get("url"),
        payload.get("current_url"),
        payload.get("target_url"),
        context.get("url"),
        context.get("current_url"),
        context.get("browser_url"),
    ]:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def truncate(value, limit=MAX_TEXT_LENGTH):
    value = re.sub(r"\s+", " ", value or "").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def get_timestamp_iso(event):
    value = event.get("timestamp_iso")
    if isinstance(value, str):
        return value
    timestamp_ms = get_timestamp_ms(event)
    if timestamp_ms is None:
        return ""
    return pd.to_datetime(timestamp_ms, unit="ms", utc=True).isoformat()


def build_screenshot_index():
    print("Indexing screenshots...")
    index = defaultdict(list)
    for path in DATA_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            index[path.name].append(str(path))
    print(f"Unique screenshot filenames: {len(index):,}")
    return index


def prepare_events(events):
    prepared = []
    for event in sort_session_events(events):
        timestamp_ms = get_timestamp_ms(event)
        if timestamp_ms is None:
            continue
        prepared.append(
            {
                "timestamp_ms": int(timestamp_ms),
                "timestamp_iso": get_timestamp_iso(event),
                "event_type": get_event_type(event),
                "app": get_event_app(event),
                "screenshots": get_screenshot_filenames(event),
                "text": get_context_text(event),
                "url": get_browser_url(event),
                "raw": event,
            }
        )
    return prepared


def load_targets():
    if not INSPECTION_FILE.exists():
        raise FileNotFoundError(f"Inspection file not found:\n{INSPECTION_FILE}")
    inspection = pd.read_csv(INSPECTION_FILE, keep_default_na=False)
    required = {"session_id", "segment_index", "start", "end"}
    missing = required - set(inspection.columns)
    if missing:
        raise RuntimeError("Inspection file is missing:\n" + "\n".join(sorted(missing)))
    inspection["segment_index"] = pd.to_numeric(inspection["segment_index"], errors="raise").astype(int)
    inspection["start"] = pd.to_datetime(inspection["start"], utc=True)
    inspection["end"] = pd.to_datetime(inspection["end"], utc=True)
    rows = []
    for target in TARGETS:
        matches = inspection[
            (inspection["session_id"] == target["session_id"])
            & (inspection["segment_index"] == target["segment_index"])
        ]
        if matches.empty:
            raise RuntimeError(f"Target segment not found: {target}")
        row = matches.iloc[0].copy()
        row["target_label"] = target["label"]
        rows.append(row)
    return pd.DataFrame(rows)


def inspect_target(target_row, prepared_events, screenshot_index):
    start = pd.Timestamp(target_row["start"])
    end = pd.Timestamp(target_row["end"])
    margin = pd.Timedelta(seconds=EVENT_MARGIN_SECONDS)
    start_ms = int((start - margin).timestamp() * 1000)
    end_ms = int((end + margin).timestamp() * 1000)

    relevant = [
        event for event in prepared_events
        if start_ms <= event["timestamp_ms"] <= end_ms
    ]

    screenshot_records = []
    seen = set()
    for event in relevant:
        for filename in event["screenshots"]:
            if filename in seen:
                continue
            seen.add(filename)
            screenshot_records.append(
                {
                    "filename": filename,
                    "timestamp": event["timestamp_iso"],
                    "event_type": event["event_type"],
                    "app": event["app"],
                    "text": truncate(event["text"]),
                    "url": event["url"],
                    "paths": screenshot_index.get(filename, []),
                }
            )

    screenshot_records.sort(key=lambda x: x["timestamp"])
    return relevant, screenshot_records[:MAX_SCREENSHOTS_PER_SEGMENT]


def build_event_timeline(events):
    lines = []
    for event in events[:120]:
        line = f"{event['timestamp_iso']} | {event['event_type']} | {event['app']}"
        if event["url"]:
            line += f" | URL={event['url']}"
        if event["text"]:
            line += f" | TEXT={truncate(event['text'])}"
        lines.append(line)
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("DATASET B SCREENSHOT INSPECTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()

    print(f"Selected target segments: {len(targets)}")
    print("Loading raw Dataset B sessions...")

    all_sessions = load_all_events(DATA_ROOT)
    dataset_b_sessions = {}
    for session_id, events in all_sessions.items():
        datasets = {event.get("_dataset") for event in events if event.get("_dataset")}
        if "B" in datasets:
            dataset_b_sessions[session_id] = events

    print(f"Dataset B sessions: {len(dataset_b_sessions):,}")

    screenshot_index = build_screenshot_index()

    prepared_sessions = {}
    for session_id in set(targets["session_id"]):
        if session_id not in dataset_b_sessions:
            raise RuntimeError(f"Dataset B session not found: {session_id}")
        prepared_sessions[session_id] = prepare_events(dataset_b_sessions[session_id])

    output_rows = []

    for _, target in targets.iterrows():
        session_id = target["session_id"]
        relevant, screenshots = inspect_target(
            target,
            prepared_sessions[session_id],
            screenshot_index,
        )

        paths = []
        metadata = []
        for order, record in enumerate(screenshots, start=1):
            paths.extend(record["paths"])
            metadata.append(
                {
                    "order": order,
                    "filename": record["filename"],
                    "timestamp": record["timestamp"],
                    "event_type": record["event_type"],
                    "app": record["app"],
                    "text": record["text"],
                    "url": record["url"],
                    "paths": record["paths"],
                }
            )

        paths = list(dict.fromkeys(paths))

        output_rows.append(
            {
                "target_label": target["target_label"],
                "session_id": session_id,
                "segment_index": int(target["segment_index"]),
                "start": target["start"],
                "end": target["end"],
                "duration_seconds": float(target["duration_seconds"]),
                "app_sequence": target.get("app_sequence", ""),
                "relevant_event_count": len(relevant),
                "screenshot_count": len(screenshots),
                "screenshot_metadata_json": json.dumps(metadata, ensure_ascii=False, indent=2),
                "screenshot_paths": "\n".join(paths),
                "event_timeline": build_event_timeline(relevant),
            }
        )

        print()
        print(
            f"{target['target_label']} | {session_id} | "
            f"segment={int(target['segment_index'])}"
        )
        print(f"  duration: {float(target['duration_seconds']):.1f}s")
        print(f"  screenshots: {len(screenshots)}")
        print(f"  relevant events: {len(relevant)}")

    result = pd.DataFrame(output_rows)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("=" * 70)
    print("SCREENSHOTS TO INSPECT")
    print("=" * 70)

    for _, row in result.iterrows():
        print()
        print(
            f"### {row['target_label']} | "
            f"{row['session_id']} | segment {row['segment_index']}"
        )
        for path in row["screenshot_paths"].splitlines():
            if path:
                print(path)

    print()
    print("=" * 70)
    print("SCREENSHOT INSPECTION COMPLETE")
    print("=" * 70)
    print(f"Saved:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
