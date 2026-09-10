from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
BOUNDARY_FILE = PROJECT_ROOT / "outputs" / "gt_boundaries.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "boundary_signal_analysis.csv"

# Seconds before/after each ground-truth boundary to inspect.
WINDOW_SECONDS = 10


def load_events():
    """Load all events and keep Dataset A only."""
    records = []

    for path in DATA_ROOT.rglob("events.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    event = json.loads(line)
                    session_id = event.get("session_id")

                    # Dataset can be inferred from the path.
                    dataset = "B" if "dataset_b" in str(path).lower() else "A"

                    if session_id and dataset == "A":
                        records.append(
                            {
                                "session_id": session_id,
                                "timestamp": event.get("timestamp_iso"),
                                "timestamp_ms": event.get("timestamp_ms"),
                                "event_type": event.get("event_type"),
                                "event": event,
                            }
                        )

        except Exception as exc:
            print(f"Could not read {path}: {exc}")

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No Dataset A events were loaded.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    return df


def get_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}
    return active_app.get("app_name")


def get_window_title(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}
    return active_app.get("window_title")


def get_url(event):
    context = event.get("context") or {}
    browser = context.get("browser") or {}
    return browser.get("url")


def analyse_boundary(events, boundary):
    session_id = boundary["session_id"]
    boundary_time = pd.to_datetime(boundary["ts"], utc=True)

    session_events = events[events["session_id"] == session_id].copy()

    if session_events.empty:
        return []

    start = boundary_time - pd.Timedelta(seconds=WINDOW_SECONDS)
    end = boundary_time + pd.Timedelta(seconds=WINDOW_SECONDS)

    nearby = session_events[
        (session_events["timestamp"] >= start)
        & (session_events["timestamp"] <= end)
    ].copy()

    results = []

    for _, row in nearby.iterrows():
        event = row["event"]

        results.append(
            {
                "session_id": session_id,
                "boundary_ts": boundary_time,
                "boundary_event": boundary["event"],
                "boundary_from": boundary.get("from"),
                "boundary_to": boundary.get("to"),
                "event_ts": row["timestamp"],
                "offset_seconds": (
                    row["timestamp"] - boundary_time
                ).total_seconds(),
                "event_type": row["event_type"],
                "active_app": get_app(event),
                "window_title": get_window_title(event),
                "browser_url": get_url(event),
            }
        )

    return results


def main():
    print("Loading Dataset A events...")
    events = load_events()

    print(f"Loaded {len(events):,} Dataset A events.")

    print("Loading ground-truth boundaries...")
    boundaries = pd.read_csv(BOUNDARY_FILE)

    boundaries["ts"] = pd.to_datetime(boundaries["ts"], utc=True)

    print(f"Loaded {len(boundaries):,} ground-truth boundary events.")

    all_results = []

    for _, boundary in boundaries.iterrows():
        all_results.extend(
            analyse_boundary(events, boundary)
        )

    result_df = pd.DataFrame(all_results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows written: {len(result_df):,}")

    print()
    print("Boundary event types:")
    print(boundaries["event"].value_counts())

    print()
    print("Observable event types near boundaries:")
    print(result_df["event_type"].value_counts().head(20))


if __name__ == "__main__":
    main()