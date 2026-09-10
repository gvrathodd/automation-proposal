from pathlib import Path
import json
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
BOUNDARY_FILE = PROJECT_ROOT / "outputs" / "gt_boundaries.csv"

OUTPUT_FILE = PROJECT_ROOT / "outputs" / "boundary_vs_control.csv"
SUMMARY_FILE = PROJECT_ROOT / "outputs" / "boundary_signal_summary.csv"

WINDOW_SECONDS = 10
RANDOM_SEED = 42


def load_dataset_a_events():
    """Load all Dataset A events."""
    records = []

    for path in DATA_ROOT.rglob("events.jsonl"):
        if "dataset_b" in str(path).lower():
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    event = json.loads(line)

                    session_id = event.get("session_id")
                    timestamp = event.get("timestamp_iso")

                    if session_id and timestamp:
                        records.append(
                            {
                                "session_id": session_id,
                                "timestamp": pd.to_datetime(
                                    timestamp, utc=True
                                ),
                                "event_type": event.get("event_type"),
                                "event": event,
                            }
                        )

        except Exception as exc:
            print(f"Could not read {path}: {exc}")

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No Dataset A events found.")

    return (
        df.sort_values(["session_id", "timestamp"])
        .reset_index(drop=True)
    )


def get_active_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}
    return active_app.get("app_name")


def get_window_title(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}
    return active_app.get("window_title")


def summarize_window(events, session_id, anchor_time, label):
    """Convert a +/- WINDOW_SECONDS window into numeric features."""

    start = anchor_time - pd.Timedelta(seconds=WINDOW_SECONDS)
    end = anchor_time + pd.Timedelta(seconds=WINDOW_SECONDS)

    nearby = events[
        (events["session_id"] == session_id)
        & (events["timestamp"] >= start)
        & (events["timestamp"] <= end)
    ].copy()

    if nearby.empty:
        return None

    event_types = nearby["event_type"].fillna("unknown")

    apps = [
        get_active_app(event)
        for event in nearby["event"]
    ]
    apps = [app for app in apps if app]

    windows = [
        get_window_title(event)
        for event in nearby["event"]
    ]
    windows = [title for title in windows if title]

    timestamps = nearby["timestamp"].sort_values()

    if len(timestamps) >= 2:
        gaps = timestamps.diff().dt.total_seconds().dropna()
        max_gap = gaps.max()
        median_gap = gaps.median()
    else:
        max_gap = 0.0
        median_gap = 0.0

    return {
        "session_id": session_id,
        "anchor_time": anchor_time,
        "label": label,

        "event_count": len(nearby),
        "unique_event_types": event_types.nunique(),
        "unique_apps": len(set(apps)),
        "unique_window_titles": len(set(windows)),

        "app_switch_count": (
            event_types == "app_switch"
        ).sum(),

        "window_title_change_count": (
            event_types == "window_title_change"
        ).sum(),

        "browser_navigation_count": (
            event_types == "browser_navigation"
        ).sum(),

        "browser_click_count": (
            event_types == "browser_click"
        ).sum(),

        "browser_form_input_count": (
            event_types == "browser_form_input"
        ).sum(),

        "clipboard_change_count": (
            event_types == "clipboard_change"
        ).sum(),

        "keystroke_count": (
            event_types == "keystroke"
        ).sum(),

        "mouse_click_count": (
            event_types == "mouse_click"
        ).sum(),

        "mouse_scroll_count": (
            event_types == "mouse_scroll"
        ).sum(),

        "screenshot_count": (
            event_types == "screenshot_smart"
        ).sum(),

        "max_event_gap_seconds": max_gap,
        "median_event_gap_seconds": median_gap,
    }


def build_control_times(events, boundaries):
    """
    Choose random timestamps that are not close to a GT boundary.
    We use these as normal/non-boundary examples.
    """

    rng = random.Random(RANDOM_SEED)

    boundary_lookup = {}

    for _, row in boundaries.iterrows():
        session_id = row["session_id"]
        ts = pd.to_datetime(row["ts"], utc=True)

        boundary_lookup.setdefault(session_id, []).append(ts)

    controls = []

    for session_id, session_events in events.groupby("session_id"):
        session_events = session_events.sort_values("timestamp")

        first_time = session_events["timestamp"].iloc[0]
        last_time = session_events["timestamp"].iloc[-1]

        boundaries_for_session = boundary_lookup.get(
            session_id, []
        )

        candidates = session_events[
            (session_events["timestamp"] >= first_time + pd.Timedelta(seconds=WINDOW_SECONDS))
            & (session_events["timestamp"] <= last_time - pd.Timedelta(seconds=WINDOW_SECONDS))
        ]["timestamp"].tolist()

        valid_candidates = []

        for ts in candidates:
            too_close = any(
                abs((ts - boundary_ts).total_seconds())
                <= WINDOW_SECONDS
                for boundary_ts in boundaries_for_session
            )

            if not too_close:
                valid_candidates.append(ts)

        # Match the number of controls to the number of GT boundaries,
        # up to the number of available candidates.
        target_count = min(
            len(valid_candidates),
            len(boundaries_for_session),
        )

        if target_count == 0:
            continue

        sampled = rng.sample(
            valid_candidates,
            target_count,
        )

        for ts in sampled:
            controls.append(
                {
                    "session_id": session_id,
                    "anchor_time": ts,
                }
            )

    return pd.DataFrame(controls)


def main():
    print("Loading Dataset A events...")
    events = load_dataset_a_events()

    print(f"Loaded {len(events):,} Dataset A events.")

    print("Loading ground-truth boundaries...")
    boundaries = pd.read_csv(BOUNDARY_FILE)
    boundaries["ts"] = pd.to_datetime(
        boundaries["ts"],
        utc=True,
    )

    print(f"Loaded {len(boundaries):,} GT boundaries.")

    # Remove duplicate timestamps within a session.
    # Multiple GT records can describe effectively the same transition moment.
    boundary_points = (
        boundaries[
            ["session_id", "ts"]
        ]
        .drop_duplicates()
        .rename(columns={"ts": "anchor_time"})
        .reset_index(drop=True)
    )

    print(
        f"Unique boundary timestamps: "
        f"{len(boundary_points):,}"
    )

    print("Building matched non-boundary control points...")
    controls = build_control_times(
        events,
        boundaries,
    )

    print(
        f"Control timestamps: "
        f"{len(controls):,}"
    )

    results = []

    print("Extracting boundary features...")

    for _, row in boundary_points.iterrows():
        result = summarize_window(
            events,
            row["session_id"],
            row["anchor_time"],
            "boundary",
        )

        if result is not None:
            results.append(result)

    print("Extracting control features...")

    for _, row in controls.iterrows():
        result = summarize_window(
            events,
            row["session_id"],
            row["anchor_time"],
            "control",
        )

        if result is not None:
            results.append(result)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Compare average values between boundary and control windows.
    feature_columns = [
        "event_count",
        "unique_event_types",
        "unique_apps",
        "unique_window_titles",
        "app_switch_count",
        "window_title_change_count",
        "browser_navigation_count",
        "browser_click_count",
        "browser_form_input_count",
        "clipboard_change_count",
        "keystroke_count",
        "mouse_click_count",
        "mouse_scroll_count",
        "screenshot_count",
        "max_event_gap_seconds",
        "median_event_gap_seconds",
    ]

    summary_rows = []

    boundary_df = result_df[
        result_df["label"] == "boundary"
    ]

    control_df = result_df[
        result_df["label"] == "control"
    ]

    for feature in feature_columns:
        boundary_mean = boundary_df[feature].mean()
        control_mean = control_df[feature].mean()

        if control_mean != 0:
            ratio = boundary_mean / control_mean
        else:
            ratio = None

        summary_rows.append(
            {
                "feature": feature,
                "boundary_mean": boundary_mean,
                "control_mean": control_mean,
                "boundary_to_control_ratio": ratio,
                "difference": boundary_mean - control_mean,
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "boundary_to_control_ratio",
            ascending=False,
            na_position="last",
        )
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")

    print()
    print("Top signals:")
    print(
        summary_df.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()