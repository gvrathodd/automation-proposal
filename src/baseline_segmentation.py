from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
GT_FILE = PROJECT_ROOT / "outputs" / "gt_boundaries.csv"

OUTPUT_FILE = PROJECT_ROOT / "outputs" / "baseline_boundaries.csv"
SUMMARY_FILE = PROJECT_ROOT / "outputs" / "baseline_evaluation.csv"


# We aggregate activity in short time bins.
BIN_SECONDS = 2

# Score threshold for calling a candidate boundary.
# We will tune this after seeing the first results.
BOUNDARY_THRESHOLD = 3.0


def load_events():
    """Load and chronologically sort all Dataset A events."""
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

                    if not session_id or not timestamp:
                        continue

                    records.append(
                        {
                            "session_id": session_id,
                            "timestamp": pd.to_datetime(
                                timestamp,
                                utc=True,
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


def build_bins(events):
    """
    Aggregate raw events into fixed 2-second bins.

    This is only a baseline representation. We are not assuming
    that business processes themselves are 2 seconds long.
    """

    rows = []

    for session_id, group in events.groupby("session_id"):
        group = group.sort_values("timestamp").copy()

        session_start = group["timestamp"].iloc[0]

        # Seconds elapsed since session start.
        group["elapsed_seconds"] = (
            group["timestamp"] - session_start
        ).dt.total_seconds()

        group["bin_number"] = (
            group["elapsed_seconds"] // BIN_SECONDS
        ).astype(int)

        for bin_number, bin_events in group.groupby("bin_number"):
            event_types = bin_events["event_type"].fillna("unknown")

            apps = [
                get_active_app(event)
                for event in bin_events["event"]
            ]
            apps = [app for app in apps if app]

            windows = [
                get_window_title(event)
                for event in bin_events["event"]
            ]
            windows = [window for window in windows if window]

            rows.append(
                {
                    "session_id": session_id,
                    "bin_number": int(bin_number),
                    "bin_start": (
                        session_start
                        + pd.Timedelta(
                            seconds=bin_number * BIN_SECONDS
                        )
                    ),

                    "event_count": len(bin_events),
                    "unique_apps": len(set(apps)),
                    "unique_windows": len(set(windows)),

                    "app_switches": (
                        event_types == "app_switch"
                    ).sum(),

                    "browser_navigation": (
                        event_types == "browser_navigation"
                    ).sum(),

                    "browser_clicks": (
                        event_types == "browser_click"
                    ).sum(),

                    "mouse_clicks": (
                        event_types == "mouse_click"
                    ).sum(),

                    "window_changes": (
                        event_types == "window_title_change"
                    ).sum(),
                }
            )

    return pd.DataFrame(rows)


def calculate_scores(bins):
    """
    Simple interpretable weighted score.

    Stronger signals receive larger weights based on our
    boundary-vs-control analysis.
    """

    bins = bins.copy()

    bins["boundary_score"] = (
        1.0 * bins["app_switches"]
        + 1.0 * bins["browser_navigation"]
        + 0.5 * bins["browser_clicks"]
        + 0.5 * bins["mouse_clicks"]
        + 0.25 * bins["window_changes"]
    )

    return bins


def select_boundaries(bins):
    """
    Convert high-scoring bins into candidate boundary timestamps.

    Adjacent high-scoring bins are merged into one boundary.
    """

    candidates = bins[
        bins["boundary_score"] >= BOUNDARY_THRESHOLD
    ].copy()

    predictions = []

    for session_id, group in candidates.groupby("session_id"):
        group = group.sort_values("bin_start")

        current_cluster = []
        previous_bin = None

        for _, row in group.iterrows():
            bin_number = row["bin_number"]

            if (
                previous_bin is None
                or bin_number <= previous_bin + 1
            ):
                current_cluster.append(row)
            else:
                cluster_df = pd.DataFrame(current_cluster)

                best_row = cluster_df.loc[
                    cluster_df["boundary_score"].idxmax()
                ]

                predictions.append(
                    {
                        "session_id": session_id,
                        "predicted_ts": best_row["bin_start"],
                        "score": best_row["boundary_score"],
                    }
                )

                current_cluster = [row]

            previous_bin = bin_number

        if current_cluster:
            cluster_df = pd.DataFrame(current_cluster)

            best_row = cluster_df.loc[
                cluster_df["boundary_score"].idxmax()
            ]

            predictions.append(
                {
                    "session_id": session_id,
                    "predicted_ts": best_row["bin_start"],
                    "score": best_row["boundary_score"],
                }
            )

    return pd.DataFrame(predictions)


def evaluate(predictions, gt_boundaries, tolerance_seconds=5):
    """
    Evaluate predicted boundary timestamps against GT.

    A prediction counts as a match when it falls within the
    tolerance window of an unmatched GT boundary.
    """

    gt = (
        gt_boundaries[
            ["session_id", "ts"]
        ]
        .drop_duplicates()
        .copy()
    )

    gt["ts"] = pd.to_datetime(gt["ts"], utc=True)

    matched_gt = set()
    matched_predictions = 0

    prediction_rows = []

    for _, prediction in predictions.iterrows():
        session_id = prediction["session_id"]
        predicted_ts = prediction["predicted_ts"]

        session_gt = gt[
            gt["session_id"] == session_id
        ].copy()

        if session_gt.empty:
            prediction_rows.append(
                {
                    "session_id": session_id,
                    "predicted_ts": predicted_ts,
                    "matched": False,
                    "error_seconds": None,
                }
            )
            continue

        best_index = None
        best_error = None

        for gt_index, gt_row in session_gt.iterrows():
            gt_ts = gt_row["ts"]

            key = (session_id, gt_index)

            if key in matched_gt:
                continue

            error = abs(
                (predicted_ts - gt_ts).total_seconds()
            )

            if error <= tolerance_seconds:
                if best_error is None or error < best_error:
                    best_error = error
                    best_index = gt_index

        if best_index is not None:
            matched_gt.add((session_id, best_index))
            matched_predictions += 1

            prediction_rows.append(
                {
                    "session_id": session_id,
                    "predicted_ts": predicted_ts,
                    "matched": True,
                    "error_seconds": best_error,
                }
            )
        else:
            prediction_rows.append(
                {
                    "session_id": session_id,
                    "predicted_ts": predicted_ts,
                    "matched": False,
                    "error_seconds": None,
                }
            )

    total_gt = len(gt)
    total_predictions = len(predictions)

    precision = (
        matched_predictions / total_predictions
        if total_predictions
        else 0
    )

    recall = (
        matched_predictions / total_gt
        if total_gt
        else 0
    )

    if precision + recall:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0

    evaluation = pd.DataFrame(prediction_rows)

    metrics = {
        "gt_boundaries": total_gt,
        "predicted_boundaries": total_predictions,
        "matched_boundaries": matched_predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_boundary_error_seconds": (
            evaluation.loc[
                evaluation["matched"],
                "error_seconds"
            ].mean()
            if matched_predictions
            else None
        ),
    }

    return evaluation, metrics


def main():
    print("Loading Dataset A events...")
    events = load_events()

    print(f"Loaded {len(events):,} events.")

    print("Building time bins...")
    bins = build_bins(events)

    print(f"Built {len(bins):,} time bins.")

    print("Calculating boundary scores...")
    bins = calculate_scores(bins)

    print("Selecting candidate boundaries...")
    predictions = select_boundaries(bins)

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Predicted boundaries: "
        f"{len(predictions):,}"
    )

    print("Loading ground truth...")
    gt = pd.read_csv(GT_FILE)

    evaluation, metrics = evaluate(
        predictions,
        gt,
        tolerance_seconds=5,
    )

    print()
    print("Baseline evaluation")
    print("-------------------")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    evaluation.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Saved predictions: {OUTPUT_FILE}")
    print(f"Saved evaluation: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()