from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
GT_FILE = PROJECT_ROOT / "outputs" / "gt_boundaries.csv"

PREDICTION_FILE = PROJECT_ROOT / "outputs" / "baseline_v2_boundaries.csv"
EVALUATION_FILE = PROJECT_ROOT / "outputs" / "baseline_v2_evaluation.csv"
FEATURE_FILE = PROJECT_ROOT / "outputs" / "baseline_v2_features.csv"

BIN_SECONDS = 2

# Compare 6 seconds before and 6 seconds after each candidate point.
CONTEXT_SECONDS = 6

# Initial threshold. We will tune only after seeing results.
BOUNDARY_THRESHOLD = 2.0


def load_events():
    """Load all Dataset A events in chronological order."""
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


def get_browser_url(event):
    context = event.get("context") or {}
    browser = context.get("browser") or {}

    return (
        browser.get("url")
        or browser.get("current_url")
    )


def window_features(events, start, end):
    """Summarize observable activity inside one time window."""

    nearby = events[
        (events["timestamp"] >= start)
        & (events["timestamp"] < end)
    ]

    if nearby.empty:
        return {
            "event_count": 0,
            "app_switches": 0,
            "browser_navigation": 0,
            "browser_clicks": 0,
            "mouse_clicks": 0,
            "window_changes": 0,
            "unique_apps": 0,
            "unique_windows": 0,
            "unique_urls": 0,
            "keyboard_activity": 0,
        }

    event_types = nearby["event_type"].fillna("unknown")

    apps = [
        get_active_app(event)
        for event in nearby["event"]
    ]
    apps = [x for x in apps if x]

    windows = [
        get_window_title(event)
        for event in nearby["event"]
    ]
    windows = [x for x in windows if x]

    urls = [
        get_browser_url(event)
        for event in nearby["event"]
    ]
    urls = [x for x in urls if x]

    return {
        "event_count": len(nearby),

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

        "unique_apps": len(set(apps)),
        "unique_windows": len(set(windows)),
        "unique_urls": len(set(urls)),

        "keyboard_activity": (
            event_types == "keystroke"
        ).sum(),
    }


def calculate_change_score(before, after):
    """
    Measure how much observable behavior changed.

    The score is intentionally interpretable.
    """

    score = 0.0

    # Strongest signal discovered in Baseline 1.
    score += (
        abs(
            after["app_switches"]
            - before["app_switches"]
        )
        * 0.75
    )

    # Navigation is also a strong boundary signal.
    score += (
        abs(
            after["browser_navigation"]
            - before["browser_navigation"]
        )
        * 0.75
    )

    # Supporting interaction signals.
    score += (
        abs(
            after["browser_clicks"]
            - before["browser_clicks"]
        )
        * 0.25
    )

    score += (
        abs(
            after["mouse_clicks"]
            - before["mouse_clicks"]
        )
        * 0.25
    )

    score += (
        abs(
            after["window_changes"]
            - before["window_changes"]
        )
        * 0.25
    )

    # Change in application diversity.
    score += (
        abs(
            after["unique_apps"]
            - before["unique_apps"]
        )
        * 0.50
    )

    # Change in browser/page diversity.
    score += (
        abs(
            after["unique_urls"]
            - before["unique_urls"]
        )
        * 0.50
    )

    # Change in keyboard activity.
    score += (
        abs(
            after["keyboard_activity"]
            - before["keyboard_activity"]
        )
        * 0.10
    )

    return score


def create_candidate_features(events):
    """
    Calculate a change score at every 2-second bin.

    Candidate points at session edges are skipped because
    both before and after context are required.
    """

    rows = []

    for session_id, group in events.groupby("session_id"):
        group = group.sort_values("timestamp").copy()

        first_time = group["timestamp"].iloc[0]
        last_time = group["timestamp"].iloc[-1]

        total_seconds = (
            last_time - first_time
        ).total_seconds()

        bin_count = int(
            total_seconds // BIN_SECONDS
        )

        for bin_number in range(
            1,
            bin_count,
        ):
            candidate_time = (
                first_time
                + pd.Timedelta(
                    seconds=bin_number * BIN_SECONDS
                )
            )

            before_start = (
                candidate_time
                - pd.Timedelta(
                    seconds=CONTEXT_SECONDS
                )
            )

            before_end = candidate_time

            after_start = candidate_time

            after_end = (
                candidate_time
                + pd.Timedelta(
                    seconds=CONTEXT_SECONDS
                )
            )

            if before_start < first_time:
                continue

            if after_end > last_time:
                continue

            before = window_features(
                group,
                before_start,
                before_end,
            )

            after = window_features(
                group,
                after_start,
                after_end,
            )

            score = calculate_change_score(
                before,
                after,
            )

            rows.append(
                {
                    "session_id": session_id,
                    "candidate_ts": candidate_time,
                    "boundary_score": score,

                    "before_event_count":
                        before["event_count"],
                    "after_event_count":
                        after["event_count"],

                    "before_app_switches":
                        before["app_switches"],
                    "after_app_switches":
                        after["app_switches"],

                    "before_navigation":
                        before["browser_navigation"],
                    "after_navigation":
                        after["browser_navigation"],

                    "before_browser_clicks":
                        before["browser_clicks"],
                    "after_browser_clicks":
                        after["browser_clicks"],

                    "before_mouse_clicks":
                        before["mouse_clicks"],
                    "after_mouse_clicks":
                        after["mouse_clicks"],

                    "before_window_changes":
                        before["window_changes"],
                    "after_window_changes":
                        after["window_changes"],

                    "before_unique_apps":
                        before["unique_apps"],
                    "after_unique_apps":
                        after["unique_apps"],

                    "before_unique_urls":
                        before["unique_urls"],
                    "after_unique_urls":
                        after["unique_urls"],

                    "before_keyboard":
                        before["keyboard_activity"],
                    "after_keyboard":
                        after["keyboard_activity"],
                }
            )

    return pd.DataFrame(rows)


def select_boundaries(features):
    """
    Select high-scoring candidate points and merge nearby candidates.
    """

    candidates = features[
        features["boundary_score"]
        >= BOUNDARY_THRESHOLD
    ].copy()

    predictions = []

    for session_id, group in candidates.groupby("session_id"):
        group = group.sort_values("candidate_ts")

        cluster = []
        previous_time = None

        for _, row in group.iterrows():

            current_time = row["candidate_ts"]

            if (
                previous_time is None
                or (
                    current_time - previous_time
                ).total_seconds()
                <= BIN_SECONDS
            ):
                cluster.append(row)

            else:
                cluster_df = pd.DataFrame(cluster)

                best = cluster_df.loc[
                    cluster_df["boundary_score"].idxmax()
                ]

                predictions.append(
                    {
                        "session_id": session_id,
                        "predicted_ts": best[
                            "candidate_ts"
                        ],
                        "score": best[
                            "boundary_score"
                        ],
                    }
                )

                cluster = [row]

            previous_time = current_time

        if cluster:
            cluster_df = pd.DataFrame(cluster)

            best = cluster_df.loc[
                cluster_df["boundary_score"].idxmax()
            ]

            predictions.append(
                {
                    "session_id": session_id,
                    "predicted_ts": best[
                        "candidate_ts"
                    ],
                    "score": best[
                        "boundary_score"
                    ],
                }
            )

    return pd.DataFrame(predictions)


def evaluate(predictions, gt, tolerance_seconds=5):
    """Evaluate predictions against unique GT timestamps."""

    gt = (
        gt[
            ["session_id", "ts"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    gt["ts"] = pd.to_datetime(
        gt["ts"],
        utc=True,
    )

    matched_gt = set()
    matched_predictions = 0
    prediction_rows = []

    for _, prediction in predictions.iterrows():

        session_id = prediction["session_id"]
        predicted_ts = prediction["predicted_ts"]

        session_gt = gt[
            gt["session_id"] == session_id
        ]

        best_index = None
        best_error = None

        for gt_index, gt_row in session_gt.iterrows():

            key = (session_id, gt_index)

            if key in matched_gt:
                continue

            error = abs(
                (
                    predicted_ts
                    - gt_row["ts"]
                ).total_seconds()
            )

            if error <= tolerance_seconds:

                if (
                    best_error is None
                    or error < best_error
                ):
                    best_error = error
                    best_index = gt_index

        if best_index is not None:

            matched_gt.add(
                (session_id, best_index)
            )

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

    precision = (
        matched_predictions / len(predictions)
        if len(predictions)
        else 0
    )

    recall = (
        matched_predictions / len(gt)
        if len(gt)
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    evaluation = pd.DataFrame(prediction_rows)

    metrics = {
        "gt_boundaries": len(gt),
        "predicted_boundaries": len(predictions),
        "matched_boundaries": matched_predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_boundary_error_seconds": (
            evaluation.loc[
                evaluation["matched"],
                "error_seconds",
            ].mean()
            if matched_predictions
            else None
        ),
    }

    return evaluation, metrics


def main():
    print("Loading Dataset A events...")
    events = load_events()

    print(
        f"Loaded {len(events):,} Dataset A events."
    )

    print("Building context-change features...")
    features = create_candidate_features(events)

    print(
        f"Built {len(features):,} candidate points."
    )

    features.to_csv(
        FEATURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Selecting candidate boundaries...")
    predictions = select_boundaries(features)

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Predicted boundaries: "
        f"{len(predictions):,}"
    )

    print("Loading ground truth...")
    gt = pd.read_csv(GT_FILE)

    print("Evaluating...")
    evaluation, metrics = evaluate(
        predictions,
        gt,
        tolerance_seconds=5,
    )

    evaluation.to_csv(
        EVALUATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Baseline V2 evaluation")
    print("----------------------")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()
    print(f"Saved features: {FEATURE_FILE}")
    print(f"Saved predictions: {PREDICTION_FILE}")
    print(f"Saved evaluation: {EVALUATION_FILE}")


if __name__ == "__main__":
    main()