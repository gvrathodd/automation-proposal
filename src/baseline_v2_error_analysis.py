from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"

GT_FILE = PROJECT_ROOT / "outputs" / "gt_boundaries.csv"
PRED_FILE = PROJECT_ROOT / "outputs" / "baseline_v2_boundaries.csv"

CASES_FILE = PROJECT_ROOT / "outputs" / "baseline_v2_error_cases.csv"
EVENTS_FILE = PROJECT_ROOT / "outputs" / "baseline_v2_error_event_context.csv"

TOLERANCE_SECONDS = 5
EVENT_WINDOW_SECONDS = 5

# Number of examples to inspect for each category.
N_EXAMPLES = 10


def load_events():
    """Load Dataset A events."""
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


def match_predictions_to_gt(predictions, gt):
    """
    Match each prediction to at most one GT boundary.

    Matching criterion:
        same session + timestamp within TOLERANCE_SECONDS.
    """

    matched_gt = set()
    matches = []

    for pred_index, prediction in predictions.iterrows():
        session_id = prediction["session_id"]
        predicted_ts = prediction["predicted_ts"]

        session_gt = gt[
            gt["session_id"] == session_id
        ]

        best_gt_index = None
        best_error = None

        for gt_index, gt_row in session_gt.iterrows():
            if (session_id, gt_index) in matched_gt:
                continue

            gt_ts = gt_row["ts"]

            error = abs(
                (predicted_ts - gt_ts).total_seconds()
            )

            if error <= TOLERANCE_SECONDS:
                if best_error is None or error < best_error:
                    best_error = error
                    best_gt_index = gt_index

        if best_gt_index is not None:
            matched_gt.add(
                (session_id, best_gt_index)
            )

            matches.append(
                {
                    "prediction_index": pred_index,
                    "gt_index": best_gt_index,
                    "error_seconds": best_error,
                }
            )

    return matches, matched_gt


def build_case_table(predictions, gt):
    """
    Create TP / FP / FN case table.
    """

    matches, matched_gt = match_predictions_to_gt(
        predictions,
        gt,
    )

    matched_prediction_indices = {
        item["prediction_index"]
        for item in matches
    }

    rows = []

    # True positives and false positives.
    for pred_index, prediction in predictions.iterrows():

        match = next(
            (
                item
                for item in matches
                if item["prediction_index"] == pred_index
            ),
            None,
        )

        if match is not None:
            gt_row = gt.loc[match["gt_index"]]

            rows.append(
                {
                    "case_type": "TP",
                    "session_id": prediction["session_id"],
                    "predicted_ts": prediction["predicted_ts"],
                    "gt_ts": gt_row["ts"],
                    "error_seconds": match["error_seconds"],
                    "score": prediction["score"],
                    "gt_index": match["gt_index"],
                }
            )

        else:
            rows.append(
                {
                    "case_type": "FP",
                    "session_id": prediction["session_id"],
                    "predicted_ts": prediction["predicted_ts"],
                    "gt_ts": None,
                    "error_seconds": None,
                    "score": prediction["score"],
                    "gt_index": None,
                }
            )

    # False negatives.
    for gt_index, gt_row in gt.iterrows():

        if (
            gt_row["session_id"],
            gt_index,
        ) in matched_gt:
            continue

        rows.append(
            {
                "case_type": "FN",
                "session_id": gt_row["session_id"],
                "predicted_ts": None,
                "gt_ts": gt_row["ts"],
                "error_seconds": None,
                "score": None,
                "gt_index": gt_index,
            }
        )

    return pd.DataFrame(rows)


def choose_examples(cases):
    """
    Pick examples from each class.

    For false positives, choose strongest-scoring examples first.

    For false negatives, choose examples spread across sessions
    rather than taking many from one session.
    """

    examples = []

    # True positives:
    tp = cases[cases["case_type"] == "TP"].copy()

    tp = tp.sort_values(
        "error_seconds",
        ascending=False,
    )

    examples.append(tp.head(N_EXAMPLES))

    # False positives:
    fp = cases[cases["case_type"] == "FP"].copy()

    fp = fp.sort_values(
        "score",
        ascending=False,
    )

    examples.append(fp.head(N_EXAMPLES))

    # False negatives:
    fn = cases[cases["case_type"] == "FN"].copy()

    # First get one example per session.
    fn_spread = (
        fn.sort_values("gt_ts")
        .drop_duplicates(
            subset=["session_id"]
        )
        .head(N_EXAMPLES)
    )

    if len(fn_spread) < N_EXAMPLES:
        remaining = fn[
            ~fn.index.isin(fn_spread.index)
        ]

        fn_spread = pd.concat(
            [
                fn_spread,
                remaining.head(
                    N_EXAMPLES - len(fn_spread)
                ),
            ]
        )

    examples.append(fn_spread)

    result = pd.concat(
        examples,
        ignore_index=True,
    )

    result["case_id"] = range(1, len(result) + 1)

    return result


def collect_event_context(events, examples):
    """
    For each selected error-analysis case, show the raw
    computer activity around the relevant timestamp.
    """

    rows = []

    for _, case in examples.iterrows():

        if pd.notna(case["predicted_ts"]):
            anchor_time = case["predicted_ts"]
        else:
            anchor_time = case["gt_ts"]

        session_id = case["session_id"]

        start = (
            anchor_time
            - pd.Timedelta(seconds=EVENT_WINDOW_SECONDS)
        )

        end = (
            anchor_time
            + pd.Timedelta(seconds=EVENT_WINDOW_SECONDS)
        )

        nearby = events[
            (events["session_id"] == session_id)
            & (events["timestamp"] >= start)
            & (events["timestamp"] <= end)
        ].copy()

        for _, event_row in nearby.iterrows():

            event = event_row["event"]

            rows.append(
                {
                    "case_id": case["case_id"],
                    "case_type": case["case_type"],
                    "session_id": session_id,
                    "anchor_ts": anchor_time,
                    "offset_seconds": (
                        event_row["timestamp"]
                        - anchor_time
                    ).total_seconds(),
                    "event_ts": event_row["timestamp"],
                    "event_type": event_row["event_type"],
                    "active_app": get_active_app(event),
                    "window_title": get_window_title(event),
                    "browser_url": get_browser_url(event),
                }
            )

    return pd.DataFrame(rows)


def main():
    print("Loading events...")
    events = load_events()

    print(f"Loaded {len(events):,} Dataset A events.")

    print("Loading ground truth...")
    gt = pd.read_csv(GT_FILE)

    gt["ts"] = pd.to_datetime(
        gt["ts"],
        utc=True,
    )

    # Match the same GT preprocessing used by the baseline evaluator.
    gt = (
        gt[
            ["session_id", "ts"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"GT timestamps: {len(gt):,}")

    print("Loading baseline predictions...")
    predictions = pd.read_csv(PRED_FILE)

    predictions["predicted_ts"] = pd.to_datetime(
        predictions["predicted_ts"],
        utc=True,
    )

    print(
        f"Predictions: "
        f"{len(predictions):,}"
    )

    print("Building TP / FP / FN cases...")

    cases = build_case_table(
        predictions,
        gt,
    )

    print()
    print("Case counts:")
    print(
        cases["case_type"]
        .value_counts()
        .to_string()
    )

    examples = choose_examples(cases)

    print()
    print("Selected examples:")
    print(
        examples[
            [
                "case_id",
                "case_type",
                "session_id",
                "predicted_ts",
                "gt_ts",
                "error_seconds",
                "score",
            ]
        ].to_string(index=False)
    )

    examples.to_csv(
        CASES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    event_context = collect_event_context(
        events,
        examples,
    )

    event_context.to_csv(
        EVENTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Saved cases: {CASES_FILE}")
    print(f"Saved event context: {EVENTS_FILE}")


if __name__ == "__main__":
    main()