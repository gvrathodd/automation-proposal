from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

PROB_FILE = OUTPUTS / "segment_model_validation_probabilities.csv"
SPLIT_FILE = OUTPUTS / "segment_model_split.csv"
GT_FILE = OUTPUTS / "gt_boundaries.csv"

RESULT_FILE = OUTPUTS / "local_peak_validation_results.csv"


# We only use the validation sessions for model-selection.
# Test data remains untouched.
THRESHOLDS = np.arange(0.20, 0.81, 0.05)

# Minimum time separation between selected boundaries.
MIN_DISTANCES = [2.0, 3.0, 4.0, 5.0, 6.0]

MATCH_TOLERANCE = 5.0


def load_data():
    probs = pd.read_csv(PROB_FILE)
    split = pd.read_csv(SPLIT_FILE)
    gt = pd.read_csv(GT_FILE)

    probs["candidate_ts"] = pd.to_datetime(
        probs["candidate_ts"], utc=True
    )
    gt["ts"] = pd.to_datetime(
        gt["ts"], utc=True
    )

    validation_sessions = set(
        split.loc[split["split"] == "validation", "session_id"]
    )

    probs = probs[
        probs["session_id"].isin(validation_sessions)
    ].copy()

    gt = gt[
        gt["session_id"].isin(validation_sessions)
    ].copy()

    return probs, gt


def select_local_peaks(
    session_df: pd.DataFrame,
    threshold: float,
    min_distance: float,
) -> list[pd.Timestamp]:
    """
    Select probability local maxima.

    A candidate is eligible if:
      1. probability >= threshold
      2. it is a local maximum compared with adjacent candidates

    Then enforce minimum temporal spacing greedily,
    keeping the highest-probability candidate in each neighborhood.
    """
    df = session_df.sort_values("candidate_ts").reset_index(drop=True)

    if df.empty:
        return []

    probabilities = df["probability"].to_numpy(dtype=float)
    timestamps = df["candidate_ts"].tolist()

    # Local maxima.
    peak_indices = []

    for i in range(len(df)):
        left = probabilities[i - 1] if i > 0 else -np.inf
        right = probabilities[i + 1] if i < len(df) - 1 else -np.inf

        if probabilities[i] >= threshold and \
           probabilities[i] >= left and \
           probabilities[i] >= right:
            peak_indices.append(i)

    if not peak_indices:
        return []

    # Highest-probability peaks first.
    peak_indices.sort(
        key=lambda i: probabilities[i],
        reverse=True,
    )

    selected = []

    for idx in peak_indices:
        ts = timestamps[idx]

        too_close = False

        for selected_ts in selected:
            distance = abs(
                (ts - selected_ts).total_seconds()
            )

            if distance < min_distance:
                too_close = True
                break

        if not too_close:
            selected.append(ts)

    return sorted(selected)


def evaluate(
    predictions: dict[str, list[pd.Timestamp]],
    gt: pd.DataFrame,
):
    tp = 0
    fp = 0
    fn = 0
    errors = []

    for session_id, pred_times in predictions.items():

        pred_times = sorted(pred_times)

        gt_times = sorted(
            gt.loc[
                gt["session_id"] == session_id,
                "ts"
            ].tolist()
        )

        matched_gt = set()

        for pred in pred_times:
            candidates = [
                (
                    abs((pred - actual).total_seconds()),
                    j,
                )
                for j, actual in enumerate(gt_times)
                if j not in matched_gt
            ]

            if not candidates:
                fp += 1
                continue

            error, j = min(candidates)

            if error <= MATCH_TOLERANCE:
                tp += 1
                matched_gt.add(j)
                errors.append(error)
            else:
                fp += 1

        fn += len(gt_times) - len(matched_gt)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    mean_error = (
        float(np.mean(errors))
        if errors
        else np.nan
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error_seconds": mean_error,
        "predicted_boundaries": tp + fp,
        "gt_boundaries": tp + fn,
    }


def main():
    probs, gt = load_data()

    print(f"Validation probability rows: {len(probs):,}")
    print(f"Validation GT boundaries: {len(gt):,}")
    print(
        f"Validation sessions: "
        f"{probs['session_id'].nunique()}"
    )

    results = []

    grouped = dict(tuple(probs.groupby("session_id")))

    for threshold in THRESHOLDS:

        for min_distance in MIN_DISTANCES:

            predictions = {}

            for session_id, session_df in grouped.items():

                predictions[session_id] = select_local_peaks(
                    session_df,
                    threshold=threshold,
                    min_distance=min_distance,
                )

            metrics = evaluate(predictions, gt)

            results.append({
                "threshold": round(float(threshold), 2),
                "min_distance_seconds": min_distance,
                **metrics,
            })

            print(
                f"threshold={threshold:.2f} "
                f"distance={min_distance:.1f}s "
                f"F1={metrics['f1']:.4f} "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"pred={metrics['predicted_boundaries']}"
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["f1", "precision"],
        ascending=[False, False],
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    print("\nBest validation configuration:")
    print(results_df.iloc[0].to_string())

    print(f"\nSaved: {RESULT_FILE}")


if __name__ == "__main__":
    main()