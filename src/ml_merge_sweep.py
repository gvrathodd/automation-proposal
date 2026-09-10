from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROBABILITY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "ml_test_probabilities.csv"
)

GT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "gt_boundaries.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "ml_merge_sweep.csv"
)


# ============================================================
# Configuration
# ============================================================

CLASSIFICATION_THRESHOLD = 0.40

MERGE_GAPS = [0, 1, 2, 3, 4, 5]

TOLERANCES = [1, 2, 3, 5]


# ============================================================
# Load probabilities
# ============================================================

def load_probabilities():
    if not PROBABILITY_FILE.exists():
        raise FileNotFoundError(
            f"Probability file not found:\n{PROBABILITY_FILE}\n\n"
            "Run ml_boundary_postprocess.py first."
        )

    df = pd.read_csv(
        PROBABILITY_FILE
    )

    required_columns = {
        "session_id",
        "candidate_ts",
        "probability",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Probability file is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    df["candidate_ts"] = pd.to_datetime(
        df["candidate_ts"],
        utc=True,
        errors="coerce",
    )

    if df["candidate_ts"].isna().any():
        raise RuntimeError(
            "Invalid candidate_ts values found."
        )

    df["predicted_positive"] = (
        df["probability"]
        >= CLASSIFICATION_THRESHOLD
    )

    return df


# ============================================================
# Load GT
# ============================================================

def load_ground_truth():
    if not GT_FILE.exists():
        raise FileNotFoundError(
            f"GT file not found:\n{GT_FILE}"
        )

    gt = pd.read_csv(
        GT_FILE
    )

    required_columns = {
        "session_id",
        "ts",
    }

    missing = (
        required_columns
        - set(gt.columns)
    )

    if missing:
        raise RuntimeError(
            "GT file is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    gt["ts"] = pd.to_datetime(
        gt["ts"],
        utc=True,
        errors="coerce",
    )

    if gt["ts"].isna().any():
        raise RuntimeError(
            "Invalid GT timestamps found."
        )

    gt = (
        gt[
            [
                "session_id",
                "ts",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "session_id",
                "ts",
            ]
        )
        .reset_index(drop=True)
    )

    return gt


# ============================================================
# Merge positive candidates
# ============================================================

def merge_candidates(
    positive_candidates,
    merge_gap_seconds,
):
    """
    Group consecutive positive candidate timestamps into
    boundary episodes.

    Chain-based merging is used intentionally:
    if each consecutive pair is <= merge_gap_seconds apart,
    they belong to the same cluster.

    The highest-probability candidate becomes the predicted
    boundary timestamp for each cluster.
    """

    predictions = []

    if positive_candidates.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "predicted_ts",
                "probability",
                "cluster_size",
            ]
        )

    for session_id, group in (
        positive_candidates
        .groupby("session_id")
    ):

        group = (
            group
            .sort_values("candidate_ts")
            .reset_index(drop=True)
        )

        cluster = []
        previous_time = None

        for _, row in group.iterrows():

            current_time = row[
                "candidate_ts"
            ]

            if (
                previous_time is None
                or (
                    current_time
                    - previous_time
                ).total_seconds()
                <= merge_gap_seconds
            ):
                cluster.append(row)

            else:
                cluster_df = pd.DataFrame(
                    cluster
                )

                best = cluster_df.loc[
                    cluster_df[
                        "probability"
                    ].idxmax()
                ]

                predictions.append(
                    {
                        "session_id":
                            session_id,
                        "predicted_ts":
                            best[
                                "candidate_ts"
                            ],
                        "probability":
                            best[
                                "probability"
                            ],
                        "cluster_size":
                            len(cluster_df),
                    }
                )

                cluster = [row]

            previous_time = current_time

        if cluster:

            cluster_df = pd.DataFrame(
                cluster
            )

            best = cluster_df.loc[
                cluster_df[
                    "probability"
                ].idxmax()
            ]

            predictions.append(
                {
                    "session_id":
                        session_id,
                    "predicted_ts":
                        best[
                            "candidate_ts"
                        ],
                    "probability":
                        best[
                            "probability"
                        ],
                    "cluster_size":
                        len(cluster_df),
                }
            )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# Temporal evaluation
# ============================================================

def evaluate_at_tolerance(
    predictions,
    gt,
    tolerance_seconds,
):
    """
    One-to-one nearest temporal matching.

    Each prediction can match at most one GT boundary,
    and each GT boundary can match at most one prediction.
    """

    total_predictions = len(
        predictions
    )

    total_gt = len(
        gt
    )

    if total_predictions == 0:
        return {
            "matched_boundaries": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_error": np.nan,
            "median_error": np.nan,
        }

    predictions = (
        predictions
        .sort_values(
            [
                "session_id",
                "predicted_ts",
            ]
        )
        .reset_index(drop=True)
    )

    matched_gt = set()
    errors = []

    for _, prediction in predictions.iterrows():

        session_id = prediction[
            "session_id"
        ]

        predicted_ts = prediction[
            "predicted_ts"
        ]

        session_gt = gt[
            gt["session_id"]
            == session_id
        ]

        best_index = None
        best_error = None

        for gt_index, gt_row in (
            session_gt.iterrows()
        ):

            key = (
                session_id,
                gt_index,
            )

            if key in matched_gt:
                continue

            error = abs(
                (
                    predicted_ts
                    - gt_row["ts"]
                ).total_seconds()
            )

            if (
                error <= tolerance_seconds
                and (
                    best_error is None
                    or error < best_error
                )
            ):
                best_error = error
                best_index = gt_index

        if best_index is not None:

            matched_gt.add(
                (
                    session_id,
                    best_index,
                )
            )

            errors.append(
                best_error
            )

    matched = len(errors)

    precision = (
        matched / total_predictions
        if total_predictions
        else 0.0
    )

    recall = (
        matched / total_gt
        if total_gt
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "matched_boundaries":
            matched,
        "precision":
            precision,
        "recall":
            recall,
        "f1":
            f1,
        "mean_error":
            (
                float(np.mean(errors))
                if errors
                else np.nan
            ),
        "median_error":
            (
                float(np.median(errors))
                if errors
                else np.nan
            ),
    }


# ============================================================
# Session count statistics
# ============================================================

def build_count_statistics(
    predictions,
    gt,
    test_sessions,
):
    rows = []

    for session_id in sorted(
        test_sessions
    ):

        predicted_count = int(
            (
                predictions[
                    "session_id"
                ]
                == session_id
            ).sum()
        )

        gt_count = int(
            (
                gt[
                    "session_id"
                ]
                == session_id
            ).sum()
        )

        absolute_error = abs(
            predicted_count
            - gt_count
        )

        relative_error = (
            absolute_error / gt_count
            if gt_count
            else np.nan
        )

        rows.append(
            {
                "session_id":
                    session_id,
                "predicted_boundaries":
                    predicted_count,
                "gt_boundaries":
                    gt_count,
                "predicted_segments":
                    predicted_count + 1,
                "gt_segments":
                    gt_count + 1,
                "absolute_boundary_count_error":
                    absolute_error,
                "relative_boundary_count_error":
                    relative_error,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ML MERGE-WINDOW SWEEP")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load saved probabilities
    # --------------------------------------------------------

    probabilities = load_probabilities()

    positives = (
        probabilities[
            probabilities[
                "predicted_positive"
            ]
        ]
        .copy()
    )

    print()
    print(
        f"Test candidate rows: "
        f"{len(probabilities):,}"
    )

    print(
        f"Raw positive candidates: "
        f"{len(positives):,}"
    )

    print(
        f"Classification threshold: "
        f"{CLASSIFICATION_THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # 2. Load GT and restrict to test sessions
    # --------------------------------------------------------

    gt_all = load_ground_truth()

    test_sessions = set(
        probabilities[
            "session_id"
        ]
    )

    gt = gt_all[
        gt_all[
            "session_id"
        ].isin(test_sessions)
    ].copy()

    print(
        f"GT boundaries in test sessions: "
        f"{len(gt):,}"
    )

    # --------------------------------------------------------
    # 3. Run merge sweep
    # --------------------------------------------------------

    all_results = []

    print()
    print("=" * 70)
    print("RUNNING MERGE SWEEP")
    print("=" * 70)

    for merge_gap in MERGE_GAPS:

        print()
        print(
            f"Testing merge gap: "
            f"{merge_gap}s"
        )

        predictions = merge_candidates(
            positives,
            merge_gap,
        )

        count_stats = build_count_statistics(
            predictions,
            gt,
            test_sessions,
        )

        mean_count_error = float(
            count_stats[
                "absolute_boundary_count_error"
            ].mean()
        )

        median_count_error = float(
            count_stats[
                "absolute_boundary_count_error"
            ].median()
        )

        mean_relative_count_error = float(
            count_stats[
                "relative_boundary_count_error"
            ].mean()
        )

        result_base = {
            "merge_gap_seconds":
                merge_gap,
            "raw_positive_candidates":
                len(positives),
            "predicted_boundaries":
                len(predictions),
            "predicted_segments":
                len(predictions)
                + len(test_sessions),
            "gt_boundaries":
                len(gt),
            "gt_segments":
                len(gt)
                + len(test_sessions),
            "mean_boundary_count_error":
                mean_count_error,
            "median_boundary_count_error":
                median_count_error,
            "mean_relative_boundary_count_error":
                mean_relative_count_error,
        }

        for tolerance in TOLERANCES:

            metrics = evaluate_at_tolerance(
                predictions,
                gt,
                tolerance,
            )

            result_base[
                f"matched_{tolerance}s"
            ] = metrics[
                "matched_boundaries"
            ]

            result_base[
                f"precision_{tolerance}s"
            ] = metrics[
                "precision"
            ]

            result_base[
                f"recall_{tolerance}s"
            ] = metrics[
                "recall"
            ]

            result_base[
                f"f1_{tolerance}s"
            ] = metrics[
                "f1"
            ]

            result_base[
                f"mean_error_{tolerance}s"
            ] = metrics[
                "mean_error"
            ]

            result_base[
                f"median_error_{tolerance}s"
            ] = metrics[
                "median_error"
            ]

        all_results.append(
            result_base
        )

        print(
            f"  Predicted boundaries: "
            f"{len(predictions):,}"
        )

        print(
            f"  Mean count error/session: "
            f"{mean_count_error:.2f}"
        )

        print(
            f"  ±1s F1: "
            f"{result_base['f1_1s']:.4f}"
        )

        print(
            f"  ±2s F1: "
            f"{result_base['f1_2s']:.4f}"
        )

        print(
            f"  ±3s F1: "
            f"{result_base['f1_3s']:.4f}"
        )

        print(
            f"  ±5s F1: "
            f"{result_base['f1_5s']:.4f}"
        )

    # --------------------------------------------------------
    # 4. Build result table
    # --------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 5. Print compact comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MERGE-WINDOW COMPARISON")
    print("=" * 70)

    display_columns = [
        "merge_gap_seconds",
        "predicted_boundaries",
        "gt_boundaries",
        "mean_boundary_count_error",
        "mean_relative_boundary_count_error",
        "precision_1s",
        "recall_1s",
        "f1_1s",
        "precision_3s",
        "recall_3s",
        "f1_3s",
        "precision_5s",
        "recall_5s",
        "f1_5s",
        "mean_error_5s",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 6. Identify best windows
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BEST MERGE WINDOWS BY TEMPORAL F1")
    print("=" * 70)

    for tolerance in TOLERANCES:

        column = (
            f"f1_{tolerance}s"
        )

        best_row = results.loc[
            results[column].idxmax()
        ]

        print(
            f"±{tolerance}s: "
            f"{best_row['merge_gap_seconds']:.0f}s "
            f"merge → "
            f"F1={best_row[column]:.4f}, "
            f"precision="
            f"{best_row[f'precision_{tolerance}s']:.4f}, "
            f"recall="
            f"{best_row[f'recall_{tolerance}s']:.4f}, "
            f"predicted="
            f"{best_row['predicted_boundaries']:.0f}"
        )

    best_count_row = results.loc[
        results[
            "mean_boundary_count_error"
        ].idxmin()
    ]

    print()
    print(
        "Best window by mean session boundary-count error:"
    )

    print(
        f"  {best_count_row['merge_gap_seconds']:.0f}s "
        f"merge → "
        f"mean count error="
        f"{best_count_row['mean_boundary_count_error']:.2f}, "
        f"predicted boundaries="
        f"{best_count_row['predicted_boundaries']:.0f}"
    )

    # --------------------------------------------------------
    # 7. Final output
    # --------------------------------------------------------

    print()
    print(
        f"Saved sweep results:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()