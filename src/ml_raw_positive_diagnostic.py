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
    / "ml_raw_positive_diagnostic.csv"
)


# ============================================================
# Configuration
# ============================================================

THRESHOLD = 0.40

# Candidate grid is spaced at 2 seconds.
CANDIDATE_SPACING_SECONDS = 2

# Tolerance used to determine whether a positive candidate
# is close enough to a GT boundary to be considered detected.
GT_MATCH_TOLERANCE_SECONDS = 5


# ============================================================
# Load test probabilities
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
        "predicted_positive",
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

    return df


# ============================================================
# Load GT boundaries
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

    # Normalize exactly as used elsewhere in the project.
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
# Build zero-merge clusters
# ============================================================

def build_zero_merge_clusters(
    positive_candidates
):
    """
    Because candidates are normally spaced at 2 seconds,
    this function groups only candidates that occupy the
    exact same timestamp.

    In normal data this therefore behaves like "no merging".
    """

    clusters = []

    for session_id, group in (
        positive_candidates
        .groupby("session_id")
    ):

        group = group.sort_values(
            "candidate_ts"
        )

        current_cluster = []
        previous_ts = None

        for _, row in group.iterrows():

            current_ts = row[
                "candidate_ts"
            ]

            if (
                previous_ts is None
                or current_ts != previous_ts
            ):
                if current_cluster:
                    clusters.append(
                        current_cluster
                    )

                current_cluster = [row]

            else:
                current_cluster.append(row)

            previous_ts = current_ts

        if current_cluster:
            clusters.append(
                current_cluster
            )

    rows = []

    for cluster in clusters:
        cluster_df = pd.DataFrame(
            cluster
        )

        best = cluster_df.loc[
            cluster_df[
                "probability"
            ].idxmax()
        ]

        rows.append(
            {
                "session_id":
                    best["session_id"],
                "predicted_ts":
                    best["candidate_ts"],
                "probability":
                    best["probability"],
                "cluster_size":
                    len(cluster_df),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Match raw positive candidates to GT
# ============================================================

def match_candidates_to_gt(
    positives,
    gt,
    tolerance_seconds,
):
    """
    Determine how many unique GT boundaries are within
    tolerance of at least one raw positive candidate.

    This is a diagnostic, not the final one-to-one
    segmentation metric.
    """

    gt_detected = set()

    positive_gt_links = []

    for _, candidate in positives.iterrows():

        session_id = candidate[
            "session_id"
        ]

        candidate_ts = candidate[
            "candidate_ts"
        ]

        session_gt = gt[
            gt["session_id"]
            == session_id
        ]

        best_gt_index = None
        best_error = None

        for gt_index, gt_row in (
            session_gt.iterrows()
        ):

            error = abs(
                (
                    candidate_ts
                    - gt_row["ts"]
                ).total_seconds()
            )

            if error <= tolerance_seconds:

                if (
                    best_error is None
                    or error < best_error
                ):
                    best_error = error
                    best_gt_index = gt_index

        if best_gt_index is not None:

            gt_detected.add(
                (
                    session_id,
                    best_gt_index,
                )
            )

            positive_gt_links.append(
                {
                    "session_id":
                        session_id,
                    "candidate_ts":
                        candidate_ts,
                    "gt_index":
                        best_gt_index,
                    "error_seconds":
                        best_error,
                }
            )

    return (
        len(gt_detected),
        pd.DataFrame(
            positive_gt_links
        )
    )


# ============================================================
# Session statistics
# ============================================================

def build_session_statistics(
    positives,
    zero_merge_clusters,
    gt,
):
    sessions = sorted(
        set(
            positives[
                "session_id"
            ]
        )
    )

    rows = []

    for session_id in sessions:

        positive_count = int(
            (
                positives["session_id"]
                == session_id
            ).sum()
        )

        zero_merge_count = int(
            (
                zero_merge_clusters[
                    "session_id"
                ]
                == session_id
            ).sum()
        )

        gt_count = int(
            (
                gt["session_id"]
                == session_id
            ).sum()
        )

        rows.append(
            {
                "session_id":
                    session_id,
                "raw_positive_candidates":
                    positive_count,
                "zero_merge_boundaries":
                    zero_merge_count,
                "gt_boundaries":
                    gt_count,
                "raw_positive_to_gt_ratio":
                    (
                        positive_count / gt_count
                        if gt_count
                        else np.nan
                    ),
                "zero_merge_to_gt_ratio":
                    (
                        zero_merge_count / gt_count
                        if gt_count
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ML RAW POSITIVE DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load probabilities
    # --------------------------------------------------------

    probabilities = load_probabilities()

    print()
    print(
        f"Probability rows: "
        f"{len(probabilities):,}"
    )

    print(
        f"Sessions: "
        f"{probabilities['session_id'].nunique():,}"
    )

    # Recalculate positive candidates from probabilities
    # rather than relying only on the saved boolean column.
    probabilities[
        "predicted_positive"
    ] = (
        probabilities[
            "probability"
        ]
        >= THRESHOLD
    )

    positives = probabilities[
        probabilities[
            "predicted_positive"
        ]
    ].copy()

    print(
        f"Threshold: "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Raw positive candidates: "
        f"{len(positives):,}"
    )

    # --------------------------------------------------------
    # 2. Zero-merge diagnostic
    # --------------------------------------------------------

    zero_merge = build_zero_merge_clusters(
        positives
    )

    print()
    print("-" * 70)
    print("ZERO-MERGE DIAGNOSTIC")
    print("-" * 70)

    print(
        f"Raw positive candidates: "
        f"{len(positives):,}"
    )

    print(
        f"Zero-merge boundaries: "
        f"{len(zero_merge):,}"
    )

    if len(zero_merge) > 0:

        print(
            f"Average positive candidates "
            f"per zero-merge boundary: "
            f"{len(positives) / len(zero_merge):.3f}"
        )

    # --------------------------------------------------------
    # 3. Load GT
    # --------------------------------------------------------

    gt_all = load_ground_truth()

    test_sessions = set(
        probabilities[
            "session_id"
        ]
    )

    gt = gt_all[
        gt_all["session_id"].isin(
            test_sessions
        )
    ].copy()

    print()
    print(
        f"GT boundaries in test sessions: "
        f"{len(gt):,}"
    )

    # --------------------------------------------------------
    # 4. Raw positive coverage of GT
    # --------------------------------------------------------

    detected_gt_count, links = (
        match_candidates_to_gt(
            positives,
            gt,
            GT_MATCH_TOLERANCE_SECONDS,
        )
    )

    coverage = (
        detected_gt_count / len(gt)
        if len(gt)
        else 0.0
    )

    print()
    print("-" * 70)
    print("RAW POSITIVE GT COVERAGE")
    print("-" * 70)

    print(
        f"GT matching tolerance: "
        f"±{GT_MATCH_TOLERANCE_SECONDS}s"
    )

    print(
        f"GT boundaries: "
        f"{len(gt):,}"
    )

    print(
        f"GT boundaries with at least one "
        f"raw positive nearby: "
        f"{detected_gt_count:,}"
    )

    print(
        f"Raw positive GT coverage: "
        f"{coverage:.4f}"
    )

    print(
        f"Raw positive GT coverage (%): "
        f"{coverage * 100:.2f}%"
    )

    print(
        f"GT boundaries with no raw positive nearby: "
        f"{len(gt) - detected_gt_count:,}"
    )

    # --------------------------------------------------------
    # 5. Session-level statistics
    # --------------------------------------------------------

    stats = build_session_statistics(
        positives,
        zero_merge,
        gt,
    )

    print()
    print("-" * 70)
    print("SESSION-LEVEL STATISTICS")
    print("-" * 70)

    print(
        stats.to_string(
            index=False
        )
    )

    print()
    print("Summary:")

    print(
        stats[
            [
                "raw_positive_candidates",
                "zero_merge_boundaries",
                "gt_boundaries",
                "raw_positive_to_gt_ratio",
                "zero_merge_to_gt_ratio",
            ]
        ]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # 6. Check which GT boundaries are missed
    # --------------------------------------------------------

    detected_keys = set()

    if not links.empty:
        detected_keys = set(
            zip(
                links["session_id"],
                links["gt_index"],
            )
        )

    missed_rows = []

    for gt_index, row in gt.iterrows():

        key = (
            row["session_id"],
            gt_index,
        )

        if key not in detected_keys:

            missed_rows.append(
                {
                    "session_id":
                        row["session_id"],
                    "gt_ts":
                        row["ts"],
                }
            )

    missed_gt = pd.DataFrame(
        missed_rows
    )

    print()
    print("-" * 70)
    print("MISSED GT BOUNDARIES")
    print("-" * 70)

    print(
        f"Missed GT boundaries: "
        f"{len(missed_gt):,}"
    )

    if not missed_gt.empty:

        print()
        print(
            missed_gt.head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 7. Save output
    # --------------------------------------------------------

    stats.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Saved session diagnostic:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()