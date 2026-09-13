
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 2 — SEGMENT DURATION / MERGE-RISK DIAGNOSTIC
# ============================================================
#
# Purpose:
#   Quantify whether the 540 Step-1-generated Dataset B segments
#   contain suspiciously long segments that may have absorbed
#   multiple real work units because a boundary was missed.
#
# IMPORTANT:
#   This is NOT a new segmentation/model experiment.
#   It only diagnoses the existing Step-1 output.
#
# Inputs:
#   1. dataset_b_segments.csv
#   2. step2_dataset_b_segment_behavior.csv
#
# Outputs:
#   1. step2_segment_duration_diagnostics.csv
#   2. step2_duration_outlier_review.csv
#
# The outlier queue is a REVIEW QUEUE, not an automatic claim
# that a segment is incorrectly merged.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = ROOT / "outputs"

SEGMENTS_FILE = (
    OUTPUTS / "dataset_b_segments.csv"
)

BEHAVIOR_FILE = (
    OUTPUTS / "step2_dataset_b_segment_behavior.csv"
)

DIAGNOSTICS_FILE = (
    OUTPUTS / "step2_segment_duration_diagnostics.csv"
)

REVIEW_FILE = (
    OUTPUTS / "step2_duration_outlier_review.csv"
)


# ============================================================
# Configuration
# ============================================================

# We use robust / percentile-based rules instead of an arbitrary
# fixed duration threshold.
FAMILY_Q95 = 0.95
FAMILY_Q99 = 0.99

# A segment that is >= 2x its family's median is suspicious
# enough to enter the review ranking.
MEDIAN_RATIO_REVIEW = 2.0

# A segment >= family 95th percentile receives a duration flag.
# We do NOT call it merged solely from this.
USE_FAMILY_Q95 = True

# More complexity signals raise review priority:
# multiple apps/transitions/pages/forms/clipboard interactions.
COMPLEXITY_APP_THRESHOLD = 3
COMPLEXITY_TRANSITION_THRESHOLD = 2
COMPLEXITY_PAGE_THRESHOLD = 2
COMPLEXITY_EVENT_THRESHOLD = 75


# ============================================================
# Helpers
# ============================================================

def to_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )
    return df


def safe_mode(series):
    values = series.dropna().astype(str)
    values = values[values.str.len() > 0]

    if values.empty:
        return "UNKNOWN"

    modes = values.mode()

    return (
        modes.iloc[0]
        if not modes.empty
        else "UNKNOWN"
    )


def count_items(value, separator=" | "):
    if not isinstance(value, str):
        return 0

    value = value.strip()

    if not value:
        return 0

    return len(
        [
            x
            for x in value.split(separator)
            if x.strip()
        ]
    )


def print_percentiles(series, label):
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty:
        print(f"{label}: no valid values")
        return

    print()
    print(label)

    print(
        f"  count:   {len(clean):,}"
    )

    print(
        f"  mean:    {clean.mean():.2f}s"
    )

    print(
        f"  median:  {clean.median():.2f}s"
    )

    print(
        f"  min:     {clean.min():.2f}s"
    )

    for p in [0.75, 0.90, 0.95, 0.99]:
        print(
            f"  p{int(p * 100)}:     "
            f"{clean.quantile(p):.2f}s"
        )

    print(
        f"  max:     {clean.max():.2f}s"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "STEP 2 — SEGMENT DURATION / MERGE-RISK DIAGNOSTIC"
    )
    print("=" * 70)

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Step-1 segment file:\n"
            f"{SEGMENTS_FILE}"
        )

    if not BEHAVIOR_FILE.exists():
        raise FileNotFoundError(
            f"Missing Step-2 behavior file:\n"
            f"{BEHAVIOR_FILE}"
        )

    # --------------------------------------------------------
    # Load Step-1 segments.
    # --------------------------------------------------------

    print()
    print(
        "Loading Step-1 Dataset B segments..."
    )

    segments = pd.read_csv(
        SEGMENTS_FILE,
        keep_default_na=False,
    )

    required_segment_columns = {
        "session_id",
        "segment_index",
        "start",
        "end",
        "duration_seconds",
    }

    missing = (
        required_segment_columns
        - set(segments.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing columns in dataset_b_segments.csv:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    segments["segment_index"] = pd.to_numeric(
        segments["segment_index"],
        errors="raise",
    ).astype(int)

    segments["start"] = pd.to_datetime(
        segments["start"],
        utc=True,
    )

    segments["end"] = pd.to_datetime(
        segments["end"],
        utc=True,
    )

    segments = segments.rename(
        columns={
            "duration_seconds":
                "segment_duration_seconds"
        }
    )

    # --------------------------------------------------------
    # Load Step-2 behavioral representation.
    # --------------------------------------------------------

    print(
        "Loading Step-2 segment behavior..."
    )

    behavior = pd.read_csv(
        BEHAVIOR_FILE,
        keep_default_na=False,
    )

    required_behavior_columns = {
        "session_id",
        "segment_index",
    }

    missing = (
        required_behavior_columns
        - set(behavior.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing columns in step2_dataset_b_segment_behavior.csv:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    behavior["segment_index"] = pd.to_numeric(
        behavior["segment_index"],
        errors="raise",
    ).astype(int)

    # --------------------------------------------------------
    # Merge the existing Step-1 and Step-2 tables.
    # --------------------------------------------------------

    key_columns = [
        "session_id",
        "segment_index",
    ]

    behavior_payload_columns = [
        column
        for column in behavior.columns
        if column not in key_columns
    ]

    # Prevent duplicate copies of fields already present.
    merged = segments.merge(
        behavior[
            key_columns
            + behavior_payload_columns
        ],
        on=key_columns,
        how="left",
        suffixes=(
            "",
            "_behavior",
        ),
    )

    # --------------------------------------------------------
    # Normalize numeric behavior fields.
    # --------------------------------------------------------

    numeric_columns = [
        "segment_duration_seconds",
        "duration_seconds",
        "event_count",
        "unique_event_type_count",
        "unique_app_count",
        "important_transition_count",
        "clipboard_change_count",
        "keyboard_count",
        "browser_form_input_count",
        "browser_click_count",
        "mouse_click_count",
        "browser_navigation_count",
        "unique_browser_path_count",
        "extracted_text_event_count",
        "screenshot_count",
    ]

    merged = to_numeric(
        merged,
        numeric_columns,
    )

    # --------------------------------------------------------
    # Ensure we have one authoritative duration column.
    # --------------------------------------------------------

    if (
        "segment_duration_seconds"
        not in merged.columns
    ):

        if "duration_seconds" in merged.columns:
            merged["segment_duration_seconds"] = (
                merged["duration_seconds"]
            )
        else:
            merged["segment_duration_seconds"] = (
                merged["end"]
                - merged["start"]
            ).dt.total_seconds()

    # --------------------------------------------------------
    # Family assignment.
    # --------------------------------------------------------

    if "family_id" not in merged.columns:

        # The Step-2 behavior file normally contains family_id.
        # If not, use a neutral placeholder rather than inventing
        # a process family.
        merged["family_id"] = (
            "UNASSIGNED"
        )

    merged["family_id"] = (
        merged["family_id"]
        .fillna("UNASSIGNED")
        .astype(str)
    )

    # --------------------------------------------------------
    # Duration distribution — ALL 540 segments.
    # --------------------------------------------------------

    print()
    print(
        "Dataset-wide duration distribution:"
    )

    print_percentiles(
        merged[
            "segment_duration_seconds"
        ],
        "ALL STEP-1 SEGMENTS",
    )

    print()
    print(
        "Duration bucket counts:"
    )

    duration_buckets = pd.cut(
        merged[
            "segment_duration_seconds"
        ],
        bins=[
            -np.inf,
            10,
            20,
            40,
            60,
            120,
            300,
            np.inf,
        ],
        labels=[
            "<10s",
            "10-20s",
            "20-40s",
            "40-60s",
            "1-2m",
            "2-5m",
            "5m+",
        ],
        right=False,
    )

    print(
        duration_buckets
        .value_counts(
            sort=False,
        )
        .to_string()
    )

    # --------------------------------------------------------
    # Family-level baselines.
    # --------------------------------------------------------

    family_stats = (
        merged
        .groupby(
            "family_id",
            dropna=False,
        )[
            "segment_duration_seconds"
        ]
        .agg(
            family_segment_count="count",
            family_mean_duration="mean",
            family_median_duration="median",
            family_std_duration="std",
            family_min_duration="min",
            family_max_duration="max",
            family_q95_duration=lambda x:
                x.quantile(
                    FAMILY_Q95
                ),
            family_q99_duration=lambda x:
                x.quantile(
                    FAMILY_Q99
                ),
        )
        .reset_index()
    )

    merged = merged.merge(
        family_stats,
        on="family_id",
        how="left",
    )

    # --------------------------------------------------------
    # Duration outlier measures.
    # --------------------------------------------------------

    merged[
        "duration_to_family_median"
    ] = (
        merged[
            "segment_duration_seconds"
        ]
        / merged[
            "family_median_duration"
        ].replace(
            0,
            np.nan,
        )
    )

    merged[
        "duration_above_family_q95_seconds"
    ] = (
        merged[
            "segment_duration_seconds"
        ]
        - merged[
            "family_q95_duration"
        ]
    )

    merged[
        "duration_above_family_q99_seconds"
    ] = (
        merged[
            "segment_duration_seconds"
        ]
        - merged[
            "family_q99_duration"
        ]
    )

    merged[
        "duration_q95_flag"
    ] = (
        merged[
            "segment_duration_seconds"
        ]
        >= merged[
            "family_q95_duration"
        ]
    )

    merged[
        "duration_q99_flag"
    ] = (
        merged[
            "segment_duration_seconds"
        ]
        >= merged[
            "family_q99_duration"
        ]
    )

    merged[
        "duration_2x_median_flag"
    ] = (
        merged[
            "duration_to_family_median"
        ]
        >= MEDIAN_RATIO_REVIEW
    )

    # --------------------------------------------------------
    # Complexity signals.
    # --------------------------------------------------------

    merged[
        "computed_app_count"
    ] = (
        merged
        .get(
            "unique_app_count",
            pd.Series(
                0,
                index=merged.index,
            ),
        )
        .fillna(0)
    )

    merged[
        "computed_transition_count"
    ] = (
        merged
        .get(
            "important_transition_count",
            pd.Series(
                0,
                index=merged.index,
            ),
        )
        .fillna(0)
    )

    merged[
        "computed_page_count"
    ] = (
        merged
        .get(
            "unique_browser_path_count",
            pd.Series(
                0,
                index=merged.index,
            ),
        )
        .fillna(0)
    )

    merged[
        "computed_event_count"
    ] = (
        merged
        .get(
            "event_count",
            pd.Series(
                0,
                index=merged.index,
            ),
        )
        .fillna(0)
    )

    merged[
        "complexity_flag"
    ] = (
        (
            merged[
                "computed_app_count"
            ]
            >= COMPLEXITY_APP_THRESHOLD
        )
        |
        (
            merged[
                "computed_transition_count"
            ]
            >= COMPLEXITY_TRANSITION_THRESHOLD
        )
        |
        (
            merged[
                "computed_page_count"
            ]
            >= COMPLEXITY_PAGE_THRESHOLD
        )
        |
        (
            merged[
                "computed_event_count"
            ]
            >= COMPLEXITY_EVENT_THRESHOLD
        )
    )

    # --------------------------------------------------------
    # Additional workflow signals.
    # --------------------------------------------------------

    for column in [
        "clipboard_change_count",
        "browser_form_input_count",
        "browser_navigation_count",
        "browser_click_count",
        "mouse_click_count",
        "keyboard_count",
    ]:

        if column not in merged.columns:
            merged[column] = 0

        merged[column] = (
            pd.to_numeric(
                merged[column],
                errors="coerce",
            )
            .fillna(0)
        )

    merged[
        "interaction_density_per_10s"
    ] = (
        (
            merged[
                "clipboard_change_count"
            ]
            + merged[
                "browser_form_input_count"
            ]
            + merged[
                "browser_navigation_count"
            ]
            + merged[
                "browser_click_count"
            ]
            + merged[
                "mouse_click_count"
            ]
            + merged[
                "keyboard_count"
            ]
        )
        / merged[
            "segment_duration_seconds"
        ].replace(
            0,
            np.nan,
        )
        * 10
    )

    # --------------------------------------------------------
    # Review priority.
    #
    # This is ONLY a review-ranking score.
    # --------------------------------------------------------

    merged["merge_review_score"] = 0.0

    # Duration evidence.
    merged.loc[
        merged[
            "duration_q95_flag"
        ],
        "merge_review_score",
    ] += 2.0

    merged.loc[
        merged[
            "duration_q99_flag"
        ],
        "merge_review_score",
    ] += 2.0

    merged.loc[
        merged[
            "duration_2x_median_flag"
        ],
        "merge_review_score",
    ] += 2.0

    # Complexity evidence.
    merged.loc[
        merged[
            "complexity_flag"
        ],
        "merge_review_score",
    ] += 2.0

    # Multiple page/application behavior.
    merged.loc[
        merged[
            "computed_app_count"
        ] >= 4,
        "merge_review_score",
    ] += 1.0

    merged.loc[
        merged[
            "computed_page_count"
        ] >= 3,
        "merge_review_score",
    ] += 1.0

    merged.loc[
        merged[
            "computed_transition_count"
        ] >= 3,
        "merge_review_score",
    ] += 1.0

    # Very long segments get a small additional priority.
    merged.loc[
        merged[
            "segment_duration_seconds"
        ] >= 120,
        "merge_review_score",
    ] += 1.0

    merged[
        "manual_review_priority"
    ] = pd.cut(
        merged[
            "merge_review_score"
        ],
        bins=[
            -np.inf,
            2,
            4,
            6,
            np.inf,
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "VERY_HIGH",
        ],
        right=False,
    )

    # --------------------------------------------------------
    # Save full diagnostics.
    # --------------------------------------------------------

    diagnostic_columns = [
        "session_id",
        "segment_index",
        "family_id",
        "start",
        "end",
        "segment_duration_seconds",
        "duration_to_family_median",
        "family_median_duration",
        "family_q95_duration",
        "family_q99_duration",
        "duration_above_family_q95_seconds",
        "duration_above_family_q99_seconds",
        "duration_q95_flag",
        "duration_q99_flag",
        "duration_2x_median_flag",
        "complexity_flag",
        "computed_app_count",
        "computed_transition_count",
        "computed_page_count",
        "computed_event_count",
        "interaction_density_per_10s",
        "clipboard_change_count",
        "browser_form_input_count",
        "browser_navigation_count",
        "browser_click_count",
        "mouse_click_count",
        "keyboard_count",
        "apps",
        "app_sequence",
        "browser_paths",
        "screenshot_count",
        "merge_review_score",
        "manual_review_priority",
    ]

    diagnostic_columns = [
        column
        for column in diagnostic_columns
        if column in merged.columns
    ]

    diagnostics = (
        merged[
            diagnostic_columns
        ]
        .sort_values(
            [
                "merge_review_score",
                "segment_duration_seconds",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    diagnostics.to_csv(
        DIAGNOSTICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Review queue.
    # --------------------------------------------------------

    review = diagnostics[
        (
            diagnostics[
                "duration_q95_flag"
            ]
            |
            diagnostics[
                "duration_2x_median_flag"
            ]
            |
            diagnostics[
                "complexity_flag"
            ]
        )
    ].copy()

    review = review.sort_values(
        [
            "merge_review_score",
            "segment_duration_seconds",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    review.insert(
        0,
        "review_rank",
        np.arange(
            1,
            len(review) + 1,
        ),
    )

    review.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Print family summary.
    # --------------------------------------------------------

    family_summary = (
        merged
        .groupby(
            "family_id",
            dropna=False,
        )
        .agg(
            segments=(
                "segment_index",
                "size",
            ),
            sessions=(
                "session_id",
                "nunique",
            ),
            total_observed_seconds=(
                "segment_duration_seconds",
                "sum",
            ),
            mean_seconds=(
                "segment_duration_seconds",
                "mean",
            ),
            median_seconds=(
                "segment_duration_seconds",
                "median",
            ),
            q95_seconds=(
                "segment_duration_seconds",
                lambda x:
                    x.quantile(
                        0.95
                    ),
            ),
            q99_seconds=(
                "segment_duration_seconds",
                lambda x:
                    x.quantile(
                        0.99
                    ),
            ),
            max_seconds=(
                "segment_duration_seconds",
                "max",
            ),
            flagged_q95=(
                "duration_q95_flag",
                "sum",
            ),
            flagged_2x_median=(
                "duration_2x_median_flag",
                "sum",
            ),
            complexity_flagged=(
                "complexity_flag",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "total_observed_seconds",
                "segments",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    print()
    print("=" * 70)
    print("FAMILY DURATION SUMMARY")
    print("=" * 70)

    print(
        family_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Print longest segments.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LONGEST STEP-1 SEGMENTS")
    print("=" * 70)

    longest_columns = [
        "session_id",
        "segment_index",
        "family_id",
        "segment_duration_seconds",
        "family_median_duration",
        "duration_to_family_median",
        "family_q95_duration",
        "family_q99_duration",
        "computed_app_count",
        "computed_transition_count",
        "computed_page_count",
        "computed_event_count",
        "merge_review_score",
        "manual_review_priority",
    ]

    longest_columns = [
        column
        for column in longest_columns
        if column in diagnostics.columns
    ]

    print(
        diagnostics[
            longest_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Print review queue.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MERGE-RISK REVIEW QUEUE")
    print("=" * 70)

    print(
        f"Segments flagged for review: "
        f"{len(review):,}"
    )

    if not review.empty:

        review_columns = [
            "review_rank",
            "session_id",
            "segment_index",
            "family_id",
            "segment_duration_seconds",
            "duration_to_family_median",
            "duration_q95_flag",
            "duration_q99_flag",
            "duration_2x_median_flag",
            "complexity_flag",
            "computed_app_count",
            "computed_transition_count",
            "computed_page_count",
            "computed_event_count",
            "merge_review_score",
            "manual_review_priority",
            "app_sequence",
            "browser_paths",
        ]

        review_columns = [
            column
            for column in review_columns
            if column in review.columns
        ]

        print(
            review[
                review_columns
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Final interpretation.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"Full diagnostics:\n"
        f"{DIAGNOSTICS_FILE}"
    )

    print(
        f"\nManual-review queue:\n"
        f"{REVIEW_FILE}"
    )

    print()
    print(
        "IMPORTANT: A flagged segment is NOT automatically a "
        "merged segment. It is a segment whose duration and/or "
        "internal complexity makes manual inspection worthwhile."
    )


if __name__ == "__main__":
    main()
