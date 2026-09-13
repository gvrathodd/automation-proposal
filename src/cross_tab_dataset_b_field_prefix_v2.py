
from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

FEATURES_FILE = (
    OUT / "dataset_b_process_features_with_operator_v2.csv"
)
CLUSTER_ASSIGNMENTS = (
    OUT / "dataset_b_structural_cluster_assignments_v2.csv"
)
CLUSTER_SUMMARY = (
    OUT / "dataset_b_structural_cluster_summary_v2.csv"
)

OUT_SEGMENT_AUDIT = (
    OUT / "dataset_b_field_prefix_cluster_audit.csv"
)
OUT_CROSS_TAB = (
    OUT / "dataset_b_field_prefix_cluster_crosstab.csv"
)
OUT_PREFIX_SUMMARY = (
    OUT / "dataset_b_field_prefix_summary.csv"
)
OUT_WINDOW_NORMALIZED = (
    OUT / "dataset_b_window_normalization_preview.csv"
)


def split_tokens(value):
    if value is None:
        return []

    text = str(value).strip()

    if text in {
        "",
        "nan",
        "None",
    }:
        return []

    return [
        x.strip()
        for x in text.split("|")
        if x.strip()
    ]


def normalize_window_title(value):
    value = str(value or "").strip().lower()

    if not value:
        return ""

    # Remove browser/profile/page-count chrome so one system remains
    # one token regardless of the number of tabs/pages represented.
    value = re.sub(
        r"\s*and \d+ more pages?\s*",
        " ",
        value,
    )

    value = re.sub(
        r"\s*profile \d+\s*",
        " ",
        value,
    )

    value = re.sub(
        r"\s*microsoft.?edge\s*$",
        "",
        value,
    )

    value = re.sub(
        r"\s*\[?compatibility mode\]?\s*$",
        "",
        value,
    )

    value = re.sub(
        r"\s*-\s*(google chrome|chrome)\s*$",
        "",
        value,
    )

    value = re.sub(
        r"\s*-\s*(microsoft word|word)\s*$",
        "",
        value,
    )

    value = re.sub(
        r"\s*-\s*(microsoft excel|excel)\s*$",
        "",
        value,
    )

    value = re.sub(
        r"\s*-\s*notepad\s*$",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def extract_prefixes(field_tokens):
    """
    Extract process-style prefixes from B DOM field tokens.

    Examples:
        field:id=pi-note -> pi
        field:id=btn-pi-ok -> pi
        field:name=pi-note -> pi
    """
    prefixes = set()

    for token in field_tokens:

        value = str(token).strip().lower()

        # Remove the field: wrapper if present.
        if value.startswith("field:"):
            value = value[len("field:"):]

        # Find common pi-note / btn-pi-ok shapes.
        patterns = [
            r"(?:^|=)([a-z]{2,8})-note$",
            r"(?:^|=)btn-([a-z]{2,8})-ok$",
            r"(?:^|=)([a-z]{2,8})_note$",
            r"(?:^|=)btn-([a-z]{2,8})_ok$",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                value,
            )

            if match:
                prefixes.add(
                    match.group(1)
                )

    return prefixes


def build_cluster_prefix_crosstab(
    features,
    clusters,
):
    base_features = features.copy()

    # The operator-enriched feature table may already contain a
    # structural_cluster column. The cluster-assignment file is the
    # authoritative source for this column, so remove any duplicate
    # before merging to avoid pandas _x/_y suffixes.
    if "structural_cluster" in base_features.columns:
        base_features = base_features.drop(
            columns=["structural_cluster"]
        )

    merged = base_features.merge(
        clusters[
            [
                "session_id",
                "segment_index",
                "structural_cluster",
            ]
        ],
        on=[
            "session_id",
            "segment_index",
        ],
        how="left",
        validate="one_to_one",
    )

    field_prefixes = []

    for value in merged[
        "field_token_set"
    ]:
        field_prefixes.append(
            extract_prefixes(
                split_tokens(value)
            )
        )

    merged[
        "field_prefixes"
    ] = [
        "|".join(sorted(x))
        for x in field_prefixes
    ]

    merged[
        "field_prefix_count"
    ] = [
        len(x)
        for x in field_prefixes
    ]

    merged[
        "window_normalized"
    ] = [
        "|".join(
            sorted(
                normalize_window_title(x)
                for x in split_tokens(value)
                if normalize_window_title(x)
            )
        )
        for value in merged[
            "window_token_set"
        ]
    ]

    # For an interpretable cross-tab, choose one prefix where possible.
    # Multi-prefix segments are explicitly marked.
    merged[
        "primary_field_prefix"
    ] = [
        (
            next(iter(sorted(x)))
            if len(x) == 1
            else (
                "MULTI"
                if len(x) > 1
                else "NONE"
            )
        )
        for x in field_prefixes
    ]

    return merged


def main():
    print("=" * 70)
    print(
        "DATASET B FIELD-PREFIX × STRUCTURAL-CLUSTER AUDIT"
    )
    print("=" * 70)

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            FEATURES_FILE
        )

    if not CLUSTER_ASSIGNMENTS.exists():
        raise FileNotFoundError(
            CLUSTER_ASSIGNMENTS
        )

    features = pd.read_csv(
        FEATURES_FILE,
        keep_default_na=False,
    )

    clusters = pd.read_csv(
        CLUSTER_ASSIGNMENTS,
        keep_default_na=False,
    )

    print()
    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Cluster assignment rows: {len(clusters):,}"
    )

    # --------------------------------------------------------
    # Normalize window titles preview.
    # --------------------------------------------------------

    preview_rows = []

    for _, row in features.iterrows():

        original = split_tokens(
            row.get(
                "window_token_set",
                "",
            )
        )

        normalized = sorted(
            {
                normalize_window_title(x)
                for x in original
                if normalize_window_title(x)
            }
        )

        for before, after in zip(
            original,
            [
                normalize_window_title(x)
                for x in original
            ],
        ):
            preview_rows.append(
                {
                    "session_id":
                        row[
                            "session_id"
                        ],
                    "segment_index":
                        row[
                            "segment_index"
                        ],
                    "original_window_token":
                        before,
                    "normalized_window_token":
                        after,
                }
            )

    pd.DataFrame(
        preview_rows
    ).drop_duplicates().to_csv(
        OUT_WINDOW_NORMALIZED,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Build cross-tab.
    # --------------------------------------------------------

    merged = build_cluster_prefix_crosstab(
        features,
        clusters,
    )

    print()
    print("=" * 70)
    print(
        "FIELD PREFIX SUMMARY"
    )
    print("=" * 70)

    prefix_counts = Counter()

    for prefixes in merged[
        "field_prefixes"
    ]:
        for prefix in split_tokens(
            prefixes
        ):
            prefix_counts[prefix] += 1

    prefix_summary_rows = []

    for prefix, count in prefix_counts.most_common():
        subset = merged[
            merged[
                "field_prefixes"
            ].str.contains(
                rf"(?:^|\|){re.escape(prefix)}(?:\||$)",
                regex=True,
            )
        ]

        prefix_summary_rows.append(
            {
                "field_prefix":
                    prefix,
                "segments_with_prefix":
                    len(subset),
                "fraction_of_segments":
                    len(subset) / len(merged),
                "distinct_structural_clusters":
                    subset[
                        "structural_cluster"
                    ].nunique(),
                "distinct_operators":
                    (
                        len(
                            set(
                                h
                                for value
                                in subset[
                                    "operator_hashes"
                                ]
                                for h in value.split("|")
                                if h.strip()
                            )
                        )
                    ),
                "median_duration_seconds":
                    subset[
                        "duration_seconds"
                    ].median(),
                "total_duration_seconds":
                    subset[
                        "duration_seconds"
                    ].sum(),
                "segments_with_word":
                    int(
                        subset[
                            "has_word"
                        ].astype(int).sum()
                    ),
                "segments_with_notepad":
                    int(
                        subset[
                            "has_notepad"
                        ].astype(int).sum()
                    ),
                "segments_with_excel":
                    int(
                        subset[
                            "has_excel"
                        ].astype(int).sum()
                    ),
            }
        )

    prefix_summary = pd.DataFrame(
        prefix_summary_rows
    )

    print(
        prefix_summary.to_string(
            index=False
        )
    )

    prefix_summary.to_csv(
        OUT_PREFIX_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prefix × structural cluster count table.
    # --------------------------------------------------------

    crosstab = pd.crosstab(
        merged[
            "primary_field_prefix"
        ],
        merged[
            "structural_cluster"
        ],
    )

    crosstab.to_csv(
        OUT_CROSS_TAB,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print(
        "FIELD PREFIX × STRUCTURAL CLUSTER"
    )
    print("=" * 70)

    # Print a compact top-cluster view.
    cluster_sizes = (
        merged[
            "structural_cluster"
        ]
        .value_counts()
        .head(25)
        .index
        .tolist()
    )

    compact = crosstab.reindex(
        columns=cluster_sizes,
        fill_value=0,
    )

    print(
        compact.to_string()
    )

    # --------------------------------------------------------
    # For each prefix, identify whether it maps mostly to a
    # small number of clusters or is spread widely.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PREFIX → TOP CLUSTERS"
    )
    print("=" * 70)

    audit_rows = []

    for prefix in sorted(
        prefix_counts
    ):

        subset = merged[
            merged[
                "field_prefixes"
            ].str.contains(
                rf"(?:^|\|){re.escape(prefix)}(?:\||$)",
                regex=True,
            )
        ]

        cluster_counts = (
            subset[
                "structural_cluster"
            ]
            .value_counts()
        )

        top = cluster_counts.head(
            10
        )

        top_text = " | ".join(
            f"{int(k)}:{int(v)}"
            for k, v
            in top.items()
        )

        audit_rows.append(
            {
                "field_prefix":
                    prefix,
                "segments":
                    len(subset),
                "clusters":
                    subset[
                        "structural_cluster"
                    ].nunique(),
                "largest_cluster_share":
                    (
                        top.iloc[0]
                        / len(subset)
                        if len(top)
                        else 0.0
                    ),
                "top_clusters":
                    top_text,
            }
        )

        print(
            f"{prefix:8s} "
            f"segments={len(subset):3d} "
            f"clusters={subset['structural_cluster'].nunique():3d} "
            f"top={top_text}"
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        OUT_SEGMENT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Variant evidence:
    # Prefix × application pattern.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PREFIX × APPLICATION PATTERN"
    )
    print("=" * 70)

    variant_rows = []

    for prefix in sorted(
        prefix_counts
    ):

        subset = merged[
            merged[
                "field_prefixes"
            ].str.contains(
                rf"(?:^|\|){re.escape(prefix)}(?:\||$)",
                regex=True,
            )
        ]

        word = int(
            subset[
                "has_word"
            ].sum()
        )

        notepad = int(
            subset[
                "has_notepad"
            ].sum()
        )

        excel = int(
            subset[
                "has_excel"
            ].sum()
        )

        browser_only = int(
            (
                (
                    subset[
                        "has_browser"
                    ]
                    == 1
                )
                & (
                    subset[
                        "has_word"
                    ]
                    == 0
                )
                & (
                    subset[
                        "has_notepad"
                    ]
                    == 0
                )
                & (
                    subset[
                        "has_excel"
                    ]
                    == 0
                )
            ).sum()
        )

        variant_rows.append(
            {
                "field_prefix":
                    prefix,
                "browser_only":
                    browser_only,
                "word_assisted":
                    word,
                "notepad_assisted":
                    notepad,
                "excel_assisted":
                    excel,
            }
        )

    variant_df = pd.DataFrame(
        variant_rows
    )

    print(
        variant_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Integrity checks.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "INTEGRITY CHECKS"
    )
    print("=" * 70)

    if len(merged) != len(features):
        raise AssertionError(
            "Feature/cluster merge changed row count."
        )

    if "structural_cluster" not in merged.columns:
        raise AssertionError(
            "Authoritative structural_cluster column missing after merge."
        )

    missing_clusters = int(
        merged[
            "structural_cluster"
        ].isna().sum()
    )

    if missing_clusters:
        raise AssertionError(
            f"{missing_clusters} feature rows have no structural cluster."
        )

    print(
        "Feature → cluster merge: PASSED"
    )

    print(
        f"Segments with exactly one field prefix: "
        f"{int((merged['field_prefix_count'] == 1).sum()):,}"
    )

    print(
        f"Segments with multiple field prefixes: "
        f"{int((merged['field_prefix_count'] > 1).sum()):,}"
    )

    print(
        f"Segments with no recognized field prefix: "
        f"{int((merged['field_prefix_count'] == 0).sum()):,}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUT_SEGMENT_AUDIT
    )

    print(
        OUT_CROSS_TAB
    )

    print(
        OUT_PREFIX_SUMMARY
    )

    print(
        OUT_WINDOW_NORMALIZED
    )

    print()
    print(
        "Next decision:"
    )

    print(
        "If the five prefixes are clean and each splits into "
        "recognizable app-based variants, use the prefixes as "
        "the initial B process families and use OCR only for "
        "business naming/evidence."
    )


if __name__ == "__main__":
    main()
