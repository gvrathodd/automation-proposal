
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

INPUT_FILE = (
    OUTPUTS / "labeler_gt_executions_with_signatures_v3.csv"
)

OUT_RESULTS = (
    OUTPUTS / "labeler_jaccard_sweep_results.csv"
)

OUT_CLUSTER_DETAILS = (
    OUTPUTS / "labeler_jaccard_cluster_details.csv"
)

OUT_ASSIGNMENTS = (
    OUTPUTS / "labeler_jaccard_best_assignments.csv"
)


# ============================================================
# WHAT THIS EXPERIMENT TESTS
# ============================================================
#
# Previous exact-set hashing produced:
#   666 clusters / 1,752 executions
#   63% singleton clusters
#   ARI ~ 0.10
#
# We now keep the same underlying evidence but fix granularity:
#
#   1. Coarsen window titles to stable system/document identity.
#   2. Build token sets instead of one giant exact tuple.
#   3. Remove ubiquitous tokens (>80% of executions).
#   4. Remove one-off tokens (<2 executions).
#   5. Cluster using Jaccard similarity.
#
# We test similarity thresholds 0.55..0.75 and report:
#   - ARI (primary)
#   - weighted purity
#   - cluster count
#   - singleton fraction
#   - largest cluster
#
# We explicitly DO NOT use case_id / employee IDs.
# We also DO NOT use the previous flags component.
#
# This is Dataset-A validation only.
# ============================================================


COMPONENTS = {
    "url": "url_paths",
    "apps": "apps",
    "windows": "window_titles",
    "fields": "form_fields",
}

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def parse_collection(value):
    if value is None:
        return []

    text = str(value).strip()

    if (
        not text
        or text in {"nan", "None", "()", "[]"}
    ):
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []

    if isinstance(parsed, (tuple, list, set)):
        return [
            str(x).strip()
            for x in parsed
            if str(x).strip()
        ]

    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]

    return []


def normalize_basic_token(text):
    text = str(text).strip().lower()

    if not text:
        return ""

    # Normalize obvious volatile identifiers if any survived upstream.
    text = re.sub(
        r"\bE\d{3,8}\b",
        "{id}",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\b[A-Z]{1,8}-\d{4,12}(?:-\d{1,6})?\b",
        "{id}",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\b\d{6,}\b",
        "{n}",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def coarsen_window_title(title):
    """
    Keep stable semantic carriers and remove volatile chrome.

    We prefer:
      - known system/application names
      - document/file basename-like components
      - stable title tokens when no stronger structure exists

    The output remains Japanese-safe; no translation is needed.
    """

    value = normalize_basic_token(
        title
    )

    if not value:
        return ""

    # Drop obvious browser/application chrome.
    drop_patterns = [
        r"\s*-\s*google chrome.*$",
        r"\s*-\s*microsoft edge.*$",
        r"\s*-\s*notepad.*$",
        r"\s*-\s*microsoft word.*$",
        r"\s*-\s*microsoft excel.*$",
        r"\s*-\s*word.*$",
        r"\s*-\s*excel.*$",
    ]

    for pattern in drop_patterns:
        value = re.sub(
            pattern,
            "",
            value,
            flags=re.I,
        )

    # Drop common desktop-agent/setup noise.
    if (
        "procmine-desktop-agent" in value
        or value in {
            "settings",
            "setup",
        }
    ):
        return ""

    # Normalize separators.
    value = re.sub(
        r"[_\-]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def field_token(value):
    """
    Reduce verbose DOM attributes to a stable structural field token.
    """
    value = normalize_basic_token(
        value
    )

    if not value:
        return ""

    # Preserve attribute type and structural name/placeholder.
    if "=" in value:
        key, val = value.split(
            "=",
            1,
        )

        key = key.strip()
        val = val.strip()

        val = re.sub(
            r"\s+",
            "_",
            val,
        )

        return f"field:{key}:{val}"

    return f"field:{value}"


def build_raw_token_sets(df):
    rows = []

    for _, row in df.iterrows():

        token_sets = {}

        # URL paths.
        url_tokens = {
            f"url:{normalize_basic_token(x)}"
            for x in parse_collection(
                row["url_paths"]
            )
            if normalize_basic_token(x)
        }

        token_sets["url"] = url_tokens

        # Apps.
        app_tokens = {
            f"app:{normalize_basic_token(x)}"
            for x in parse_collection(
                row["apps"]
            )
            if normalize_basic_token(x)
        }

        token_sets["apps"] = app_tokens

        # Window titles.
        window_tokens = set()

        for title in parse_collection(
            row["window_titles"]
        ):

            coarse = coarsen_window_title(
                title
            )

            if coarse:
                # Keep the whole coarse identity as one token.
                window_tokens.add(
                    f"window:{coarse}"
                )

        token_sets["windows"] = window_tokens

        # Form fields.
        field_tokens = {
            field_token(x)
            for x in parse_collection(
                row["form_fields"]
            )
        }

        field_tokens = {
            x
            for x in field_tokens
            if x
        }

        token_sets["fields"] = field_tokens

        combined = set().union(
            *token_sets.values()
        )

        rows.append(
            combined
        )

    return rows


def apply_document_frequency_stoplist(
    token_sets,
    n,
):
    """
    Keep tokens appearing in [2, 80% of executions].
    """

    df_counts = Counter()

    for tokens in token_sets:
        for token in tokens:
            df_counts[token] += 1

    min_count = 2
    max_count = int(
        np.floor(
            0.80 * n
        )
    )

    kept = {
        token
        for token, count in df_counts.items()
        if (
            count >= min_count
            and count <= max_count
        )
    }

    filtered = [
        tokens.intersection(
            kept
        )
        for tokens in token_sets
    ]

    return (
        filtered,
        df_counts,
        kept,
    )


def jaccard_distance_matrix(
    token_sets,
):
    n = len(token_sets)

    matrix = np.ones(
        (n, n),
        dtype=np.float32,
    )

    np.fill_diagonal(
        matrix,
        0.0,
    )

    for i in range(n):
        a = token_sets[i]

        if not a:
            # Empty sets cannot be meaningfully compared.
            # Keep them at maximal distance from non-identical
            # examples, but identical empty examples get 0.
            for j in range(i + 1, n):
                b = token_sets[j]

                if not b:
                    distance = 0.0
                else:
                    distance = 1.0

                matrix[i, j] = distance
                matrix[j, i] = distance

            continue

        for j in range(i + 1, n):
            b = token_sets[j]

            if not b:
                distance = 1.0
            else:
                union_size = len(
                    a | b
                )

                if union_size == 0:
                    distance = 0.0
                else:
                    similarity = (
                        len(
                            a & b
                        )
                        / union_size
                    )

                    distance = (
                        1.0
                        - similarity
                    )

            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix


def evaluate_cluster_labels(
    labels,
    true_labels,
):
    cluster_count = len(
        np.unique(labels)
    )

    counts = Counter(
        labels
    )

    singleton_fraction = (
        sum(
            1
            for count in counts.values()
            if count == 1
        )
        / cluster_count
        if cluster_count
        else np.nan
    )

    largest_cluster_size = (
        max(
            counts.values()
        )
        if counts
        else 0
    )

    # Cluster purity weighted by cluster size.
    weighted_purity = 0.0

    for cluster_label in np.unique(
        labels
    ):

        indices = np.where(
            labels == cluster_label
        )[0]

        cluster_truth = [
            true_labels[i]
            for i in indices
        ]

        dominant_count = Counter(
            cluster_truth
        ).most_common(1)[0][1]

        weighted_purity += (
            dominant_count
        )

    weighted_purity /= len(
        true_labels
    )

    ari = adjusted_rand_score(
        true_labels,
        labels,
    )

    return {
        "clusters": cluster_count,
        "weighted_purity":
            weighted_purity,
        "ARI": ari,
        "singleton_fraction":
            singleton_fraction,
        "largest_cluster_size":
            largest_cluster_size,
    }


def run():
    print("=" * 70)
    print(
        "DATASET-A JACCARD / COARSENING LABELER EXPERIMENT V3"
    )
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    required = {
        "process_code",
        "url_paths",
        "apps",
        "window_titles",
        "form_fields",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    print(
        f"Executions: "
        f"{len(df):,}"
    )

    print(
        f"True process codes: "
        f"{df['process_code'].nunique():,}"
    )

    raw_sets = build_raw_token_sets(
        df
    )

    filtered_sets, token_df, kept_tokens = (
        apply_document_frequency_stoplist(
            raw_sets,
            len(df),
        )
    )

    print()
    print(
        "Token audit:"
    )

    print(
        f"Unique raw tokens: "
        f"{len(token_df):,}"
    )

    print(
        f"Tokens retained after DF stop-list: "
        f"{len(kept_tokens):,}"
    )

    print(
        f"Removed ubiquitous tokens (>80%): "
        f"{sum(1 for c in token_df.values() if c > 0.80 * len(df)):,}"
    )

    print(
        f"Removed rare tokens (<2): "
        f"{sum(1 for c in token_df.values() if c < 2):,}"
    )

    token_counts = [
        len(tokens)
        for tokens in filtered_sets
    ]

    print()
    print(
        "Filtered token-set size:"
    )

    print(
        pd.Series(
            token_counts
        ).describe().to_string()
    )

    # --------------------------------------------------------
    # Test the cumulative component configurations.
    #
    # flags are deliberately excluded.
    # --------------------------------------------------------

    configurations = [
        (
            "url",
            ["url"],
        ),
        (
            "url+apps",
            ["url", "apps"],
        ),
        (
            "url+apps+windows",
            ["url", "apps", "windows"],
        ),
        (
            "url+apps+windows+fields",
            [
                "url",
                "apps",
                "windows",
                "fields",
            ],
        ),
    ]

    result_rows = []
    assignment_frames = []

    true_labels = (
        df[
            "process_code"
        ]
        .astype(str)
        .tolist()
    )

    # Build component-level filtered sets with the SAME
    # document-frequency stop list.
    component_sets = {}

    for component_name, column_name in COMPONENTS.items():
        component_rows = []

        for _, row in df.iterrows():

            if component_name == "windows":
                values = [
                    coarsen_window_title(x)
                    for x in parse_collection(
                        row[column_name]
                    )
                ]

                values = [
                    x
                    for x in values
                    if x
                ]

                tokens = {
                    f"window:{x}"
                    for x in values
                }

            elif component_name == "fields":
                tokens = {
                    field_token(x)
                    for x in parse_collection(
                        row[column_name]
                    )
                }

                tokens = {
                    x
                    for x in tokens
                    if x
                }

            else:
                prefix = (
                    "url"
                    if component_name == "url"
                    else "app"
                )

                tokens = {
                    f"{prefix}:{normalize_basic_token(x)}"
                    for x in parse_collection(
                        row[column_name]
                    )
                    if normalize_basic_token(x)
                }

            component_rows.append(
                tokens
            )

        component_df = Counter()

        for tokens in component_rows:
            for token in tokens:
                component_df[token] += 1

        allowed = {
            token
            for token, count
            in component_df.items()
            if (
                count >= 2
                and count <= int(
                    np.floor(
                        0.80 * len(df)
                    )
                )
            )
        }

        component_sets[
            component_name
        ] = [
            tokens.intersection(
                allowed
            )
            for tokens in component_rows
        ]

    for config_name, components in configurations:

        sets_for_config = [
            set().union(
                *[
                    component_sets[
                        component
                    ][i]
                    for component
                    in components
                ]
            )
            for i in range(
                len(df)
            )
        ]

        print()
        print(
            "=" * 70
        )
        print(
            f"CONFIGURATION: {config_name}"
        )
        print(
            "=" * 70
        )

        print(
            f"Non-empty token sets: "
            f"{sum(bool(x) for x in sets_for_config):,}"
            f" / {len(sets_for_config):,}"
        )

        print(
            "Computing Jaccard distance matrix..."
        )

        distance_matrix = (
            jaccard_distance_matrix(
                sets_for_config
            )
        )

        for threshold in THRESHOLDS:

            print(
                f"  clustering similarity >= "
                f"{threshold:.2f}..."
            )

            distance_threshold = (
                1.0 - threshold
            )

            clustering = (
                AgglomerativeClustering(
                    metric="precomputed",
                    linkage="average",
                    distance_threshold=distance_threshold,
                    n_clusters=None,
                )
            )

            labels = clustering.fit_predict(
                distance_matrix
            )

            metrics = evaluate_cluster_labels(
                labels,
                true_labels,
            )

            result_rows.append(
                {
                    "configuration":
                        config_name,
                    "similarity_threshold":
                        threshold,
                    **metrics,
                    "executions":
                        len(df),
                    "true_process_codes":
                        len(
                            set(
                                true_labels
                            )
                        ),
                }
            )

            # Keep assignments so the best configuration can be
            # selected later.
            frame = pd.DataFrame(
                {
                    "gt_execution_id":
                        df[
                            "gt_execution_id"
                        ].astype(str).values,
                    "process_code":
                        true_labels,
                    "cluster_label":
                        labels,
                    "configuration":
                        config_name,
                    "similarity_threshold":
                        threshold,
                }
            )

            assignment_frames.append(
                frame
            )

    results = pd.DataFrame(
        result_rows
    )

    # --------------------------------------------------------
    # Select best defensible configuration.
    #
    # Primary: ARI.
    # Secondary: cluster count between 15 and 40.
    # If none lands in that range, report the highest ARI.
    # --------------------------------------------------------

    in_range = results[
        results[
            "clusters"
        ].between(
            15,
            40,
        )
    ].copy()

    if not in_range.empty:
        best_row = (
            in_range
            .sort_values(
                [
                    "ARI",
                    "singleton_fraction",
                    "clusters",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )
    else:
        best_row = (
            results
            .sort_values(
                [
                    "ARI",
                    "singleton_fraction",
                    "clusters",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

    results[
        "selected_as_best"
    ] = False

    results.loc[
        (
            results[
                "configuration"
            ]
            == best_row[
                "configuration"
            ]
        )
        & (
            results[
                "similarity_threshold"
            ]
            == best_row[
                "similarity_threshold"
            ]
        ),
        "selected_as_best",
    ] = True

    # --------------------------------------------------------
    # Save metrics.
    # --------------------------------------------------------

    results.to_csv(
        OUT_RESULTS,
        index=False,
        encoding="utf-8-sig",
    )

    all_assignments = pd.concat(
        assignment_frames,
        ignore_index=True,
    )

    # Filter to best configuration.
    best_assignments = (
        all_assignments[
            (
                all_assignments[
                    "configuration"
                ]
                == best_row[
                    "configuration"
                ]
            )
            & (
                all_assignments[
                    "similarity_threshold"
                ]
                == best_row[
                    "similarity_threshold"
                ]
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Merge by the stable execution key. Never rely on row position.
    base = df[
        [
            "session_id",
            "gt_execution_id",
            "process_code",
            "process_name",
        ]
    ].copy()

    base[
        "gt_execution_id"
    ] = base[
        "gt_execution_id"
    ].astype(str)

    best_assignments[
        "gt_execution_id"
    ] = best_assignments[
        "gt_execution_id"
    ].astype(str)

    if best_assignments[
        "gt_execution_id"
    ].duplicated().any():
        raise AssertionError(
            "Duplicate gt_execution_id found in best assignments."
        )

    best_assignments = base.merge(
        best_assignments[
            [
                "gt_execution_id",
                "cluster_label",
                "configuration",
                "similarity_threshold",
            ]
        ],
        on="gt_execution_id",
        how="left",
        validate="one_to_one",
        suffixes=("_base", "_assignment"),
    )

    if best_assignments[
        "cluster_label"
    ].isna().any():
        missing_count = int(
            best_assignments[
                "cluster_label"
            ].isna().sum()
        )
        raise AssertionError(
            f"{missing_count} GT executions lost their cluster assignment."
        )

    # Verify that the process code used for clustering survived the
    # keyed merge unchanged.
    # The assignment frame's process_code is already represented by
    # the base table, so we additionally reconstruct it from the
    # selected threshold frame and compare explicitly.
    selected_assignment_codes = (
        all_assignments[
            (
                all_assignments[
                    "configuration"
                ]
                == best_row[
                    "configuration"
                ]
            )
            & (
                all_assignments[
                    "similarity_threshold"
                ]
                == best_row[
                    "similarity_threshold"
                ]
            )
        ][
            [
                "gt_execution_id",
                "process_code",
            ]
        ]
        .copy()
    )

    selected_assignment_codes[
        "gt_execution_id"
    ] = selected_assignment_codes[
        "gt_execution_id"
    ].astype(str)

    check = best_assignments.merge(
        selected_assignment_codes,
        on="gt_execution_id",
        how="left",
        validate="one_to_one",
        suffixes=("_base", "_cluster"),
    )

    if not (
        check[
            "process_code_base"
        ].astype(str)
        == check[
            "process_code_cluster"
        ].astype(str)
    ).all():
        raise AssertionError(
            "Process-code mismatch after keyed assignment merge."
        )

    print(
        "Assignment-key integrity check: PASSED "
        f"({len(check):,} executions, one-to-one by gt_execution_id)."
    )

    best_assignments.to_csv(
        OUT_ASSIGNMENTS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Cluster details for selected configuration.
    # --------------------------------------------------------

    cluster_detail_rows = []

    for cluster_label, group in (
        best_assignments.groupby(
            "cluster_label"
        )
    ):

        counts = Counter(
            group[
                "process_code"
            ]
        )

        dominant_code, dominant_count = (
            counts.most_common(1)[0]
        )

        cluster_detail_rows.append(
            {
                "configuration":
                    best_row[
                        "configuration"
                    ],
                "similarity_threshold":
                    best_row[
                        "similarity_threshold"
                    ],
                "cluster_label":
                    cluster_label,
                "cluster_size":
                    len(group),
                "dominant_process_code":
                    dominant_code,
                "dominant_count":
                    dominant_count,
                "cluster_purity":
                    dominant_count / len(group),
                "distinct_process_codes":
                    len(counts),
            }
        )

    cluster_details = (
        pd.DataFrame(
            cluster_detail_rows
        )
        .sort_values(
            "cluster_size",
            ascending=False,
        )
    )

    cluster_details.to_csv(
        OUT_CLUSTER_DETAILS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "JACCARD SWEEP RESULTS"
    )
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "BEST DEFENSIBLE CONFIGURATION"
    )
    print("=" * 70)

    print(
        best_row.to_string()
    )

    print()
    print(
        "Interpretation:"
    )
    print(
        "ARI is the primary metric. Purity is only read together "
        "with cluster count and singleton fraction."
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUT_RESULTS
    )

    print(
        OUT_CLUSTER_DETAILS
    )

    print(
        OUT_ASSIGNMENTS
    )


if __name__ == "__main__":
    run()
