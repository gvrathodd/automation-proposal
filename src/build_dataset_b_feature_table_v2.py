
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"

FEATURES_FILE = OUT / "dataset_b_process_features.csv"
SUMMARY_FILE = OUT / "dataset_b_feature_summary.csv"
TOKENS_FILE = OUT / "dataset_b_token_inventory.csv"

OUT_OPERATOR_FEATURES = (
    OUT / "dataset_b_process_features_with_operator_v2.csv"
)
OUT_CLUSTER_SUMMARY = (
    OUT / "dataset_b_structural_cluster_summary_v2.csv"
)
OUT_CLUSTER_ASSIGNMENTS = (
    OUT / "dataset_b_structural_cluster_assignments_v2.csv"
)
OUT_CLUSTER_TOKENS = (
    OUT / "dataset_b_structural_cluster_tokens_v2.csv"
)


# ============================================================
# Goal
# ============================================================
#
# BEFORE OCR:
#
# 1. Print the B summary and token inventory.
# 2. Measure:
#      - segments with form fields
#      - distinct URL tokens
#      - distinct window tokens
# 3. Run the same structural Jaccard clustering idea used on A,
#    using URL + apps + windows + fields, but:
#      - same DF stop-list: >=2 and <=80%
#      - similarity threshold: 0.40
#      - no ground-truth metric
# 4. Inspect cluster count / distribution / representative tokens.
# 5. Add operator identity from raw event source.username_hash.
#
# NO OCR.
# NO translation.
# NO changes to Step 1.
# NO final labels.
# ============================================================


def parse_collection(value):
    if value is None:
        return []

    text = str(value).strip()

    if text in {
        "",
        "nan",
        "None",
        "()",
        "[]",
    }:
        return []

    # Feature-table token columns are stored as pipe-delimited strings.
    if "|" in text:
        return [
            part.strip()
            for part in text.split("|")
            if part.strip()
        ]

    # Keep compatibility with older tuple/list representations.
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text] if text else []

    if isinstance(
        parsed,
        (list, tuple, set),
    ):
        return [
            str(x).strip()
            for x in parsed
            if str(x).strip()
        ]

    if isinstance(
        parsed,
        str,
    ) and parsed.strip():
        return [parsed.strip()]

    return []


def normalize_token(text):
    text = str(text or "").strip().lower()

    if not text:
        return ""

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


def build_component_tokens(row):
    tokens = set()

    for value in parse_collection(
        row.get("url_token_set", "")
    ):
        value = normalize_token(value)
        if value:
            tokens.add(
                f"url:{value}"
            )

    for value in parse_collection(
        row.get("apps_token_set", "")
    ):
        value = normalize_token(value)
        if value:
            tokens.add(
                f"app:{value}"
            )

    for value in parse_collection(
        row.get("window_token_set", "")
    ):
        value = normalize_token(value)
        if value:
            tokens.add(
                f"window:{value}"
            )

    for value in parse_collection(
        row.get("field_token_set", "")
    ):
        value = normalize_token(value)
        if value:
            tokens.add(
                f"field:{value}"
            )

    return tokens


def apply_stoplist(
    token_sets,
    min_df,
    max_fraction,
):
    counts = Counter()

    for token_set in token_sets:
        for token in token_set:
            counts[token] += 1

    max_df = int(
        np.floor(
            max_fraction
            * len(token_sets)
        )
    )

    allowed = {
        token
        for token, count in counts.items()
        if (
            count >= min_df
            and count <= max_df
        )
    }

    filtered = [
        token_set.intersection(
            allowed
        )
        for token_set in token_sets
    ]

    return (
        filtered,
        counts,
        allowed,
    )


def jaccard_matrix(
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

        for j in range(
            i + 1,
            n,
        ):
            b = token_sets[j]

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


# ============================================================
# Operator extraction
# ============================================================

def parse_event_timestamp(event):
    if event.get(
        "timestamp_ms"
    ) is not None:
        try:
            return pd.to_datetime(
                int(
                    event[
                        "timestamp_ms"
                    ]
                ),
                unit="ms",
                utc=True,
            )
        except Exception:
            pass

    for key in (
        "timestamp_iso",
        "timestamp",
        "ts_utc",
    ):
        if key in event:
            try:
                value = pd.to_datetime(
                    event[key],
                    utc=True,
                )

                if not pd.isna(value):
                    return value

            except Exception:
                pass

    return pd.NaT


def extract_operator_hashes(event):
    """
    Search likely source locations for username/operator identity.
    We keep only hashes/IDs, never raw personal names.
    """
    candidates = []

    source = event.get(
        "source"
    )

    if isinstance(
        source,
        dict,
    ):
        for key in (
            "username_hash",
            "user_hash",
            "operator_hash",
        ):
            value = source.get(key)

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                candidates.append(
                    value.strip()
                )

    # A defensive recursive search for the documented field.
    def walk(obj):
        if isinstance(
            obj,
            dict,
        ):
            for key, value in obj.items():
                if str(key).lower() == "username_hash":
                    if (
                        isinstance(
                            value,
                            str,
                        )
                        and value.strip()
                    ):
                        candidates.append(
                            value.strip()
                        )

                walk(value)

        elif isinstance(
            obj,
            list,
        ):
            for item in obj:
                walk(item)

    walk(event)

    return {
        value
        for value in candidates
        if value
    }


def build_b_session_index():
    index = {}

    dataset_roots = list(
        DATA_ROOT.rglob("dataset_b")
    )

    for root in dataset_roots:
        try:
            children = list(root.iterdir())
        except OSError:
            continue

        for child in children:
            if not child.is_dir():
                continue

            if not child.name.startswith("ses_"):
                continue

            event_files = list(
                child.rglob("events.jsonl")
            )

            if not event_files:
                continue

            total_size = sum(
                f.stat().st_size
                for f in event_files
                if f.exists()
            )

            candidate = (
                len(event_files),
                total_size,
                str(child),
                child,
            )

            current = index.get(
                child.name
            )

            if (
                current is None
                or candidate[:3]
                > current[:3]
            ):
                index[
                    child.name
                ] = candidate

    return {
        session_id: value[-1]
        for session_id, value in index.items()
    }


def find_b_session(
    session_id,
    session_index=None,
):
    if session_index is None:
        session_index = build_b_session_index()

    return session_index.get(
        session_id
    )


def load_session_operator_events(
    session_id,
    session_index=None,
):
    folder = find_b_session(
        session_id,
        session_index=session_index,
    )

    if folder is None:
        return []

    rows = []

    for event_file in sorted(
        folder.rglob(
            "events.jsonl"
        )
    ):
        try:
            with event_file.open(
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        event = json.loads(
                            line
                        )
                    except Exception:
                        continue

                    timestamp = parse_event_timestamp(
                        event
                    )

                    if pd.isna(
                        timestamp
                    ):
                        continue

                    hashes = (
                        extract_operator_hashes(
                            event
                        )
                    )

                    rows.append(
                        (
                            timestamp,
                            hashes,
                        )
                    )

        except OSError:
            continue

    rows.sort(
        key=lambda x: x[0]
    )

    return rows


def operator_hashes_for_segment(
    events,
    start,
    end,
):
    found = set()

    for timestamp, hashes in events:
        if (
            timestamp >= start
            and timestamp <= end
        ):
            found.update(
                hashes
            )

    return found


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "DATASET B PRE-OCR STRUCTURAL AUDIT V2"
    )
    print("=" * 70)

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            FEATURES_FILE
        )

    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            SUMMARY_FILE
        )

    if not TOKENS_FILE.exists():
        raise FileNotFoundError(
            TOKENS_FILE
        )

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    tokens = pd.read_csv(
        TOKENS_FILE
    )

    features = pd.read_csv(
        FEATURES_FILE,
        keep_default_na=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "B FEATURE SUMMARY"
    )
    print("=" * 70)

    print(
        summary.T.to_string(
            header=False
        )
    )

    # --------------------------------------------------------
    # Token inventory
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "B TOKEN INVENTORY"
    )
    print("=" * 70)

    for prefix in (
        "url",
        "window",
        "field",
        "apps",
    ):
        sub = tokens[
            tokens[
                "token"
            ]
            .astype(str)
            .str.startswith(
                prefix + ":"
            )
        ].copy()

        print()
        print(
            f"{prefix}: "
            f"{len(sub):,} distinct tokens"
        )

        print(
            sub.head(
                25
            ).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Explicit load-bearing metrics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LOAD-BEARING B METRICS"
    )
    print("=" * 70)

    segments = len(features)

    form_count = int(
        (
            pd.to_numeric(
                features[
                    "unique_form_fields"
                ],
                errors="coerce",
            ).fillna(0)
            > 0
        ).sum()
    )

    window_count = int(
        (
            pd.to_numeric(
                features[
                    "unique_window_identities"
                ],
                errors="coerce",
            ).fillna(0)
            > 0
        ).sum()
    )

    print(
        f"Segments with form fields: "
        f"{form_count:,} / {segments:,} "
        f"({form_count / segments:.2%})"
    )

    print(
        f"Distinct URL tokens: "
        f"{len(tokens[tokens.token.astype(str).str.startswith('url:')]):,}"
    )

    print(
        f"Distinct window tokens: "
        f"{len(tokens[tokens.token.astype(str).str.startswith('window:')]):,}"
    )

    # --------------------------------------------------------
    # Build structural tokens
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BUILDING B STRUCTURAL TOKENS"
    )
    print("=" * 70)

    raw_sets = [
        build_component_tokens(
            row
        )
        for _, row
        in features.iterrows()
    ]

    filtered_sets, token_df, allowed = (
        apply_stoplist(
            raw_sets,
            min_df=2,
            max_fraction=0.80,
        )
    )

    print(
        f"Raw distinct structural tokens: "
        f"{len(token_df):,}"
    )

    print(
        f"Retained after DF stop-list: "
        f"{len(allowed):,}"
    )

    print(
        f"Removed rare (<2): "
        f"{sum(1 for v in token_df.values() if v < 2):,}"
    )

    print(
        f"Removed ubiquitous (>80%): "
        f"{sum(1 for v in token_df.values() if v > 0.80 * segments):,}"
    )

    print(
        "Empty filtered token sets: "
        f"{sum(not x for x in filtered_sets):,}"
    )

    # --------------------------------------------------------
    # Jaccard cluster at 0.40
    # --------------------------------------------------------

    threshold = 0.40

    print()
    print("=" * 70)
    print(
        "B STRUCTURAL JACCARD CLUSTERING"
    )
    print("=" * 70)

    print(
        f"Similarity threshold: {threshold:.2f}"
    )

    distance = jaccard_matrix(
        filtered_sets
    )

    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - threshold,
        n_clusters=None,
    )

    labels = model.fit_predict(
        distance
    )

    features[
        "structural_cluster"
    ] = labels

    cluster_sizes = (
        features[
            "structural_cluster"
        ]
        .value_counts()
        .rename("segment_count")
    )

    print(
        f"Clusters: "
        f"{len(cluster_sizes):,}"
    )

    print(
        "Cluster size distribution:"
    )

    print(
        cluster_sizes.describe().to_string()
    )

    print(
        f"Largest cluster: "
        f"{cluster_sizes.max():,}"
    )

    print(
        f"Singleton clusters: "
        f"{int((cluster_sizes == 1).sum()):,} "
        f"({(cluster_sizes == 1).mean():.2%})"
    )

    # --------------------------------------------------------
    # Representative tokens per cluster.
    # --------------------------------------------------------

    cluster_rows = []
    cluster_token_rows = []

    for cluster_id, group in features.groupby(
        "structural_cluster",
        sort=False,
    ):

        indices = group.index.tolist()

        token_counter = Counter()

        for index in indices:
            token_counter.update(
                filtered_sets[index]
            )

        top_tokens = [
            token
            for token, _ in token_counter.most_common(
                12
            )
        ]

        cluster_rows.append(
            {
                "structural_cluster":
                    cluster_id,
                "segment_count":
                    len(group),
                "median_duration_seconds":
                    group[
                        "duration_seconds"
                    ].median(),
                "mean_events":
                    group[
                        "event_count"
                    ].mean(),
                "segments_with_form_fields":
                    int(
                        (
                            group[
                                "unique_form_fields"
                            ].astype(float)
                            > 0
                        ).sum()
                    ),
                "segments_with_browser":
                    int(
                        group[
                            "has_browser"
                        ].astype(int).sum()
                    ),
                "segments_with_word":
                    int(
                        group[
                            "has_word"
                        ].astype(int).sum()
                    ),
                "segments_with_excel":
                    int(
                        group[
                            "has_excel"
                        ].astype(int).sum()
                    ),
                "segments_with_notepad":
                    int(
                        group[
                            "has_notepad"
                        ].astype(int).sum()
                    ),
                "top_tokens":
                    " | ".join(
                        top_tokens
                    ),
            }
        )

        for token, count in token_counter.most_common(
            20
        ):
            cluster_token_rows.append(
                {
                    "structural_cluster":
                        cluster_id,
                    "token":
                        token,
                    "segment_frequency_in_cluster":
                        count,
                    "cluster_size":
                        len(group),
                    "cluster_fraction":
                        count / len(group),
                }
            )

    cluster_summary = (
        pd.DataFrame(
            cluster_rows
        )
        .sort_values(
            "segment_count",
            ascending=False,
        )
    )

    cluster_tokens = (
        pd.DataFrame(
            cluster_token_rows
        )
    )

    print()
    print(
        "Largest structural clusters:"
    )

    print(
        cluster_summary.head(
            25
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Operator enrichment
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ADDING OPERATOR IDENTITY"
    )
    print("=" * 70)

    operator_cache = {}

    operator_values = []

    sessions = sorted(
        features[
            "session_id"
        ].astype(str).unique()
    )

    print(
        "Indexing Dataset B session folders once..."
    )

    b_session_index = (
        build_b_session_index()
    )

    print(
        f"Indexed Dataset B sessions: "
        f"{len(b_session_index):,}"
    )

    for i, session_id in enumerate(
        sessions,
        start=1,
    ):

        print(
            f"Operator scan {i}/{len(sessions)}: "
            f"{session_id}"
        )

        events = load_session_operator_events(
            session_id,
            session_index=b_session_index,
        )

        operator_cache[
            session_id
        ] = events

        print(
            f"  events indexed: "
            f"{len(events):,}"
        )

    for _, row in features.iterrows():

        session_id = str(
            row[
                "session_id"
            ]
        )

        start = pd.to_datetime(
            row["start"],
            utc=True,
        )

        end = pd.to_datetime(
            row["end"],
            utc=True,
        )

        hashes = operator_hashes_for_segment(
            operator_cache[
                session_id
            ],
            start,
            end,
        )

        # Keep hashes, but never raw user names.
        operator_values.append(
            " | ".join(
                sorted(
                    hashes
                )
            )
        )

    features[
        "operator_hashes"
    ] = operator_values

    features[
        "operator_count"
    ] = features[
        "operator_hashes"
    ].apply(
        lambda value:
            0
            if not value
            else len(
                [
                    x
                    for x in value.split("|")
                    if x.strip()
                ]
            )
    )

    features.to_csv(
        OUT_OPERATOR_FEATURES,
        index=False,
        encoding="utf-8-sig",
    )

    cluster_summary.to_csv(
        OUT_CLUSTER_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    features[
        [
            "session_id",
            "segment_index",
            "start",
            "end",
            "structural_cluster",
            "operator_hashes",
            "operator_count",
        ]
    ].to_csv(
        OUT_CLUSTER_ASSIGNMENTS,
        index=False,
        encoding="utf-8-sig",
    )

    cluster_tokens.to_csv(
        OUT_CLUSTER_TOKENS,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 70
    )
    print(
        "OPERATOR SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        f"Segments with detected operator: "
        f"{int((features['operator_count'] > 0).sum()):,} "
        f"/ {len(features):,}"
    )

    print(
        f"Distinct operators observed in segments: "
        f"{len(set(
            h
            for value in features['operator_hashes']
            for h in value.split('|')
            if h.strip()
        )):,}"
    )

    print()
    print(
        "Operator count per segment:"
    )

    print(
        features[
            "operator_count"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 70)
    print(
        "PRE-OCR DECISION DATA"
    )
    print("=" * 70)

    print(
        "Interpretation guideline:"
    )
    print(
        "  - If B has rich form fields + many distinct URL/window tokens "
        "and structural clusters are reasonably sized/coherent, OCR is "
        "mainly semantic enrichment."
    )
    print(
        "  - If B structural clusters collapse into a few giant mixed "
        "clusters or field/page evidence is sparse, OCR becomes load-bearing."
    )

    print()
    print(
        "Outputs:"
    )

    print(
        OUT_OPERATOR_FEATURES
    )

    print(
        OUT_CLUSTER_SUMMARY
    )

    print(
        OUT_CLUSTER_ASSIGNMENTS
    )

    print(
        OUT_CLUSTER_TOKENS
    )


if __name__ == "__main__":
    main()
