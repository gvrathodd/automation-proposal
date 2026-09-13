
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# DATASET-A PROCESS SIGNATURE LABELER VALIDATION
# ============================================================
#
# Stage 1:
#   Use COMPLETE GT execution intervals from gt_manifest.json.
#   Build deterministic signatures from raw events.
#   Measure:
#       - number of signatures/clusters
#       - weighted purity
#       - macro cluster purity
#       - ARI
#       - fragmentation
#
#   Signature variants are evaluated incrementally:
#       1. URL paths only
#       2. URL paths + app set
#       3. + window-title templates
#       4. + browser field/element attributes
#       5. + clipboard/Office flags
#
# IMPORTANT:
#   case IDs / employee IDs are NEVER included in signatures.
#
# Stage 2:
#   Use EXISTING predicted test-session segments from
#   segment_model_test_boundaries.csv.
#   Match each predicted segment to the GT execution with
#   maximum temporal IoU >= 0.50.
#   Assign the matched GT process_code.
#   Measure signature purity/ARI again.
#
# This script is diagnostic only. It does not modify the
# segmentation model and does not generate Dataset-B labels.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
OUTPUTS = ROOT / "outputs"

PREDICTED_BOUNDARIES = (
    OUTPUTS / "segment_model_test_boundaries.csv"
)

SPLIT_FILE = (
    OUTPUTS / "segment_model_split.csv"
)

OUT_EXECUTIONS = (
    OUTPUTS / "labeler_gt_executions_with_signatures.csv"
)

OUT_STAGE1 = (
    OUTPUTS / "labeler_stage1_signature_results.csv"
)

OUT_STAGE2 = (
    OUTPUTS / "labeler_stage2_signature_results.csv"
)

OUT_STAGE2_MATCHES = (
    OUTPUTS / "labeler_stage2_matches.csv"
)

# ------------------------------------------------------------
# Signature behavior
# ------------------------------------------------------------

URL_KEY = "url"
APP_KEY = "apps"
WINDOW_KEY = "window_titles"
FIELD_KEY = "form_fields"
FLAG_KEY = "flags"


ID_PATTERNS = [
    # employee IDs
    re.compile(r"\bE\d{3,8}\b", re.I),

    # common case/request/reference identifiers
    re.compile(
        r"\b[A-Z]{1,6}-\d{4,12}(?:-\d{1,6})?\b",
        re.I,
    ),

    # numeric-only long identifiers
    re.compile(
        r"\b\d{6,}\b"
    ),
]


def parse_ts(value):
    try:
        return pd.to_datetime(
            value,
            utc=True,
        )
    except Exception:
        return pd.NaT


def read_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


# ============================================================
# GT MANIFEST PARSING
# ============================================================

def discover_gt_manifests():
    manifests = {}

    for path in DATA_ROOT.rglob(
        "gt_manifest.json"
    ):
        manifests[path.parent.name] = path

    return manifests


def extract_manifest_executions(
    manifest_data,
):
    """
    Parse the documented structure:

    processes[
        {
            code,
            family_name,
            domain,
            executions[
                {
                    code,
                    variant,
                    case_id,
                    start_ts,
                    end_ts,
                    ...
                }
            ]
        }
    ]
    """

    rows = []

    processes = manifest_data.get(
        "processes",
        [],
    )

    if not isinstance(
        processes,
        list,
    ):
        return rows

    for process in processes:

        if not isinstance(
            process,
            dict,
        ):
            continue

        process_code = str(
            process.get(
                "code",
                "",
            )
            or ""
        )

        family_name = str(
            process.get(
                "family_name",
                "",
            )
            or ""
        )

        domain = str(
            process.get(
                "domain",
                "",
            )
            or ""
        )

        executions = process.get(
            "executions",
            [],
        )

        if not isinstance(
            executions,
            list,
        ):
            continue

        for execution in executions:

            if not isinstance(
                execution,
                dict,
            ):
                continue

            start = parse_ts(
                execution.get(
                    "start_ts"
                )
            )

            end = parse_ts(
                execution.get(
                    "end_ts"
                )
            )

            code = str(
                execution.get(
                    "code",
                    process_code,
                )
                or process_code
            )

            rows.append(
                {
                    "process_code": code,
                    "process_name": family_name,
                    "domain": domain,
                    "variant": str(
                        execution.get(
                            "variant",
                            "",
                        )
                        or ""
                    ),
                    "case_id": str(
                        execution.get(
                            "case_id",
                            "",
                        )
                        or ""
                    ),
                    "start_ts": start,
                    "end_ts": end,
                    "phase": execution.get(
                        "phase"
                    ),
                    "seq": execution.get(
                        "seq"
                    ),
                    "exec_id": str(
                        execution.get(
                            "exec_id",
                            "",
                        )
                        or ""
                    ),
                    "split_id": str(
                        execution.get(
                            "split_id",
                            "",
                        )
                        or ""
                    ),
                    "continues_from_prev": bool(
                        execution.get(
                            "continues_from_prev",
                            False,
                        )
                    ),
                    "continues_to_next": bool(
                        execution.get(
                            "continues_to_next",
                            False,
                        )
                    ),
                }
            )

    return rows


def load_gt_executions():
    rows = []

    manifests = discover_gt_manifests()

    print(
        f"Dataset A gt_manifest sessions: "
        f"{len(manifests):,}"
    )

    for session_id, manifest_path in sorted(
        manifests.items()
    ):

        try:
            data = read_json(
                manifest_path
            )
        except Exception as exc:
            print(
                f"WARNING: {manifest_path}: {exc}"
            )
            continue

        for row in extract_manifest_executions(
            data
        ):

            row["session_id"] = (
                session_id
            )

            row["manifest_path"] = (
                str(manifest_path)
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df[
        df["start_ts"].notna()
        & df["end_ts"].notna()
    ].copy()

    df[
        "duration_seconds"
    ] = (
        df["end_ts"]
        - df["start_ts"]
    ).dt.total_seconds()

    df = df[
        df["duration_seconds"] >= 0
    ].copy()

    df[
        "gt_execution_id"
    ] = [
        f"{sid}:GT:{i:06d}"
        for i, sid in enumerate(
            df["session_id"],
            start=1,
        )
    ]

    return (
        df.sort_values(
            [
                "session_id",
                "start_ts",
                "end_ts",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# RAW EVENT PARSING
# ============================================================

def event_timestamp(event):
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
    ):

        if key in event:

            timestamp = parse_ts(
                event[key]
            )

            if not pd.isna(
                timestamp
            ):
                return timestamp

    return pd.NaT


def get_app(event):
    context = event.get(
        "context"
    ) or {}

    active_app = context.get(
        "active_app"
    ) or {}

    value = active_app.get(
        "app_name"
    )

    if (
        isinstance(value, str)
        and value.strip()
    ):
        return value.strip().lower()

    return ""


def get_window_title(event):
    context = event.get(
        "context"
    ) or {}

    active_app = context.get(
        "active_app"
    ) or {}

    value = active_app.get(
        "window_title"
    )

    if (
        isinstance(value, str)
        and value.strip()
    ):
        return value.strip()

    return ""


def normalize_window_title(
    title
):
    """
    Strip volatile identifiers/names/numbers while retaining the
    structural title shape.
    """
    if not title:
        return ""

    value = title

    for pattern in ID_PATTERNS:
        value = pattern.sub(
            "{ID}",
            value,
        )

    # Dates/times.
    value = re.sub(
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        "{DATE}",
        value,
    )

    value = re.sub(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        "{TIME}",
        value,
    )

    # Remaining standalone long numbers.
    value = re.sub(
        r"\b\d{3,}\b",
        "{N}",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip().lower()

    return value


def get_browser_url(event):
    context = event.get(
        "context"
    ) or {}

    browser_tab = context.get(
        "active_browser_tab"
    ) or {}

    if isinstance(
        browser_tab,
        dict,
    ):

        for key in (
            "url",
            "href",
        ):

            value = browser_tab.get(
                key
            )

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return value.strip()

    # A few logs may carry URL in payload.
    payload = event.get(
        "payload"
    ) or {}

    for key in (
        "url",
        "current_url",
        "target_url",
    ):

        value = payload.get(
            key
        )

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    return ""


def normalize_url_path(url):
    """
    Keep structural URL paths while stripping identifiers.

    Example:
       /payroll-items/E2008
    ->
       /payroll-items/{id}
    """

    if not url:
        return ""

    value = url.strip()

    # Strip query / fragment.
    value = re.split(
        r"[?#]",
        value,
        maxsplit=1,
    )[0]

    # Extract path-like part.
    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(
            value
        )

        if parsed.path:
            value = parsed.path
    except Exception:
        pass

    value = value.strip()

    for pattern in ID_PATTERNS:
        value = pattern.sub(
            "{id}",
            value,
        )

    # Numeric path components.
    value = re.sub(
        r"/\d{3,}(?=/|$)",
        "/{id}",
        value,
    )

    # Common hashed/opaque long tokens.
    value = re.sub(
        r"/[A-Fa-f0-9]{8,}(?=/|$)",
        "/{id}",
        value,
    )

    value = re.sub(
        r"/+",
        "/",
        value,
    )

    if not value.startswith("/"):
        value = "/" + value

    return value.lower()


def add_nested_strings(
    value,
    output,
):
    """
    Recursively collect useful string attributes.
    This is intentionally conservative.
    """
    if isinstance(
        value,
        dict,
    ):

        for key, child in value.items():

            key_lower = str(
                key
            ).lower()

            if key_lower in {
                "id",
                "name",
                "placeholder",
                "aria-label",
                "aria_label",
                "type",
                "role",
                "name",
                "field",
                "target_field",
            }:

                if isinstance(
                    child,
                    str,
                ):

                    text = child.strip()

                    if text:
                        output.add(
                            f"{key_lower}={text.lower()}"
                        )

            add_nested_strings(
                child,
                output,
            )

    elif isinstance(
        value,
        list,
    ):

        for item in value:
            add_nested_strings(
                item,
                output,
            )


def get_form_field_signature(
    event
):
    """
    Pull structural browser element attributes from browser_click
    and browser_form_input events.
    """
    event_type = str(
        event.get(
            "event_type",
            ""
        )
    ).lower()

    if event_type not in {
        "browser_click",
        "browser_form_input",
    }:
        return []

    payload = event.get(
        "payload"
    ) or {}

    output = set()

    for key in (
        "element",
        "field",
        "target_field",
    ):

        if key in payload:
            add_nested_strings(
                payload[key],
                output,
            )

    context = event.get(
        "context"
    ) or {}

    if "active_browser_tab" in context:
        # Do not use browser text here; only structural attrs.
        pass

    # Keep only field-like strings to avoid exploding the signature.
    filtered = []

    for item in output:

        if any(
            token in item
            for token in (
                "name=",
                "placeholder=",
                "aria-label=",
                "aria_label=",
                "target_field=",
                "id=",
            )
        ):

            filtered.append(
                item
            )

    return filtered


def get_extracted_text(
    event
):
    context = event.get(
        "context"
    ) or {}

    value = context.get(
        "extracted_text"
    )

    if (
        isinstance(value, str)
        and value.strip()
    ):
        return value.strip()

    return ""


def execution_event_features(
    events,
    start,
    end,
):
    """
    Aggregate the raw events falling inside one GT execution.
    """
    url_paths = set()
    apps = set()
    windows = set()
    fields = set()

    clipboard = False
    word = False
    notepad = False
    excel = False

    extracted_text_count = 0

    matching_events = 0

    for event in events:

        timestamp = event_timestamp(
            event
        )

        if pd.isna(timestamp):
            continue

        if timestamp < start:
            continue

        if timestamp > end:
            continue

        matching_events += 1

        app = get_app(event)

        if app:
            apps.add(app)

            if "word" in app:
                word = True

            if "notepad" in app:
                notepad = True

            if "excel" in app:
                excel = True

        title = normalize_window_title(
            get_window_title(event)
        )

        if title:
            windows.add(title)

        url = normalize_url_path(
            get_browser_url(event)
        )

        if url:
            url_paths.add(url)

        fields.update(
            get_form_field_signature(
                event
            )
        )

        event_type = str(
            event.get(
                "event_type",
                ""
            )
        ).lower()

        if event_type == "clipboard_change":
            clipboard = True

        if get_extracted_text(event):
            extracted_text_count += 1

    return {
        "event_count":
            matching_events,

        "url_paths":
            tuple(
                sorted(
                    url_paths
                )
            ),

        "apps":
            tuple(
                sorted(
                    apps
                )
            ),

        "window_titles":
            tuple(
                sorted(
                    windows
                )
            ),

        "form_fields":
            tuple(
                sorted(
                    fields
                )
            ),

        "clipboard":
            clipboard,

        "word":
            word,

        "notepad":
            notepad,

        "excel":
            excel,

        "extracted_text_events":
            extracted_text_count,
    }


# ============================================================
# SESSION RAW EVENT LOADING
# ============================================================

def locate_session_folder(
    session_id
):
    direct = DATA_ROOT / session_id

    if direct.exists():
        return direct

    matches = [
        p
        for p in DATA_ROOT.rglob(
            session_id
        )
        if p.is_dir()
    ]

    if not matches:
        raise FileNotFoundError(
            session_id
        )

    return matches[0]


def load_session_events(
    session_id
):
    session_folder = (
        locate_session_folder(
            session_id
        )
    )

    rows = []

    event_files = sorted(
        session_folder.rglob(
            "events.jsonl"
        )
    )

    for chunk_number, event_file in enumerate(
        event_files,
        start=1,
    ):

        try:
            with event_file.open(
                "r",
                encoding="utf-8",
            ) as handle:

                for source_line, line in enumerate(
                    handle,
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        event = json.loads(
                            line
                        )
                    except Exception:
                        continue

                    # Preserve stable tie-breaking fields externally.
                    rows.append(
                        (
                            event_timestamp(
                                event
                            ),
                            chunk_number,
                            source_line,
                            event,
                        )
                    )

        except OSError:
            continue

    rows = [
        row
        for row in rows
        if not pd.isna(
            row[0]
        )
    ]

    rows.sort(
        key=lambda row: (
            row[0],
            row[1],
            row[2],
        )
    )

    return [
        row[3]
        for row in rows
    ]


# ============================================================
# SIGNATURE BUILDING
# ============================================================

def canonical_signature(
    row,
    components,
):
    parts = []

    if URL_KEY in components:
        parts.append(
            (
                URL_KEY,
                row[
                    "url_paths"
                ],
            )
        )

    if APP_KEY in components:
        parts.append(
            (
                APP_KEY,
                row[
                    "apps"
                ],
            )
        )

    if WINDOW_KEY in components:
        parts.append(
            (
                WINDOW_KEY,
                row[
                    "window_titles"
                ],
            )
        )

    if FIELD_KEY in components:
        parts.append(
            (
                FIELD_KEY,
                row[
                    "form_fields"
                ],
            )
        )

    if FLAG_KEY in components:

        parts.append(
            (
                FLAG_KEY,
                (
                    "clipboard="
                    + str(
                        bool(
                            row[
                                "clipboard"
                            ]
                        )
                    ),
                    "word="
                    + str(
                        bool(
                            row[
                                "word"
                            ]
                        )
                    ),
                    "notepad="
                    + str(
                        bool(
                            row[
                                "notepad"
                            ]
                        )
                    ),
                    "excel="
                    + str(
                        bool(
                            row[
                                "excel"
                            ]
                        )
                    ),
                ),
            )
        )

    return tuple(parts)


def signature_hash(
    signature
):
    return hashlib.sha1(
        repr(
            signature
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:16]


# ============================================================
# PURITY / ARI
# ============================================================

def adjusted_rand_index(
    labels_true,
    labels_pred,
):
    """
    Self-contained ARI implementation.
    """
    from math import comb

    if len(labels_true) != len(labels_pred):
        raise ValueError(
            "Label lengths differ."
        )

    n = len(labels_true)

    if n <= 1:
        return 1.0

    contingency = defaultdict(int)

    true_counts = Counter()
    pred_counts = Counter()

    for true_label, pred_label in zip(
        labels_true,
        labels_pred,
    ):
        contingency[
            (
                true_label,
                pred_label,
            )
        ] += 1

        true_counts[
            true_label
        ] += 1

        pred_counts[
            pred_label
        ] += 1

    sum_comb_cells = sum(
        comb(count, 2)
        for count
        in contingency.values()
        if count >= 2
    )

    sum_comb_true = sum(
        comb(count, 2)
        for count
        in true_counts.values()
        if count >= 2
    )

    sum_comb_pred = sum(
        comb(count, 2)
        for count
        in pred_counts.values()
        if count >= 2
    )

    total_pairs = comb(
        n,
        2,
    )

    expected = (
        sum_comb_true
        * sum_comb_pred
        / total_pairs
        if total_pairs
        else 0.0
    )

    max_index = (
        sum_comb_true
        + sum_comb_pred
    ) / 2

    denominator = (
        max_index
        - expected
    )

    if denominator == 0:
        return 1.0

    return (
        sum_comb_cells
        - expected
    ) / denominator


def evaluate_signatures(
    df,
    components,
    stage,
):
    work = df.copy()

    work[
        "signature"
    ] = work.apply(
        lambda row:
            canonical_signature(
                row,
                components,
            ),
        axis=1,
    )

    work[
        "signature_hash"
    ] = work[
        "signature"
    ].map(
        signature_hash
    )

    grouped = (
        work.groupby(
            "signature_hash"
        )
    )

    cluster_rows = []

    for signature_hash_value, group in grouped:

        counts = Counter(
            group[
                "process_code"
            ]
        )

        dominant_code, dominant_count = (
            counts.most_common(1)[0]
        )

        purity = (
            dominant_count
            / len(group)
        )

        cluster_rows.append(
            {
                "stage":
                    stage,

                "components":
                    "+".join(
                        sorted(
                            components
                        )
                    ),

                "signature_hash":
                    signature_hash_value,

                "cluster_size":
                    len(group),

                "distinct_process_codes":
                    len(counts),

                "dominant_process_code":
                    dominant_code,

                "dominant_process_count":
                    dominant_count,

                "cluster_purity":
                    purity,
            }
        )

    cluster_df = pd.DataFrame(
        cluster_rows
    )

    if cluster_df.empty:
        return (
            cluster_df,
            {
                "stage":
                    stage,
                "components":
                    "+".join(
                        sorted(
                            components
                        )
                    ),
                "executions":
                    0,
                "clusters":
                    0,
                "true_process_codes":
                    0,
                "weighted_purity":
                    np.nan,
                "mean_cluster_purity":
                    np.nan,
                "singleton_cluster_fraction":
                    np.nan,
                "ARI":
                    np.nan,
            },
        )

    weighted_purity = (
        sum(
            row[
                "cluster_size"
            ]
            * row[
                "cluster_purity"
            ]
            for _, row
            in cluster_df.iterrows()
        )
        / len(work)
    )

    mean_cluster_purity = (
        cluster_df[
            "cluster_purity"
        ].mean()
    )

    singleton_fraction = (
        (
            cluster_df[
                "cluster_size"
            ]
            == 1
        ).mean()
    )

    ari = adjusted_rand_index(
        work[
            "process_code"
        ].tolist(),
        work[
            "signature_hash"
        ].tolist(),
    )

    metrics = {
        "stage":
            stage,

        "components":
            "+".join(
                sorted(
                    components
                )
            ),

        "executions":
            len(work),

        "clusters":
            len(cluster_df),

        "true_process_codes":
            work[
                "process_code"
            ].nunique(),

        "weighted_purity":
            weighted_purity,

        "mean_cluster_purity":
            mean_cluster_purity,

        "singleton_cluster_fraction":
            singleton_fraction,

        "ARI":
            ari,
    }

    return (
        cluster_df,
        metrics,
    )


# ============================================================
# RAW EVENTS + GT FEATURES
# ============================================================

def enrich_gt_executions(
    gt_df
):
    records = []

    session_cache = {}

    sessions = sorted(
        gt_df[
            "session_id"
        ].unique()
    )

    for session_index, session_id in enumerate(
        sessions,
        start=1,
    ):

        print(
            f"Loading events for session "
            f"{session_index}/{len(sessions)}: "
            f"{session_id}"
        )

        session_cache[
            session_id
        ] = load_session_events(
            session_id
        )

    for index, row in gt_df.iterrows():

        features = execution_event_features(
            session_cache[
                row[
                    "session_id"
                ]
            ],
            row[
                "start_ts"
            ],
            row[
                "end_ts"
            ],
        )

        record = row.to_dict()
        record.update(
            features
        )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# STAGE 2 — PREDICTED TEST SEGMENTS
# ============================================================

def interval_iou(
    p_start,
    p_end,
    g_start,
    g_end,
):
    start = max(
        p_start,
        g_start,
    )

    end = min(
        p_end,
        g_end,
    )

    if end <= start:
        return 0.0

    intersection = (
        end - start
    ).total_seconds()

    union = (
        max(
            p_end,
            g_end,
        )
        - min(
            p_start,
            g_start,
        )
    ).total_seconds()

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def load_predicted_boundaries():
    if not PREDICTED_BOUNDARIES.exists():
        raise FileNotFoundError(
            PREDICTED_BOUNDARIES
        )

    pred = pd.read_csv(
        PREDICTED_BOUNDARIES,
        keep_default_na=False,
    )

    required = {
        "session_id",
        "predicted_ts",
    }

    missing = (
        required
        - set(
            pred.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Predicted-boundary file missing:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    pred[
        "predicted_ts"
    ] = pd.to_datetime(
        pred[
            "predicted_ts"
        ],
        utc=True,
    )

    return pred.sort_values(
        [
            "session_id",
            "predicted_ts",
        ]
    ).reset_index(
        drop=True
    )


def load_test_sessions():
    if not SPLIT_FILE.exists():
        return None

    split = pd.read_csv(
        SPLIT_FILE,
        keep_default_na=False,
    )

    if not {
        "session_id",
        "split",
    }.issubset(
        split.columns
    ):
        return None

    return set(
        split[
            split[
                "split"
            ]
            == "test"
        ]["session_id"]
    )


def get_session_bounds(
    session_ids
):
    bounds = []

    for session_id in sorted(
        session_ids
    ):

        session_folder = (
            locate_session_folder(
                session_id
            )
        )

        timestamps = []

        for event_file in session_folder.rglob(
            "events.jsonl"
        ):

            try:
                with event_file.open(
                    "r",
                    encoding="utf-8",
                ) as handle:

                    for line in handle:

                        line = line.strip()

                        if not line:
                            continue

                        try:
                            event = json.loads(
                                line
                            )
                        except Exception:
                            continue

                        timestamp = event_timestamp(
                            event
                        )

                        if not pd.isna(
                            timestamp
                        ):
                            timestamps.append(
                                timestamp
                            )

            except OSError:
                continue

        if timestamps:

            bounds.append(
                {
                    "session_id":
                        session_id,
                    "start":
                        min(
                            timestamps
                        ),
                    "end":
                        max(
                            timestamps
                        ),
                }
            )

    return pd.DataFrame(
        bounds
    )


def make_predicted_segments():
    pred = load_predicted_boundaries()

    test_sessions = (
        load_test_sessions()
    )

    if test_sessions is not None:
        pred = pred[
            pred[
                "session_id"
            ].isin(
                test_sessions
            )
        ].copy()

    session_ids = set(
        pred[
            "session_id"
        ]
    )

    bounds = get_session_bounds(
        session_ids
    )

    rows = []

    for session_id, group in pred.groupby(
        "session_id"
    ):

        bound = bounds[
            bounds[
                "session_id"
            ] == session_id
        ]

        if bound.empty:
            continue

        session_start = bound.iloc[0][
            "start"
        ]

        session_end = bound.iloc[0][
            "end"
        ]

        points = [
            session_start
        ]

        for timestamp in (
            group[
                "predicted_ts"
            ]
            .dropna()
            .sort_values()
            .tolist()
        ):

            if (
                timestamp
                > session_start
                and timestamp
                < session_end
            ):
                points.append(
                    timestamp
                )

        points.append(
            session_end
        )

        points = sorted(
            set(
                points
            )
        )

        for index in range(
            len(points) - 1
        ):

            start = points[
                index
            ]

            end = points[
                index + 1
            ]

            if end <= start:
                continue

            rows.append(
                {
                    "session_id":
                        session_id,

                    "pred_segment_index":
                        index,

                    "pred_start":
                        start,

                    "pred_end":
                        end,

                    "pred_duration_seconds":
                        (
                            end - start
                        ).total_seconds(),
                }
            )

    return pd.DataFrame(
        rows
    )


def make_stage2_dataset(
    gt_enriched,
    predicted_segments,
    min_iou=0.50,
):
    rows = []

    for session_id, pred_group in predicted_segments.groupby(
        "session_id"
    ):

        gt_group = gt_enriched[
            gt_enriched[
                "session_id"
            ]
            == session_id
        ]

        for _, prediction in pred_group.iterrows():

            best_iou = 0.0
            best_gt = None

            for _, truth in gt_group.iterrows():

                score = interval_iou(
                    prediction[
                        "pred_start"
                    ],
                    prediction[
                        "pred_end"
                    ],
                    truth[
                        "start_ts"
                    ],
                    truth[
                        "end_ts"
                    ],
                )

                if score > best_iou:
                    best_iou = score
                    best_gt = truth

            if (
                best_gt is None
                or best_iou < min_iou
            ):
                continue

            row = best_gt.to_dict()

            # Replace GT interval timing with prediction timing,
            # but retain GT process code as the validated label.
            row[
                "segment_start"
            ] = prediction[
                "pred_start"
            ]

            row[
                "segment_end"
            ] = prediction[
                "pred_end"
            ]

            row[
                "segment_duration_seconds"
            ] = prediction[
                "pred_duration_seconds"
            ]

            row[
                "pred_segment_index"
            ] = prediction[
                "pred_segment_index"
            ]

            row[
                "matched_iou"
            ] = best_iou

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DATASET-A PROCESS SIGNATURE LABELER VALIDATION"
    )
    print("=" * 70)

    OUTPUTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Stage 1: complete GT executions
    # --------------------------------------------------------

    print()
    print(
        "STAGE 1 — GT EXECUTION SIGNATURES"
    )

    gt = load_gt_executions()

    print(
        f"Complete GT executions: "
        f"{len(gt):,}"
    )

    print(
        f"Distinct process codes: "
        f"{gt['process_code'].nunique():,}"
    )

    enriched = enrich_gt_executions(
        gt
    )

    enriched.to_csv(
        OUT_EXECUTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    signature_variants = [
        (
            "url_only",
            {
                URL_KEY
            },
        ),
        (
            "url_plus_apps",
            {
                URL_KEY,
                APP_KEY,
            },
        ),
        (
            "url_apps_windows",
            {
                URL_KEY,
                APP_KEY,
                WINDOW_KEY,
            },
        ),
        (
            "url_apps_windows_fields",
            {
                URL_KEY,
                APP_KEY,
                WINDOW_KEY,
                FIELD_KEY,
            },
        ),
        (
            "url_apps_windows_fields_flags",
            {
                URL_KEY,
                APP_KEY,
                WINDOW_KEY,
                FIELD_KEY,
                FLAG_KEY,
            },
        ),
    ]

    stage1_rows = []

    for stage_name, components in signature_variants:

        _, metrics = evaluate_signatures(
            enriched,
            components,
            stage=stage_name,
        )

        stage1_rows.append(
            metrics
        )

    stage1_df = pd.DataFrame(
        stage1_rows
    )

    stage1_df.to_csv(
        OUT_STAGE1,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "STAGE 1 RESULTS"
    )
    print(
        stage1_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Stage 2: existing predicted test segments
    # --------------------------------------------------------

    print()
    print(
        "STAGE 2 — EXISTING PREDICTED TEST SEGMENTS"
    )

    predicted_segments = (
        make_predicted_segments()
    )

    print(
        f"Predicted test segments: "
        f"{len(predicted_segments):,}"
    )

    stage2 = make_stage2_dataset(
        enriched,
        predicted_segments,
        min_iou=0.50,
    )

    print(
        f"Predicted segments matched to GT "
        f"at IoU >= 0.50: "
        f"{len(stage2):,}"
    )

    if not stage2.empty:

        _, metrics_url = evaluate_signatures(
            stage2,
            {
                URL_KEY
            },
            stage="stage2_url_only",
        )

        _, metrics_url_apps = evaluate_signatures(
            stage2,
            {
                URL_KEY,
                APP_KEY,
            },
            stage="stage2_url_plus_apps",
        )

        _, metrics_full = evaluate_signatures(
            stage2,
            {
                URL_KEY,
                APP_KEY,
                WINDOW_KEY,
                FIELD_KEY,
                FLAG_KEY,
            },
            stage="stage2_full_signature",
        )

        stage2_results = pd.DataFrame(
            [
                metrics_url,
                metrics_url_apps,
                metrics_full,
            ]
        )

        stage2_results.to_csv(
            OUT_STAGE2,
            index=False,
            encoding="utf-8-sig",
        )

        stage2.to_csv(
            OUT_STAGE2_MATCHES,
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            "STAGE 2 RESULTS"
        )
        print(
            stage2_results.to_string(
                index=False
            )
        )

        print()
        print(
            "Matched predicted-segment coverage:"
        )

        print(
            stage2[
                "matched_iou"
            ].describe().to_string()
        )

    else:

        print(
            "No predicted segments could be matched "
            "to GT executions at IoU >= 0.50."
        )

        pd.DataFrame().to_csv(
            OUT_STAGE2,
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame().to_csv(
            OUT_STAGE2_MATCHES,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LABELER VALIDATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"GT executions with signatures:\n"
        f"{OUT_EXECUTIONS}"
    )

    print(
        f"\nStage 1 signature results:\n"
        f"{OUT_STAGE1}"
    )

    print(
        f"\nStage 2 signature results:\n"
        f"{OUT_STAGE2}"
    )

    print(
        f"\nStage 2 matched segments:\n"
        f"{OUT_STAGE2_MATCHES}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "These are validation diagnostics only. "
        "No Dataset-B labels have been generated."
    )


if __name__ == "__main__":
    main()
