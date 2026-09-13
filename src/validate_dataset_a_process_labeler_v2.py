
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
OUTPUTS = ROOT / "outputs"

PREDICTED_BOUNDARIES = OUTPUTS / "segment_model_test_boundaries.csv"
SPLIT_FILE = OUTPUTS / "segment_model_split.csv"

OUT_EXECUTIONS = OUTPUTS / "labeler_gt_executions_with_signatures.csv"
OUT_STAGE1 = OUTPUTS / "labeler_stage1_signature_results.csv"
OUT_STAGE1_CLUSTERS = OUTPUTS / "labeler_stage1_cluster_details.csv"

OUT_STAGE2 = OUTPUTS / "labeler_stage2_signature_results.csv"
OUT_STAGE2_MATCHES = OUTPUTS / "labeler_stage2_matches.csv"


ID_PATTERNS = [
    re.compile(r"\bE\d{3,8}\b", re.I),
    re.compile(r"\b[A-Z]{1,6}-\d{4,12}(?:-\d{1,6})?\b", re.I),
    re.compile(r"\b\d{6,}\b"),
]


# ============================================================
# BASIC HELPERS
# ============================================================

def parse_ts(value):
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return pd.NaT


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_id_noise(value):
    if not isinstance(value, str):
        return ""

    out = value

    for pattern in ID_PATTERNS:
        out = pattern.sub("{ID}", out)

    out = re.sub(
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        "{DATE}",
        out,
    )

    out = re.sub(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        "{TIME}",
        out,
    )

    out = re.sub(
        r"\b\d{3,}\b",
        "{N}",
        out,
    )

    return re.sub(r"\s+", " ", out).strip().lower()


def find_session_folder(session_id):
    direct = DATA_ROOT / session_id

    if direct.exists():
        return direct

    matches = [
        p
        for p in DATA_ROOT.rglob(session_id)
        if p.is_dir()
    ]

    if not matches:
        raise FileNotFoundError(
            f"Dataset A session not found: {session_id}"
        )

    return matches[0]


# ============================================================
# GT MANIFEST
# ============================================================

def discover_gt_manifests():
    result = {}

    for path in DATA_ROOT.rglob("gt_manifest.json"):
        result[path.parent.name] = path

    return result


def extract_gt_executions(manifest):
    rows = []

    for process in manifest.get("processes", []):
        if not isinstance(process, dict):
            continue

        process_code = str(
            process.get("code", "") or ""
        )

        process_name = str(
            process.get("family_name", "") or ""
        )

        domain = str(
            process.get("domain", "") or ""
        )

        for execution in process.get("executions", []):
            if not isinstance(execution, dict):
                continue

            rows.append(
                {
                    "process_code": str(
                        execution.get(
                            "code",
                            process_code,
                        )
                        or process_code
                    ),
                    "process_name": process_name,
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
                    "start_ts": parse_ts(
                        execution.get("start_ts")
                    ),
                    "end_ts": parse_ts(
                        execution.get("end_ts")
                    ),
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
        f"Dataset A gt_manifest sessions: {len(manifests):,}"
    )

    for session_id, manifest_path in sorted(
        manifests.items()
    ):
        try:
            manifest = read_json(
                manifest_path
            )
        except Exception as exc:
            print(
                f"WARNING: {manifest_path}: {exc}"
            )
            continue

        for row in extract_gt_executions(
            manifest
        ):
            row["session_id"] = session_id
            row["manifest_path"] = str(
                manifest_path
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df[
        df["start_ts"].notna()
        & df["end_ts"].notna()
    ].copy()

    df["duration_seconds"] = (
        df["end_ts"] - df["start_ts"]
    ).dt.total_seconds()

    df = df[
        df["duration_seconds"] >= 0
    ].copy()

    df["gt_execution_id"] = [
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


def load_manifest_session_bounds():
    rows = []

    for session_id, manifest_path in sorted(
        discover_gt_manifests().items()
    ):
        try:
            manifest = read_json(
                manifest_path
            )
        except Exception:
            continue

        session = manifest.get(
            "session",
            {}
        )

        if not isinstance(session, dict):
            continue

        start = parse_ts(
            session.get("start_ts")
        )
        end = parse_ts(
            session.get("end_ts")
        )

        if (
            pd.isna(start)
            or pd.isna(end)
        ):
            continue

        rows.append(
            {
                "session_id": session_id,
                "start": start,
                "end": end,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# RAW EVENTS
# ============================================================

def event_timestamp(event):
    if event.get("timestamp_ms") is not None:
        try:
            return pd.to_datetime(
                int(event["timestamp_ms"]),
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
            parsed = parse_ts(
                event[key]
            )
            if not pd.isna(parsed):
                return parsed

    return pd.NaT


def get_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}

    value = active_app.get(
        "app_name"
    )

    return (
        value.strip().lower()
        if isinstance(value, str)
        and value.strip()
        else ""
    )


def get_window_title(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}

    value = active_app.get(
        "window_title"
    )

    return (
        value.strip()
        if isinstance(value, str)
        and value.strip()
        else ""
    )


def normalize_window_title(title):
    return normalize_id_noise(title)


def get_browser_url(event):
    context = event.get("context") or {}
    browser_tab = context.get(
        "active_browser_tab"
    ) or {}

    if isinstance(browser_tab, dict):
        for key in ("url", "href"):
            value = browser_tab.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return value.strip()

    payload = event.get("payload") or {}

    for key in (
        "url",
        "current_url",
        "target_url",
    ):
        value = payload.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    return ""


def normalize_url_path(url):
    if not url:
        return ""

    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        path = parsed.path or ""
    except Exception:
        path = url

    for pattern in ID_PATTERNS:
        path = pattern.sub(
            "{id}",
            path,
        )

    path = re.sub(
        r"/\d{3,}(?=/|$)",
        "/{id}",
        path,
    )

    path = re.sub(
        r"/[A-Fa-f0-9]{8,}(?=/|$)",
        "/{id}",
        path,
    )

    path = re.sub(
        r"/+",
        "/",
        path,
    ).strip()

    if not path.startswith("/"):
        path = "/" + path

    return path.lower()


def add_structural_strings(
    value,
    output,
):
    if isinstance(value, dict):
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

            add_structural_strings(
                child,
                output,
            )

    elif isinstance(value, list):
        for child in value:
            add_structural_strings(
                child,
                output,
            )


def get_form_fields(event):
    event_type = str(
        event.get(
            "event_type",
            "",
        )
    ).lower()

    if event_type not in {
        "browser_click",
        "browser_form_input",
    }:
        return []

    payload = event.get("payload") or {}

    found = set()

    for key in (
        "element",
        "field",
        "target_field",
    ):
        if key in payload:
            add_structural_strings(
                payload[key],
                found,
            )

    return sorted(
        item
        for item in found
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
        )
    )


def load_session_events(
    session_id
):
    folder = find_session_folder(
        session_id
    )

    events = []

    for event_file in sorted(
        folder.rglob("events.jsonl")
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

                    if pd.isna(
                        timestamp
                    ):
                        continue

                    events.append(
                        {
                            "timestamp":
                                timestamp,
                            "event":
                                event,
                        }
                    )

        except OSError:
            continue

    events.sort(
        key=lambda x: x["timestamp"]
    )

    return events


def aggregate_features(
    events,
    start,
    end,
):
    url_paths = set()
    apps = set()
    window_titles = set()
    form_fields = set()

    clipboard = False
    word = False
    notepad = False
    excel = False

    event_count = 0

    for item in events:

        timestamp = item["timestamp"]

        if (
            timestamp < start
            or timestamp > end
        ):
            continue

        event = item["event"]

        event_count += 1

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
            window_titles.add(title)

        url_path = normalize_url_path(
            get_browser_url(event)
        )

        if url_path:
            url_paths.add(url_path)

        form_fields.update(
            get_form_fields(event)
        )

        event_type = str(
            event.get(
                "event_type",
                "",
            )
        ).lower()

        if event_type == "clipboard_change":
            clipboard = True

    return {
        "event_count": event_count,
        "url_paths": tuple(
            sorted(url_paths)
        ),
        "apps": tuple(
            sorted(apps)
        ),
        "window_titles": tuple(
            sorted(window_titles)
        ),
        "form_fields": tuple(
            sorted(form_fields)
        ),
        "clipboard": clipboard,
        "word": word,
        "notepad": notepad,
        "excel": excel,
    }


# ============================================================
# SIGNATURE / METRICS
# ============================================================

def make_signature(row, components):
    parts = []

    if "url" in components:
        parts.append(
            ("url", row["url_paths"])
        )

    if "apps" in components:
        parts.append(
            ("apps", row["apps"])
        )

    if "windows" in components:
        parts.append(
            (
                "windows",
                row["window_titles"],
            )
        )

    if "fields" in components:
        parts.append(
            (
                "fields",
                row["form_fields"],
            )
        )

    if "flags" in components:
        parts.append(
            (
                "flags",
                (
                    "clipboard=" + str(
                        bool(
                            row[
                                "clipboard"
                            ]
                        )
                    ),
                    "word=" + str(
                        bool(
                            row[
                                "word"
                            ]
                        )
                    ),
                    "notepad=" + str(
                        bool(
                            row[
                                "notepad"
                            ]
                        )
                    ),
                    "excel=" + str(
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


def hash_signature(signature):
    return hashlib.sha1(
        repr(signature).encode(
            "utf-8"
        )
    ).hexdigest()[:16]


def ari_score(
    true_labels,
    predicted_labels,
):
    from math import comb

    n = len(true_labels)

    if n <= 1:
        return 1.0

    contingency = defaultdict(int)
    true_counts = Counter()
    pred_counts = Counter()

    for t, p in zip(
        true_labels,
        predicted_labels,
    ):
        contingency[
            (t, p)
        ] += 1
        true_counts[t] += 1
        pred_counts[p] += 1

    sum_cells = sum(
        comb(v, 2)
        for v in contingency.values()
        if v >= 2
    )

    sum_true = sum(
        comb(v, 2)
        for v in true_counts.values()
        if v >= 2
    )

    sum_pred = sum(
        comb(v, 2)
        for v in pred_counts.values()
        if v >= 2
    )

    total_pairs = comb(n, 2)

    expected = (
        sum_true * sum_pred / total_pairs
        if total_pairs
        else 0.0
    )

    max_index = (
        sum_true + sum_pred
    ) / 2

    denominator = (
        max_index - expected
    )

    if denominator == 0:
        return 1.0

    return (
        sum_cells - expected
    ) / denominator


def evaluate_signature(
    df,
    components,
    stage,
):
    work = df.copy()

    work["signature"] = work.apply(
        lambda row:
            make_signature(
                row,
                components,
            ),
        axis=1,
    )

    work["signature_hash"] = (
        work["signature"]
        .map(hash_signature)
    )

    cluster_rows = []

    for signature_hash, group in work.groupby(
        "signature_hash"
    ):
        counts = Counter(
            group["process_code"]
        )

        dominant_code, dominant_count = (
            counts.most_common(1)[0]
        )

        cluster_rows.append(
            {
                "stage": stage,
                "components": "+".join(
                    sorted(components)
                ),
                "signature_hash": signature_hash,
                "cluster_size": len(group),
                "distinct_process_codes": len(counts),
                "dominant_process_code":
                    dominant_code,
                "dominant_process_count":
                    dominant_count,
                "cluster_purity":
                    dominant_count / len(group),
            }
        )

    clusters = pd.DataFrame(
        cluster_rows
    )

    if clusters.empty:
        return (
            clusters,
            {
                "stage": stage,
                "components": "+".join(
                    sorted(components)
                ),
                "executions": 0,
                "clusters": 0,
                "true_process_codes": 0,
                "weighted_purity": np.nan,
                "mean_cluster_purity": np.nan,
                "largest_cluster_size": 0,
                "top5_cluster_fraction": np.nan,
                "singleton_cluster_fraction": np.nan,
                "ARI": np.nan,
            },
        )

    weighted_purity = (
        (
            clusters[
                "cluster_size"
            ]
            * clusters[
                "cluster_purity"
            ]
        ).sum()
        / len(work)
    )

    largest_cluster = int(
        clusters[
            "cluster_size"
        ].max()
    )

    top5 = (
        clusters
        .sort_values(
            "cluster_size",
            ascending=False,
        )
        .head(5)
    )

    top5_fraction = (
        top5["cluster_size"].sum()
        / len(work)
    )

    metrics = {
        "stage": stage,
        "components": "+".join(
            sorted(components)
        ),
        "executions": len(work),
        "clusters": len(clusters),
        "true_process_codes":
            work["process_code"].nunique(),
        "weighted_purity":
            weighted_purity,
        "mean_cluster_purity":
            clusters[
                "cluster_purity"
            ].mean(),
        "largest_cluster_size":
            largest_cluster,
        "top5_cluster_fraction":
            top5_fraction,
        "singleton_cluster_fraction":
            (
                clusters[
                    "cluster_size"
                ] == 1
            ).mean(),
        "ARI":
            ari_score(
                work[
                    "process_code"
                ].tolist(),
                work[
                    "signature_hash"
                ].tolist(),
            ),
    }

    return clusters, metrics


# ============================================================
# STAGE 1
# ============================================================

def run_stage1(gt):
    cache = {}
    rows = []

    sessions = sorted(
        gt["session_id"].unique()
    )

    for i, session_id in enumerate(
        sessions,
        start=1,
    ):
        print(
            f"Loading events for session "
            f"{i}/{len(sessions)}: "
            f"{session_id}"
        )

        cache[session_id] = (
            load_session_events(
                session_id
            )
        )

    enriched_rows = []

    for _, row in gt.iterrows():

        features = aggregate_features(
            cache[
                row["session_id"]
            ],
            row["start_ts"],
            row["end_ts"],
        )

        item = row.to_dict()
        item.update(features)
        enriched_rows.append(item)

    enriched = pd.DataFrame(
        enriched_rows
    )

    enriched.to_csv(
        OUT_EXECUTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    variants = [
        (
            "url_only",
            {"url"},
        ),
        (
            "url_plus_apps",
            {"url", "apps"},
        ),
        (
            "url_apps_windows",
            {
                "url",
                "apps",
                "windows",
            },
        ),
        (
            "url_apps_windows_fields",
            {
                "url",
                "apps",
                "windows",
                "fields",
            },
        ),
        (
            "full_signature",
            {
                "url",
                "apps",
                "windows",
                "fields",
                "flags",
            },
        ),
    ]

    metrics = []
    cluster_details = []

    for name, components in variants:

        clusters, result = evaluate_signature(
            enriched,
            components,
            name,
        )

        metrics.append(
            result
        )

        if not clusters.empty:
            cluster_details.append(
                clusters
            )

    stage1_df = pd.DataFrame(
        metrics
    )

    stage1_df.to_csv(
        OUT_STAGE1,
        index=False,
        encoding="utf-8-sig",
    )

    if cluster_details:
        pd.concat(
            cluster_details,
            ignore_index=True,
        ).sort_values(
            [
                "stage",
                "cluster_size",
            ],
            ascending=[
                True,
                False,
            ],
        ).to_csv(
            OUT_STAGE1_CLUSTERS,
            index=False,
            encoding="utf-8-sig",
        )

    return enriched


# ============================================================
# STAGE 2
# ============================================================

def load_predictions():
    pred = pd.read_csv(
        PREDICTED_BOUNDARIES,
        keep_default_na=False,
    )

    required = {
        "session_id",
        "predicted_ts",
    }

    missing = required - set(
        pred.columns
    )

    if missing:
        raise RuntimeError(
            "Prediction file missing:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    pred["predicted_ts"] = (
        pd.to_datetime(
            pred["predicted_ts"],
            utc=True,
        )
    )

    return pred


def make_predicted_segments():
    pred = load_predictions()

    test_sessions = None

    if SPLIT_FILE.exists():

        split = pd.read_csv(
            SPLIT_FILE,
            keep_default_na=False,
        )

        if {
            "session_id",
            "split",
        }.issubset(
            split.columns
        ):

            test_sessions = set(
                split[
                    split[
                        "split"
                    ]
                    == "test"
                ]["session_id"]
            )

    if test_sessions is not None:
        pred = pred[
            pred[
                "session_id"
            ].isin(
                test_sessions
            )
        ].copy()

    session_bounds = (
        load_manifest_session_bounds()
    )

    rows = []

    for session_id, group in pred.groupby(
        "session_id"
    ):

        bounds = session_bounds[
            session_bounds[
                "session_id"
            ] == session_id
        ]

        if bounds.empty:
            print(
                f"WARNING: no manifest session "
                f"bounds for {session_id}"
            )
            continue

        session_start = bounds.iloc[0][
            "start"
        ]

        session_end = bounds.iloc[0][
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
                timestamp > session_start
                and timestamp < session_end
            ):
                points.append(
                    timestamp
                )

        points.append(
            session_end
        )

        points = sorted(
            set(points)
        )

        for index in range(
            len(points) - 1
        ):

            start = points[index]
            end = points[index + 1]

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

    return pd.DataFrame(rows)


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
        max(p_end, g_end)
        - min(p_start, g_start)
    ).total_seconds()

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def build_stage2_dataset(
    enriched,
    predicted_segments,
):
    rows = []

    for session_id, pred_group in predicted_segments.groupby(
        "session_id"
    ):

        gt_group = enriched[
            enriched[
                "session_id"
            ] == session_id
        ]

        for _, prediction in pred_group.iterrows():

            best_score = 0.0
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

                if score > best_score:
                    best_score = score
                    best_gt = truth

            if (
                best_gt is None
                or best_score < 0.50
            ):
                continue

            row = best_gt.to_dict()

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
            ] = best_score

            rows.append(row)

    return pd.DataFrame(rows)


def run_stage2(enriched):
    predicted = make_predicted_segments()

    print(
        f"Predicted test segments: "
        f"{len(predicted):,}"
    )

    matched = build_stage2_dataset(
        enriched,
        predicted,
    )

    print(
        f"Predicted segments matched "
        f"at IoU >= 0.50: "
        f"{len(matched):,}"
    )

    if matched.empty:
        print(
            "No Stage-2 matches available."
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

        return

    variants = [
        (
            "stage2_url_only",
            {"url"},
        ),
        (
            "stage2_url_plus_apps",
            {
                "url",
                "apps",
            },
        ),
        (
            "stage2_full_signature",
            {
                "url",
                "apps",
                "windows",
                "fields",
                "flags",
            },
        ),
    ]

    metrics = []

    for name, components in variants:
        _, result = evaluate_signature(
            matched,
            components,
            name,
        )

        metrics.append(
            result
        )

    stage2_df = pd.DataFrame(
        metrics
    )

    stage2_df.to_csv(
        OUT_STAGE2,
        index=False,
        encoding="utf-8-sig",
    )

    matched.to_csv(
        OUT_STAGE2_MATCHES,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "STAGE 2 RESULTS"
    )
    print(
        stage2_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Matched IoU distribution:"
    )

    print(
        matched[
            "matched_iou"
        ].describe().to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print(
        "DATASET-A PROCESS SIGNATURE LABELER VALIDATION V2"
    )
    print("=" * 70)

    OUTPUTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "STAGE 1 — COMPLETE GT EXECUTIONS"
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

    enriched = run_stage1(
        gt
    )

    print()
    print(
        "STAGE 1 RESULTS"
    )

    print(
        pd.read_csv(
            OUT_STAGE1
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "STAGE 2 — EXISTING PREDICTED TEST SEGMENTS"
    )

    run_stage2(
        enriched
    )

    print()
    print("=" * 70)
    print(
        "LABELER VALIDATION V2 COMPLETE"
    )
    print("=" * 70)

    print(
        f"GT execution signatures:\n"
        f"{OUT_EXECUTIONS}"
    )

    print(
        f"\nStage 1 metrics:\n"
        f"{OUT_STAGE1}"
    )

    print(
        f"\nStage 1 cluster details:\n"
        f"{OUT_STAGE1_CLUSTERS}"
    )

    print(
        f"\nStage 2 metrics:\n"
        f"{OUT_STAGE2}"
    )

    print(
        f"\nStage 2 matches:\n"
        f"{OUT_STAGE2_MATCHES}"
    )


if __name__ == "__main__":
    main()
