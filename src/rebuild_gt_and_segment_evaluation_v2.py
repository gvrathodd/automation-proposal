
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# REBUILD DATASET-A GT FROM gt_manifest + EVALUATE EXISTING
# FINAL STEP-1 PREDICTED BOUNDARIES
#
# Existing prediction file:
#   outputs/segment_model_test_boundaries.csv
#
# Actual prediction schema:
#   session_id
#   predicted_ts
#   probability
#   cluster_size
#
# This script DOES NOT retrain the model.
#
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
OUT = ROOT / "outputs"

PRED_FILE = OUT / "segment_model_test_boundaries.csv"
SPLIT_FILE = OUT / "segment_model_split.csv"

OUT_FRAGMENTS = OUT / "gt_manifest_execution_fragments.csv"
OUT_EXECUTIONS = OUT / "gt_manifest_executions_rebuilt.csv"
OUT_BOUNDARIES = OUT / "gt_manifest_boundaries_rebuilt.csv"
OUT_EVAL = OUT / "segment_level_evaluation.csv"
OUT_MATCHES = OUT / "segment_matches.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def parse_ts(value):
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return pd.NaT


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_execution_records(obj):
    """
    Recursively find execution-like dictionaries.
    """
    found = []

    if isinstance(obj, dict):
        if (
            "start_ts" in obj
            and (
                "process_code" in obj
                or "process_name" in obj
                or "case_id" in obj
            )
        ):
            found.append(obj)

        for value in obj.values():
            found.extend(
                find_execution_records(value)
            )

    elif isinstance(obj, list):
        for item in obj:
            found.extend(
                find_execution_records(item)
            )

    return found


def discover_dataset_a_manifests():
    manifests = sorted(
        DATA_ROOT.rglob("gt_manifest.json")
    )

    result = {}

    for path in manifests:
        result[path.parent.name] = path

    return result


# ------------------------------------------------------------
# Build execution fragments from gt_manifest.json
# ------------------------------------------------------------

def build_fragments():
    rows = []

    manifests = discover_dataset_a_manifests()

    print(
        f"Dataset A sessions with gt_manifest.json: "
        f"{len(manifests):,}"
    )

    for session_id, manifest_path in sorted(
        manifests.items()
    ):
        try:
            data = read_json(manifest_path)
        except Exception as exc:
            print(
                f"WARNING: failed to read {manifest_path}: "
                f"{exc}"
            )
            continue

        records = find_execution_records(data)

        seen = set()

        for record_index, record in enumerate(records):

            start_ts = parse_ts(
                record.get("start_ts")
            )

            end_ts = parse_ts(
                record.get("end_ts")
            )

            row_key = (
                session_id,
                str(
                    record.get(
                        "process_code",
                        "",
                    )
                ),
                str(
                    record.get(
                        "case_id",
                        "",
                    )
                ),
                start_ts,
                end_ts,
                str(
                    record.get(
                        "exec_id",
                        "",
                    )
                ),
            )

            if row_key in seen:
                continue

            seen.add(row_key)

            rows.append(
                {
                    "session_id": session_id,
                    "manifest_path": str(
                        manifest_path
                    ),
                    "manifest_record_index": record_index,
                    "process_code": str(
                        record.get(
                            "process_code",
                            "",
                        )
                        or ""
                    ),
                    "process_name": str(
                        record.get(
                            "process_name",
                            "",
                        )
                        or ""
                    ),
                    "case_id": str(
                        record.get(
                            "case_id",
                            "",
                        )
                        or ""
                    ),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "phase": record.get(
                        "phase"
                    ),
                    "exec_id": str(
                        record.get(
                            "exec_id",
                            "",
                        )
                        or ""
                    ),
                    "split_id": str(
                        record.get(
                            "split_id",
                            "",
                        )
                        or ""
                    ),
                    "continues_from_prev": bool(
                        record.get(
                            "continues_from_prev",
                            False,
                        )
                    ),
                    "continues_to_next": bool(
                        record.get(
                            "continues_to_next",
                            False,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Join explicit continuation fragments
# ------------------------------------------------------------

def build_logical_executions(
    fragments,
):
    if fragments.empty:
        return pd.DataFrame()

    fragments = (
        fragments
        .sort_values(
            [
                "session_id",
                "start_ts",
                "end_ts",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    logical_rows = []

    for session_id, group in fragments.groupby(
        "session_id",
        sort=False,
    ):

        current = []

        def flush_current():
            if not current:
                return

            valid_starts = [
                row["start_ts"]
                for row in current
                if not pd.isna(
                    row["start_ts"]
                )
            ]

            valid_ends = [
                row["end_ts"]
                for row in current
                if not pd.isna(
                    row["end_ts"]
                )
            ]

            if not valid_starts:
                current.clear()
                return

            first = current[0]

            logical_rows.append(
                {
                    "session_id": session_id,
                    "process_code": first[
                        "process_code"
                    ],
                    "process_name": first[
                        "process_name"
                    ],
                    "case_id": first[
                        "case_id"
                    ],
                    "start_ts": min(
                        valid_starts
                    ),
                    "end_ts": (
                        max(valid_ends)
                        if valid_ends
                        else pd.NaT
                    ),
                    "fragment_count": len(
                        current
                    ),
                    "exec_ids": " | ".join(
                        row["exec_id"]
                        for row in current
                        if row["exec_id"]
                    ),
                    "split_ids": " | ".join(
                        row["split_id"]
                        for row in current
                        if row["split_id"]
                    ),
                }
            )

            current.clear()

        for row in group.to_dict("records"):

            if not current:
                current.append(row)
                continue

            previous = current[-1]

            same_process = (
                row["process_code"]
                == previous["process_code"]
            )

            case_compatible = (
                not row["case_id"]
                or not previous["case_id"]
                or (
                    row["case_id"]
                    == previous["case_id"]
                )
            )

            explicit_continuation = (
                previous[
                    "continues_to_next"
                ]
                or row[
                    "continues_from_prev"
                ]
            )

            if (
                same_process
                and case_compatible
                and explicit_continuation
            ):
                current.append(row)
            else:
                flush_current()
                current.append(row)

        flush_current()

    result = pd.DataFrame(logical_rows)

    if result.empty:
        return result

    result[
        "duration_seconds"
    ] = (
        result["end_ts"]
        - result["start_ts"]
    ).dt.total_seconds()

    result[
        "complete_interval"
    ] = (
        result["start_ts"].notna()
        & result["end_ts"].notna()
        & (
            result["duration_seconds"]
            >= 0
        )
    )

    result[
        "logical_execution_id"
    ] = [
        f"{sid}:GTEXEC:{i:05d}"
        for i, sid in enumerate(
            result["session_id"],
            start=1,
        )
    ]

    return result


# ------------------------------------------------------------
# GT boundary reconstruction
# ------------------------------------------------------------

def build_gt_boundaries(logical):
    rows = []

    complete = logical[
        logical["complete_interval"]
    ]

    for _, row in complete.iterrows():

        rows.append(
            {
                "session_id":
                    row["session_id"],
                "ts":
                    row["start_ts"],
                "boundary_type":
                    "execution_start",
                "logical_execution_id":
                    row[
                        "logical_execution_id"
                    ],
            }
        )

        rows.append(
            {
                "session_id":
                    row["session_id"],
                "ts":
                    row["end_ts"],
                "boundary_type":
                    "execution_end",
                "logical_execution_id":
                    row[
                        "logical_execution_id"
                    ],
            }
        )

    gt = pd.DataFrame(rows)

    if gt.empty:
        return gt

    # Collapse only genuinely identical (session, timestamp)
    # boundaries. Do not collapse merely near timestamps.
    gt = (
        gt
        .sort_values(
            [
                "session_id",
                "ts",
                "boundary_type",
            ]
        )
        .drop_duplicates(
            subset=[
                "session_id",
                "ts",
            ]
        )
        .reset_index(drop=True)
    )

    return gt


# ------------------------------------------------------------
# Existing prediction loader
# ------------------------------------------------------------

def load_predictions():
    if not PRED_FILE.exists():
        raise FileNotFoundError(
            PRED_FILE
        )

    pred = pd.read_csv(
        PRED_FILE,
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
            "Prediction file missing columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    pred["predicted_ts"] = pd.to_datetime(
        pred["predicted_ts"],
        utc=True,
    )

    return (
        pred
        .sort_values(
            [
                "session_id",
                "predicted_ts",
            ]
        )
        .reset_index(drop=True)
    )


# ------------------------------------------------------------
# Test sessions from split file
# ------------------------------------------------------------

def get_test_sessions(pred):
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
                    split["split"] == "test"
                ]["session_id"]
            )

            if test_sessions:
                return test_sessions

    return set(
        pred["session_id"]
    )


# ------------------------------------------------------------
# Raw Dataset A session bounds
# ------------------------------------------------------------

def find_session_bounds(
    session_ids,
):
    rows = []

    for event_file in DATA_ROOT.rglob(
        "events.jsonl"
    ):

        current = event_file.parent
        session_id = None

        while (
            current != DATA_ROOT
            and current.parent != current
        ):
            if (
                (
                    current
                    / "gt_manifest.json"
                ).exists()
            ):
                session_id = current.name
                break

            current = current.parent

        if (
            session_id is None
            or session_id not in session_ids
        ):
            continue

        timestamps = []

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

                    timestamp = None

                    if event.get(
                        "timestamp_ms"
                    ) is not None:

                        try:
                            timestamp = pd.to_datetime(
                                int(
                                    event[
                                        "timestamp_ms"
                                    ]
                                ),
                                unit="ms",
                                utc=True,
                            )
                        except Exception:
                            timestamp = None

                    if timestamp is None:

                        for key in (
                            "timestamp_iso",
                            "timestamp",
                        ):

                            if key in event:

                                parsed = (
                                    parse_ts(
                                        event[
                                            key
                                        ]
                                    )
                                )

                                if not pd.isna(
                                    parsed
                                ):
                                    timestamp = parsed
                                    break

                    if timestamp is not None:
                        timestamps.append(
                            timestamp
                        )

        except OSError:
            continue

        if timestamps:

            rows.append(
                {
                    "session_id":
                        session_id,
                    "start":
                        min(timestamps),
                    "end":
                        max(timestamps),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "session_id",
                "start",
                "end",
            ]
        )

    return (
        pd.DataFrame(rows)
        .groupby(
            "session_id",
            as_index=False,
        )
        .agg(
            start=("start", "min"),
            end=("end", "max"),
        )
    )


# ------------------------------------------------------------
# Convert predicted boundary points into segments
# ------------------------------------------------------------

def make_predicted_segments(
    predictions,
    session_bounds,
):
    rows = []

    for session_id, group in predictions.groupby(
        "session_id",
        sort=True,
    ):

        bound = session_bounds[
            session_bounds[
                "session_id"
            ] == session_id
        ]

        if bound.empty:
            continue

        session_start = bound.iloc[0]["start"]
        session_end = bound.iloc[0]["end"]

        timestamps = (
            group[
                "predicted_ts"
            ]
            .dropna()
            .sort_values()
            .tolist()
        )

        points = [
            session_start
        ]

        for timestamp in timestamps:
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

        # Exact timestamp duplicates do not create zero-length segments.
        clean_points = []

        for point in points:
            if (
                not clean_points
                or point != clean_points[-1]
            ):
                clean_points.append(point)

        for index in range(
            len(clean_points) - 1
        ):

            start = clean_points[
                index
            ]
            end = clean_points[
                index + 1
            ]

            if end <= start:
                continue

            rows.append(
                {
                    "session_id":
                        session_id,
                    "segment_index":
                        index,
                    "start":
                        start,
                    "end":
                        end,
                    "duration_seconds":
                        (
                            end - start
                        ).total_seconds(),
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# IoU
# ------------------------------------------------------------

def interval_iou(
    prediction,
    truth,
):
    intersection_start = max(
        prediction["start"],
        truth["start"],
    )

    intersection_end = min(
        prediction["end"],
        truth["end"],
    )

    if (
        intersection_end
        <= intersection_start
    ):
        return 0.0

    intersection = (
        intersection_end
        - intersection_start
    ).total_seconds()

    union_start = min(
        prediction["start"],
        truth["start"],
    )

    union_end = max(
        prediction["end"],
        truth["end"],
    )

    union = (
        union_end
        - union_start
    ).total_seconds()

    if union <= 0:
        return 0.0

    return (
        intersection / union
    )


# ------------------------------------------------------------
# One-to-one greedy IoU matching
# ------------------------------------------------------------

def match_segments(
    predicted,
    truth,
    threshold,
):
    candidates = []

    for prediction_index, prediction in predicted.iterrows():

        for truth_index, target in truth.iterrows():

            if (
                prediction["session_id"]
                != target["session_id"]
            ):
                continue

            score = interval_iou(
                prediction,
                target,
            )

            if score >= threshold:
                candidates.append(
                    (
                        score,
                        prediction_index,
                        truth_index,
                    )
                )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    used_predictions = set()
    used_truth = set()
    matches = []

    for score, prediction_index, truth_index in candidates:

        if prediction_index in used_predictions:
            continue

        if truth_index in used_truth:
            continue

        used_predictions.add(
            prediction_index
        )

        used_truth.add(
            truth_index
        )

        matches.append(
            (
                prediction_index,
                truth_index,
                score,
            )
        )

    unmatched_predictions = (
        set(predicted.index)
        - used_predictions
    )

    unmatched_truth = (
        set(truth.index)
        - used_truth
    )

    return (
        matches,
        unmatched_predictions,
        unmatched_truth,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "GT MANIFEST REBUILD + EXISTING STEP-1 SEGMENT EVALUATION"
    )
    print("=" * 70)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print()
    print(
        "1. Rebuilding Dataset A GT from gt_manifest.json..."
    )

    fragments = build_fragments()

    print(
        f"Execution fragments: "
        f"{len(fragments):,}"
    )

    logical = build_logical_executions(
        fragments
    )

    print(
        f"Logical executions: "
        f"{len(logical):,}"
    )

    complete = logical[
        logical["complete_interval"]
    ].copy()

    incomplete_count = (
        len(logical)
        - len(complete)
    )

    print(
        f"Complete executions: "
        f"{len(complete):,}"
    )

    print(
        f"Incomplete executions: "
        f"{incomplete_count:,}"
    )

    gt_boundaries = (
        build_gt_boundaries(
            logical
        )
    )

    print(
        f"Rebuilt unique GT boundary timestamps: "
        f"{len(gt_boundaries):,}"
    )

    fragments.to_csv(
        OUT_FRAGMENTS,
        index=False,
        encoding="utf-8-sig",
    )

    logical.to_csv(
        OUT_EXECUTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    gt_boundaries.to_csv(
        OUT_BOUNDARIES,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Existing predictions
    # --------------------------------------------------------

    print()
    print(
        "2. Loading existing final Step-1 boundary predictions..."
    )

    predictions = load_predictions()

    print(
        f"Prediction rows: "
        f"{len(predictions):,}"
    )

    print(
        "Prediction columns: "
        + ", ".join(
            predictions.columns
        )
    )

    test_sessions = get_test_sessions(
        predictions
    )

    predictions = predictions[
        predictions[
            "session_id"
        ].isin(
            test_sessions
        )
    ].copy()

    gt_boundaries_test = gt_boundaries[
        gt_boundaries[
            "session_id"
        ].isin(
            test_sessions
        )
    ].copy()

    truth_test = complete[
        complete[
            "session_id"
        ].isin(
            test_sessions
        )
    ][
        [
            "session_id",
            "logical_execution_id",
            "process_code",
            "process_name",
            "case_id",
            "start_ts",
            "end_ts",
            "duration_seconds",
        ]
    ].copy()

    truth_test = truth_test.rename(
        columns={
            "start_ts": "start",
            "end_ts": "end",
        }
    )

    print(
        f"Test sessions: "
        f"{len(test_sessions):,}"
    )

    print(
        f"Predicted test boundaries: "
        f"{len(predictions):,}"
    )

    print(
        f"GT test boundaries: "
        f"{len(gt_boundaries_test):,}"
    )

    print(
        f"GT complete test executions: "
        f"{len(truth_test):,}"
    )

    # --------------------------------------------------------
    # Raw session bounds
    # --------------------------------------------------------

    print()
    print(
        "3. Reconstructing test-session time bounds..."
    )

    bounds = find_session_bounds(
        test_sessions
    )

    predicted_segments = (
        make_predicted_segments(
            predictions,
            bounds,
        )
    )

    print(
        f"Predicted segments: "
        f"{len(predicted_segments):,}"
    )

    print(
        f"GT execution segments: "
        f"{len(truth_test):,}"
    )

    # --------------------------------------------------------
    # Segment-level IoU
    # --------------------------------------------------------

    evaluations = []
    match_rows = []

    for threshold in (
        0.25,
        0.50,
        0.75,
    ):

        (
            matches,
            unmatched_predictions,
            unmatched_truth,
        ) = match_segments(
            predicted_segments,
            truth_test,
            threshold,
        )

        tp = len(matches)
        fp = len(
            unmatched_predictions
        )
        fn = len(
            unmatched_truth
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            )
            else 0.0
        )

        mean_iou = (
            float(
                np.mean(
                    [
                        score
                        for _, _, score
                        in matches
                    ]
                )
            )
            if matches
            else np.nan
        )

        evaluations.append(
            {
                "iou_threshold":
                    threshold,
                "gt_segments":
                    len(truth_test),
                "predicted_segments":
                    len(predicted_segments),
                "matched_segments":
                    tp,
                "unmatched_predicted":
                    fp,
                "unmatched_gt":
                    fn,
                "precision":
                    precision,
                "recall":
                    recall,
                "f1":
                    f1,
                "predicted_to_gt_ratio":
                    (
                        len(
                            predicted_segments
                        )
                        / len(truth_test)
                        if len(truth_test)
                        else np.nan
                    ),
                "mean_matched_iou":
                    mean_iou,
            }
        )

        for prediction_index, truth_index, score in matches:

            prediction = predicted_segments.loc[
                prediction_index
            ]

            truth = truth_test.loc[
                truth_index
            ]

            match_rows.append(
                {
                    "iou_threshold":
                        threshold,
                    "session_id":
                        prediction[
                            "session_id"
                        ],
                    "pred_segment_index":
                        prediction[
                            "segment_index"
                        ],
                    "pred_start":
                        prediction["start"],
                    "pred_end":
                        prediction["end"],
                    "pred_duration_seconds":
                        prediction[
                            "duration_seconds"
                        ],
                    "gt_execution_id":
                        truth[
                            "logical_execution_id"
                        ],
                    "gt_process_code":
                        truth[
                            "process_code"
                        ],
                    "gt_process_name":
                        truth[
                            "process_name"
                        ],
                    "gt_case_id":
                        truth[
                            "case_id"
                        ],
                    "gt_start":
                        truth["start"],
                    "gt_end":
                        truth["end"],
                    "gt_duration_seconds":
                        truth[
                            "duration_seconds"
                        ],
                    "iou":
                        score,
                }
            )

    evaluation_df = pd.DataFrame(
        evaluations
    )

    matches_df = pd.DataFrame(
        match_rows
    )

    evaluation_df.to_csv(
        OUT_EVAL,
        index=False,
        encoding="utf-8-sig",
    )

    matches_df.to_csv(
        OUT_MATCHES,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SEGMENT-LEVEL EVALUATION"
    )
    print("=" * 70)

    print(
        evaluation_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "OUTPUTS"
    )
    print("=" * 70)

    print(
        "Execution fragments:"
    )
    print(
        OUT_FRAGMENTS
    )

    print(
        "\nRebuilt logical executions:"
    )
    print(
        OUT_EXECUTIONS
    )

    print(
        "\nRebuilt GT boundaries:"
    )
    print(
        OUT_BOUNDARIES
    )

    print(
        "\nSegment-level evaluation:"
    )
    print(
        OUT_EVAL
    )

    print(
        "\nSegment matches:"
    )
    print(
        OUT_MATCHES
    )


if __name__ == "__main__":
    main()
