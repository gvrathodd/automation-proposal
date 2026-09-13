
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
OUTPUTS = ROOT / "outputs"

ENRICHED_FILE = (
    OUTPUTS
    / "labeler_gt_executions_with_signatures.csv"
)

OUT_SUMMARY = (
    OUTPUTS
    / "labeler_content_field_audit_summary.csv"
)

OUT_EVENT_CENSUS = (
    OUTPUTS
    / "labeler_content_field_audit_event_census.csv"
)

OUT_EXAMPLES = (
    OUTPUTS
    / "labeler_content_field_audit_examples.csv"
)


# ============================================================
# Purpose
# ============================================================
#
# This script diagnoses WHY the Dataset-A signature experiment
# produced one giant cluster.
#
# It performs exactly two checks:
#
# CHECK 1:
#   Re-use labeler_gt_executions_with_signatures.csv and measure
#   whether the GT intervals actually captured raw events and
#   whether the extracted feature columns are empty.
#
# CHECK 2:
#   Census raw Dataset A and Dataset B events side-by-side:
#       - layer
#       - event_type
#       - active_app presence
#       - process_name presence
#       - window_title presence
#       - browser URL presence
#       - extracted_text presence
#       - extension connected/disconnected
#       - payload app fields
#       - window-title-change payload
#       - browser navigation payload
#       - browser click/form-input counts
#
# NO MODEL TRAINING.
# NO NEW SIGNATURE DESIGN.
# NO SCREENSHOT OCR.
#
# ============================================================


def parse_json_line(line):
    try:
        return json.loads(line)
    except Exception:
        return None


def nonempty_string(value):
    return (
        isinstance(value, str)
        and value.strip() != ""
    )


def nested_get(mapping, *keys):
    cur = mapping

    for key in keys:
        if not isinstance(cur, dict):
            return None

        cur = cur.get(key)

    return cur


def context_active_app_present(event):
    context = event.get("context") or {}

    return (
        isinstance(
            context.get("active_app"),
            dict,
        )
        and bool(
            context.get("active_app")
        )
    )


def context_window_title_present(event):
    title = nested_get(
        event,
        "context",
        "active_app",
        "window_title",
    )

    return nonempty_string(title)


def context_process_name_present(event):
    process_name = nested_get(
        event,
        "context",
        "active_app",
        "process_name",
    )

    return nonempty_string(process_name)


def context_browser_url_present(event):
    browser = (
        event.get("context") or {}
    ).get(
        "active_browser_tab"
    )

    if not isinstance(
        browser,
        dict,
    ):
        return False

    for key in (
        "url",
        "href",
    ):
        if nonempty_string(
            browser.get(key)
        ):
            return True

    return False


def context_extracted_text_present(event):
    value = nested_get(
        event,
        "context",
        "extracted_text",
    )

    return nonempty_string(value)


def payload_url_present(event):
    payload = event.get(
        "payload"
    ) or {}

    if not isinstance(
        payload,
        dict,
    ):
        return False

    return any(
        nonempty_string(
            payload.get(key)
        )
        for key in (
            "url",
            "current_url",
            "target_url",
            "href",
        )
    )


def payload_window_title_present(event):
    payload = event.get(
        "payload"
    ) or {}

    if not isinstance(
        payload,
        dict,
    ):
        return False

    event_type = str(
        event.get(
            "event_type",
            event.get(
                "event",
                "",
            ),
        )
        or ""
    ).lower()

    if (
        "window_title" not in event_type
        and event_type
        not in {
            "window_title_change",
            "window_state_change",
        }
    ):
        return False

    candidates = [
        payload.get("window_title"),
        payload.get("new_title"),
        payload.get("title"),
        payload.get("new_window_title"),
    ]

    return any(
        nonempty_string(value)
        for value in candidates
    )


def payload_app_fields_present(event):
    payload = event.get(
        "payload"
    ) or {}

    if not isinstance(
        payload,
        dict,
    ):
        return False

    return (
        nonempty_string(
            payload.get("new_app")
        )
        or nonempty_string(
            payload.get("previous_app")
        )
    )


def discover_sessions():
    """
    Returns:
        {
            "A": {session_id: session_folder},
            "B": {session_id: session_folder}
        }
    """

    result = {
        "A": {},
        "B": {},
    }

    # Dataset A is unambiguous because gt_manifest.json exists.
    for manifest in DATA_ROOT.rglob(
        "gt_manifest.json"
    ):
        folder = manifest.parent
        result["A"][folder.name] = folder

    # Dataset B: find folders named dataset_b and then session dirs
    # below them. This works with the extracted Dataset B layout used
    # in the project.
    for dataset_b_root in DATA_ROOT.rglob(
        "dataset_b"
    ):
        for event_file in dataset_b_root.rglob(
            "events.jsonl"
        ):
            session_folder = event_file.parent.parent

            # Expected:
            # dataset_b/<session>/chunk/events.jsonl
            if session_folder.name.startswith(
                "ses_"
            ):
                result["B"][
                    session_folder.name
                ] = session_folder

    return result


def infer_dataset_from_path(path):
    parts = {
        part.lower()
        for part in path.parts
    }

    if "dataset_b" in parts:
        return "B"

    if "dataset_a" in parts:
        return "A"

    return None


def audit_raw_events(
    sessions,
    dataset_label,
):
    overall = Counter()
    examples = []

    event_files = []

    for session_id, session_folder in sorted(
        sessions.items()
    ):
        for event_file in session_folder.rglob(
            "events.jsonl"
        ):
            event_files.append(
                (
                    session_id,
                    event_file,
                )
            )

    total_files = len(event_files)

    print()
    print(
        f"Raw {dataset_label} event files: "
        f"{total_files:,}"
    )

    for file_index, (
        session_id,
        event_file,
    ) in enumerate(
        event_files,
        start=1,
    ):

        print(
            f"\rScanning {dataset_label}: "
            f"{file_index}/{total_files}",
            end="",
            flush=True,
        )

        try:
            with event_file.open(
                "r",
                encoding="utf-8",
            ) as handle:

                for line_no, line in enumerate(
                    handle,
                    start=1,
                ):

                    event = parse_json_line(
                        line
                    )

                    if not isinstance(
                        event,
                        dict,
                    ):
                        continue

                    overall[
                        "events"
                    ] += 1

                    event_type = str(
                        event.get(
                            "event_type",
                            event.get(
                                "event",
                                "",
                            ),
                        )
                        or ""
                    ).strip()

                    layer = str(
                        event.get(
                            "layer",
                            "",
                        )
                        or ""
                    ).strip()

                    overall[
                        f"event_type::{event_type}"
                    ] += 1

                    overall[
                        f"layer::{layer}"
                    ] += 1

                    if context_active_app_present(
                        event
                    ):
                        overall[
                            "context_active_app_nonempty"
                        ] += 1

                    if context_process_name_present(
                        event
                    ):
                        overall[
                            "context_process_name_nonempty"
                        ] += 1

                    if context_window_title_present(
                        event
                    ):
                        overall[
                            "context_window_title_nonempty"
                        ] += 1

                    if context_browser_url_present(
                        event
                    ):
                        overall[
                            "context_browser_url_nonempty"
                        ] += 1

                    if context_extracted_text_present(
                        event
                    ):
                        overall[
                            "context_extracted_text_nonempty"
                        ] += 1

                    if payload_app_fields_present(
                        event
                    ):
                        overall[
                            "payload_app_fields_nonempty"
                        ] += 1

                    if payload_window_title_present(
                        event
                    ):
                        overall[
                            "payload_window_title_nonempty"
                        ] += 1

                    if payload_url_present(
                        event
                    ):
                        overall[
                            "payload_url_nonempty"
                        ] += 1

                    if (
                        event_type
                        == "extension_connected"
                    ):
                        overall[
                            "extension_connected"
                        ] += 1

                    if (
                        event_type
                        == "extension_disconnected"
                    ):
                        overall[
                            "extension_disconnected"
                        ] += 1

                    if (
                        event_type
                        == "browser_navigation"
                    ):
                        overall[
                            "browser_navigation"
                        ] += 1

                    if (
                        event_type
                        == "browser_click"
                    ):
                        overall[
                            "browser_click"
                        ] += 1

                    if (
                        event_type
                        == "browser_form_input"
                    ):
                        overall[
                            "browser_form_input"
                        ] += 1

                    # Preserve a few first useful examples.
                    if len(examples) < 30:

                        useful = (
                            context_active_app_present(
                                event
                            )
                            or context_window_title_present(
                                event
                            )
                            or context_browser_url_present(
                                event
                            )
                            or payload_app_fields_present(
                                event
                            )
                            or payload_window_title_present(
                                event
                            )
                            or payload_url_present(
                                event
                            )
                        )

                        if useful:
                            examples.append(
                                {
                                    "dataset":
                                        dataset_label,
                                    "session_id":
                                        session_id,
                                    "event_file":
                                        str(event_file),
                                    "source_line":
                                        line_no,
                                    "event_type":
                                        event_type,
                                    "layer":
                                        layer,
                                    "context_active_app":
                                        repr(
                                            (
                                                event.get(
                                                    "context"
                                                )
                                                or {}
                                            ).get(
                                                "active_app"
                                            )
                                        ),
                                    "context_browser_tab":
                                        repr(
                                            (
                                                event.get(
                                                    "context"
                                                )
                                                or {}
                                            ).get(
                                                "active_browser_tab"
                                            )
                                        ),
                                    "window_title":
                                        str(
                                            nested_get(
                                                event,
                                                "context",
                                                "active_app",
                                                "window_title",
                                            )
                                            or ""
                                        ),
                                    "extracted_text":
                                        str(
                                            nested_get(
                                                event,
                                                "context",
                                                "extracted_text",
                                            )
                                            or ""
                                        ),
                                    "payload":
                                        repr(
                                            event.get(
                                                "payload"
                                            )
                                        ),
                                }
                            )

        except OSError as exc:
            print(
                f"\nWARNING: {event_file}: {exc}"
            )

    print()

    return overall, examples


# ============================================================
# CHECK 1
# ============================================================

def check_enriched_intervals():
    print()
    print("=" * 70)
    print(
        "CHECK 1 — GT EXECUTION INTERVAL FEATURE POPULATION"
    )
    print("=" * 70)

    if not ENRICHED_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{ENRICHED_FILE}"
        )

    df = pd.read_csv(
        ENRICHED_FILE,
        keep_default_na=False,
    )

    print(
        f"GT execution rows: {len(df):,}"
    )

    for column in [
        "event_count",
        "url_paths",
        "apps",
        "window_titles",
        "form_fields",
    ]:
        if column not in df.columns:
            print(
                f"  MISSING COLUMN: {column}"
            )
            continue

        empty = (
            df[column].isna()
            | (
                df[column].astype(str).str.strip()
                .isin(
                    [
                        "",
                        "()",
                        "[]",
                        "nan",
                        "None",
                    ]
                )
            )
        )

        fraction = (
            empty.mean()
        )

        print(
            f"  {column:20s} "
            f"empty = {empty.sum():4d} / "
            f"{len(df):4d} "
            f"({fraction:.2%})"
        )

    if "event_count" in df.columns:

        counts = pd.to_numeric(
            df["event_count"],
            errors="coerce",
        )

        print()
        print(
            "event_count distribution:"
        )
        print(
            counts.describe().to_string()
        )

        zero = (
            counts.fillna(0)
            == 0
        )

        print(
            f"\nRows with event_count == 0: "
            f"{zero.sum():,} / {len(df):,} "
            f"({zero.mean():.2%})"
        )

    return df


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "DATASET A/B CONTENT FIELD AUDIT"
    )
    print("=" * 70)

    OUTPUTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched = (
        check_enriched_intervals()
    )

    sessions = discover_sessions()

    print()
    print("=" * 70)
    print(
        "SESSION DISCOVERY"
    )
    print("=" * 70)

    print(
        f"Dataset A sessions: "
        f"{len(sessions['A']):,}"
    )

    print(
        f"Dataset B sessions: "
        f"{len(sessions['B']):,}"
    )

    # --------------------------------------------------------
    # Check 2
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CHECK 2 — RAW EVENT CENSUS"
    )
    print("=" * 70)

    audit_results = {}
    examples = []

    for dataset_label in (
        "A",
        "B",
    ):

        result, sample = audit_raw_events(
            sessions[
                dataset_label
            ],
            dataset_label,
        )

        audit_results[
            dataset_label
        ] = result

        examples.extend(
            sample
        )

    # --------------------------------------------------------
    # Build side-by-side summary.
    # --------------------------------------------------------

    all_keys = sorted(
        set(
            key
            for result
            in audit_results.values()
            for key
            in result
        )
    )

    rows = []

    for key in all_keys:

        if key.startswith(
            "event_type::"
        ):
            continue

        if key.startswith(
            "layer::"
        ):
            continue

        a = audit_results[
            "A"
        ].get(
            key,
            0,
        )

        b = audit_results[
            "B"
        ].get(
            key,
            0,
        )

        a_events = audit_results[
            "A"
        ].get(
            "events",
            0,
        )

        b_events = audit_results[
            "B"
        ].get(
            "events",
            0,
        )

        rows.append(
            {
                "metric":
                    key,

                "dataset_a_count":
                    a,

                "dataset_b_count":
                    b,

                "dataset_a_fraction":
                    (
                        a / a_events
                        if a_events
                        else 0.0
                    ),

                "dataset_b_fraction":
                    (
                        b / b_events
                        if b_events
                        else 0.0
                    ),
            }
        )

    summary_df = pd.DataFrame(
        rows
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        examples
    ).to_csv(
        OUT_EXAMPLES,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Print summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "A vs B CONTENT SUMMARY"
    )
    print("=" * 70)

    key_metrics = [
        "events",
        "context_active_app_nonempty",
        "context_process_name_nonempty",
        "context_window_title_nonempty",
        "context_browser_url_nonempty",
        "context_extracted_text_nonempty",
        "payload_app_fields_nonempty",
        "payload_window_title_nonempty",
        "payload_url_nonempty",
        "extension_connected",
        "extension_disconnected",
        "browser_navigation",
        "browser_click",
        "browser_form_input",
    ]

    display = summary_df[
        summary_df[
            "metric"
        ].isin(
            key_metrics
        )
    ].copy()

    if not display.empty:
        print(
            display.to_string(
                index=False
            )
        )

    print()
    print(
        "Layer counts:"
    )

    layer_keys = sorted(
        set(
            key
            for result
            in audit_results.values()
            for key in result
            if key.startswith(
                "layer::"
            )
        )
    )

    layer_rows = []

    for key in layer_keys:

        layer_name = key[
            len("layer::") :
        ]

        layer_rows.append(
            {
                "layer":
                    layer_name,
                "dataset_a":
                    audit_results[
                        "A"
                    ].get(
                        key,
                        0,
                    ),
                "dataset_b":
                    audit_results[
                        "B"
                    ].get(
                        key,
                        0,
                    ),
            }
        )

    if layer_rows:
        print(
            pd.DataFrame(
                layer_rows
            ).to_string(
                index=False
            )
        )

    print()
    print(
        "Top event types:"
    )

    event_keys = sorted(
        set(
            key
            for result
            in audit_results.values()
            for key in result
            if key.startswith(
                "event_type::"
            )
        )
    )

    event_rows = []

    for key in event_keys:

        event_type = key[
            len("event_type::") :
        ]

        event_rows.append(
            {
                "event_type":
                    event_type,
                "dataset_a":
                    audit_results[
                        "A"
                    ].get(
                        key,
                        0,
                    ),
                "dataset_b":
                    audit_results[
                        "B"
                    ].get(
                        key,
                        0,
                    ),
            }
        )

    event_df = pd.DataFrame(
        event_rows
    )

    if not event_df.empty:

        print(
            event_df
            .assign(
                total=lambda x:
                    x["dataset_a"]
                    + x["dataset_b"]
            )
            .sort_values(
                "total",
                ascending=False,
            )
            .drop(
                columns="total"
            )
            .head(40)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Final interpretation helper.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AUDIT INTERPRETATION"
    )
    print("=" * 70)

    a_events = audit_results[
        "A"
    ].get(
        "events",
        0,
    )

    b_events = audit_results[
        "B"
    ].get(
        "events",
        0,
    )

    def frac(dataset, key):
        total = audit_results[
            dataset
        ].get(
            "events",
            0,
        )

        count = audit_results[
            dataset
        ].get(
            key,
            0,
        )

        return (
            count / total
            if total
            else 0
        )

    print(
        f"Dataset A total events: {a_events:,}"
    )

    print(
        f"Dataset B total events: {b_events:,}"
    )

    print()

    for dataset in (
        "A",
        "B",
    ):
        print(
            f"Dataset {dataset}:"
        )

        print(
            "  active_app populated: "
            f"{frac(dataset, 'context_active_app_nonempty'):.2%}"
        )

        print(
            "  process_name populated: "
            f"{frac(dataset, 'context_process_name_nonempty'):.2%}"
        )

        print(
            "  window_title populated: "
            f"{frac(dataset, 'context_window_title_nonempty'):.2%}"
        )

        print(
            "  browser URL populated: "
            f"{frac(dataset, 'context_browser_url_nonempty'):.2%}"
        )

        print(
            "  extracted_text populated: "
            f"{frac(dataset, 'context_extracted_text_nonempty'):.2%}"
        )

        print(
            "  browser_navigation: "
            f"{frac(dataset, 'browser_navigation'):.2%}"
        )

        print(
            "  browser_click: "
            f"{frac(dataset, 'browser_click'):.2%}"
        )

        print(
            "  browser_form_input: "
            f"{frac(dataset, 'browser_form_input'):.2%}"
        )

    print()
    print(
        "Saved:"
    )
    print(
        OUT_SUMMARY
    )
    print(
        OUT_EVENT_CENSUS
    )
    print(
        OUT_EXAMPLES
    )

    print()
    print(
        "Do NOT change the labeler yet."
    )
    print(
        "Use these counts to decide where the content actually lives."
    )


if __name__ == "__main__":
    main()
