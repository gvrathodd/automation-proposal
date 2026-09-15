
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHUNK_AUDIT = (
    OUTPUT_DIR / "step2_layer1_dataset_b_chunk_audit.csv"
)

EVENT_BY_TYPE = (
    OUTPUT_DIR / "step2_layer2_event_type_layer_counts.csv"
)

SESSION_EVENT = (
    OUTPUT_DIR / "step2_layer2_session_event_counts.csv"
)

CHUNK_EVENT = (
    OUTPUT_DIR / "step2_layer2_chunk_event_counts.csv"
)

CONTENT_SUMMARY = (
    OUTPUT_DIR / "step2_layer2_raw_content_summary.csv"
)

DOM_INVENTORY = (
    OUTPUT_DIR / "step2_layer2_dom_identifier_inventory.csv"
)

URL_INVENTORY = (
    OUTPUT_DIR / "step2_layer2_url_inventory.csv"
)

APP_INVENTORY = (
    OUTPUT_DIR / "step2_layer2_application_inventory.csv"
)

WINDOW_INVENTORY = (
    OUTPUT_DIR / "step2_layer2_window_inventory.csv"
)

OPERATOR_INVENTORY = (
    OUTPUT_DIR / "step2_layer2_operator_inventory.csv"
)

EVENT_EVIDENCE = (
    OUTPUT_DIR / "step2_layer2_event_evidence_by_session.csv"
)


def recursive_values(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from recursive_values(value)

    elif isinstance(obj, list):
        for value in obj:
            yield from recursive_values(value)


def recursive_find(obj, keys):
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                found.append(value)

            found.extend(
                recursive_find(
                    value,
                    keys,
                )
            )

    elif isinstance(obj, list):
        for value in obj:
            found.extend(
                recursive_find(
                    value,
                    keys,
                )
            )

    return found


def flatten_strings(obj):
    values = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                values.append(
                    (
                        str(key),
                        value,
                    )
                )
            values.extend(
                flatten_strings(value)
            )

    elif isinstance(obj, list):
        for value in obj:
            values.extend(
                flatten_strings(value)
            )

    return values


def classify_dom_identifier(value):
    value = str(value or "").strip()

    # IDs used as our process-family evidence.
    prefix = re.search(
        r"(?<![A-Za-z0-9])(pi|la|ob|si|rt)-(?:note|ok)(?![A-Za-z0-9])",
        value,
        flags=re.I,
    )

    if prefix:
        return (
            f"process_prefix:{prefix.group(1).lower()}"
        )

    return None


def extract_dom_strings(event):
    keys = {
        "id",
        "name",
        "placeholder",
        "target_field",
        "field",
        "element",
    }

    values = []

    payload = event.get("payload")

    values.extend(
        recursive_find(
            payload,
            keys,
        )
    )

    values.extend(
        recursive_find(
            event,
            keys,
        )
    )

    strings = []

    def walk(value):
        if isinstance(value, str):
            strings.append(
                value.strip()
            )

        elif isinstance(value, dict):
            for v in value.values():
                walk(v)

        elif isinstance(value, list):
            for v in value:
                walk(v)

    for value in values:
        walk(value)

    return [
        value
        for value in strings
        if value
    ]


def extract_urls(event):
    values = recursive_find(
        event,
        {
            "url",
            "href",
            "current_url",
            "browser_url",
        },
    )

    urls = []

    def walk(value):
        if isinstance(value, str):
            value = value.strip()

            if (
                value.startswith(
                    "http://"
                )
                or value.startswith(
                    "https://"
                )
                or value.startswith(
                    "file://"
                )
                or "/#" in value
            ):
                urls.append(value)

        elif isinstance(value, dict):
            for v in value.values():
                walk(v)

        elif isinstance(value, list):
            for v in value:
                walk(v)

    for value in values:
        walk(value)

    return urls


def extract_apps(event):
    values = recursive_find(
        event,
        {
            "app_name",
            "application",
            "process_name",
            "new_app",
            "previous_app",
        },
    )

    apps = []

    def walk(value):
        if isinstance(value, str):
            value = value.strip()
            if value:
                apps.append(value)

        elif isinstance(value, dict):
            for v in value.values():
                walk(v)

        elif isinstance(value, list):
            for v in value:
                walk(v)

    for value in values:
        walk(value)

    return apps


def extract_windows(event):
    values = recursive_find(
        event,
        {
            "window_title",
            "title",
        },
    )

    windows = []

    def walk(value):
        if isinstance(value, str):
            value = value.strip()
            if value:
                windows.append(value)

        elif isinstance(value, dict):
            for v in value.values():
                walk(v)

        elif isinstance(value, list):
            for v in value:
                walk(v)

    for value in values:
        walk(value)

    return windows


def extract_operator(event):
    values = recursive_find(
        event,
        {
            "username_hash",
            "user_hash",
            "username",
        },
    )

    for value in values:
        if isinstance(
            value,
            (str, int, float),
        ):
            value = str(value).strip()

            if value:
                return value

    return ""


def parse_manifest(path):
    try:
        raw = path.read_text(
            encoding="utf-8"
        ).strip()

        if not raw:
            return {}

        if raw.startswith("{"):
            return json.loads(raw)

        for line in raw.splitlines():
            line = line.strip()

            if line:
                return json.loads(line)

    except Exception:
        return {}

    return {}


def discover_event_files():
    files = []

    # Prefer the authoritative Layer-1 chunk audit because it already
    # established the exact 20 chunks that passed integrity checks.
    if CHUNK_AUDIT.exists():
        audit = pd.read_csv(
            CHUNK_AUDIT,
            keep_default_na=False,
        )

        if "events_file" in audit.columns:
            for value in audit[
                "events_file"
            ]:
                path = Path(
                    str(value)
                )

                if path.exists():
                    files.append(path)

    if files:
        return sorted(
            set(files),
            key=str,
        )

    # Fallback discovery.
    for dataset_b in DATA_ROOT.rglob(
        "dataset_b"
    ):
        if not dataset_b.is_dir():
            continue

        files.extend(
            dataset_b.rglob(
                "events.jsonl"
            )
        )

    return sorted(
        set(files),
        key=str,
    )


def main():
    print("=" * 78)
    print(
        "STEP 2 — LAYER 2 RAW DATASET-B EVENT EVIDENCE"
    )
    print("=" * 78)

    event_files = discover_event_files()

    print(
        f"Event files discovered: {len(event_files)}"
    )

    if len(event_files) != 20:
        print(
            "WARNING: expected 20 event files from Layer 1."
        )

    event_type_layer = Counter()
    session_type = Counter()
    chunk_type = Counter()

    apps = Counter()
    windows = Counter()
    urls = Counter()
    dom_values = Counter()
    dom_prefixes = Counter()
    operators = Counter()

    total_events = 0
    parse_errors = 0

    session_totals = Counter()
    chunk_totals = Counter()

    # Content-presence counters.
    content_counters = Counter()

    session_rows = []
    chunk_rows = []

    for file_number, event_file in enumerate(
        event_files,
        start=1,
    ):
        chunk_dir = event_file.parent
        session_dir = chunk_dir.parent

        session_id = session_dir.name
        chunk_id = chunk_dir.name

        print(
            f"Scanning {file_number}/{len(event_files)}: "
            f"{session_id} / {chunk_id}"
        )

        manifest = {}

        for candidate in (
            chunk_dir / "manifest.json",
            chunk_dir / "manifest.jsonl",
        ):
            if candidate.exists():
                manifest = parse_manifest(
                    candidate
                )
                break

        local_total = 0
        local_parse_errors = 0
        local_type = Counter()
        local_layer = Counter()

        local_apps = set()
        local_windows = set()
        local_urls = set()
        local_dom = set()
        local_operators = set()

        with event_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
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
                    local_parse_errors += 1
                    continue

                total_events += 1
                local_total += 1
                session_totals[
                    session_id
                ] += 1
                chunk_totals[
                    chunk_id
                ] += 1

                event_type = str(
                    event.get(
                        "event_type",
                        "",
                    )
                )
                layer = str(
                    event.get(
                        "layer",
                        "",
                    )
                )

                event_type_layer[
                    (
                        event_type,
                        layer,
                    )
                ] += 1

                session_type[
                    (
                        session_id,
                        event_type,
                        layer,
                    )
                ] += 1

                chunk_type[
                    (
                        session_id,
                        chunk_id,
                        event_type,
                        layer,
                    )
                ] += 1

                local_type[
                    event_type
                ] += 1
                local_layer[
                    layer
                ] += 1

                if event_type in {
                    "browser_click",
                    "browser_form_input",
                    "browser_navigation",
                    "browser_error",
                    "browser_alert",
                    "browser_tab_event",
                    "extension_connected",
                    "extension_disconnected",
                }:
                    content_counters[
                        "browser_related_events"
                    ] += 1

                if event_type == "app_switch":
                    content_counters[
                        "app_switch_events"
                    ] += 1

                if event_type == "window_title_change":
                    content_counters[
                        "window_title_change_events"
                    ] += 1

                if event_type == "clipboard_change":
                    content_counters[
                        "clipboard_events"
                    ] += 1

                if event_type in {
                    "keystroke",
                    "mouse_click",
                    "mouse_scroll",
                    "shortcut",
                    "mouse_double_click",
                    "mouse_drag_drop",
                }:
                    content_counters[
                        "desktop_interaction_events"
                    ] += 1

                if event_type == "screenshot_smart":
                    content_counters[
                        "screenshot_events"
                    ] += 1

                if event_type in {
                    "browser_form_input",
                    "text_input_complete",
                    "keystroke",
                }:
                    content_counters[
                        "form_or_text_input_events"
                    ] += 1

                event_apps = extract_apps(
                    event
                )

                for app in event_apps:
                    apps[
                        app
                    ] += 1
                    local_apps.add(app)

                event_windows = extract_windows(
                    event
                )

                for window in event_windows:
                    windows[
                        window
                    ] += 1
                    local_windows.add(window)

                event_urls = extract_urls(
                    event
                )

                for url in event_urls:
                    urls[
                        url
                    ] += 1
                    local_urls.add(url)

                event_dom = extract_dom_strings(
                    event
                )

                for value in event_dom:
                    dom_values[
                        value
                    ] += 1
                    local_dom.add(value)

                    prefix = classify_dom_identifier(
                        value
                    )

                    if prefix:
                        dom_prefixes[
                            prefix
                        ] += 1

                operator = extract_operator(
                    event
                )

                if operator:
                    operators[
                        operator
                    ] += 1
                    local_operators.add(
                        operator
                    )

        parse_errors += local_parse_errors

        manifest_stats = (
            manifest.get(
                "statistics",
                {},
            )
            if isinstance(
                manifest,
                dict,
            )
            else {}
        )

        manifest_files = (
            manifest.get(
                "files",
                {},
            )
            if isinstance(
                manifest,
                dict,
            )
            else {}
        )

        screenshot_info = (
            manifest_files.get(
                "screenshots",
                {},
            )
            if isinstance(
                manifest_files,
                dict,
            )
            else {}
        )

        session_rows.append(
            {
                "session_id":
                    session_id,
                "chunk_id":
                    chunk_id,
                "events":
                    local_total,
                "unique_event_types":
                    len(
                        local_type
                    ),
                "unique_layers":
                    len(
                        local_layer
                    ),
                "unique_apps_in_chunk":
                    len(
                        local_apps
                    ),
                "unique_windows_in_chunk":
                    len(
                        local_windows
                    ),
                "unique_urls_in_chunk":
                    len(
                        local_urls
                    ),
                "unique_dom_values_in_chunk":
                    len(
                        local_dom
                    ),
                "operators_in_chunk":
                    len(
                        local_operators
                    ),
                "parse_errors":
                    local_parse_errors,
                "manifest_event_count":
                    manifest_stats.get(
                        "total_events"
                    ),
                "manifest_screenshot_count":
                    (
                        screenshot_info.get(
                            "file_count"
                        )
                        if isinstance(
                            screenshot_info,
                            dict,
                        )
                        else None
                    ),
            }
        )

    # Event type/layer table.
    event_rows = []

    for (
        event_type,
        layer,
    ), count in sorted(
        event_type_layer.items(),
        key=lambda x: (
            -x[1],
            x[0][0],
            x[0][1],
        ),
    ):

        event_rows.append(
            {
                "event_type":
                    event_type,
                "layer":
                    layer,
                "count":
                    count,
                "fraction_of_all_events_pct":
                    (
                        count
                        / total_events
                        * 100
                        if total_events
                        else 0
                    ),
            }
        )

    pd.DataFrame(
        event_rows
    ).to_csv(
        EVENT_BY_TYPE,
        index=False,
        encoding="utf-8-sig",
    )

    # Session-by-event-type table.
    rows = []

    for (
        session_id,
        event_type,
        layer,
    ), count in session_type.items():
        rows.append(
            {
                "session_id":
                    session_id,
                "event_type":
                    event_type,
                "layer":
                    layer,
                "count":
                    count,
                "fraction_of_session_events_pct":
                    (
                        count
                        / session_totals[
                            session_id
                        ]
                        * 100
                        if session_totals[
                            session_id
                        ]
                        else 0
                    ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        SESSION_EVENT,
        index=False,
        encoding="utf-8-sig",
    )

    # Chunk-by-event-type table.
    rows = []

    for (
        session_id,
        chunk_id,
        event_type,
        layer,
    ), count in chunk_type.items():

        rows.append(
            {
                "session_id":
                    session_id,
                "chunk_id":
                    chunk_id,
                "event_type":
                    event_type,
                "layer":
                    layer,
                "count":
                    count,
                "fraction_of_chunk_events_pct":
                    (
                        count
                        / chunk_totals[
                            chunk_id
                        ]
                        * 100
                        if chunk_totals[
                            chunk_id
                        ]
                        else 0
                    ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        CHUNK_EVENT,
        index=False,
        encoding="utf-8-sig",
    )

    # Inventories.
    pd.DataFrame(
        [
            {
                "token":
                    token,
                "event_occurrences":
                    count,
            }
            for token, count
            in apps.most_common()
        ]
    ).to_csv(
        APP_INVENTORY,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "window":
                    token,
                "event_occurrences":
                    count,
            }
            for token, count
            in windows.most_common()
        ]
    ).to_csv(
        WINDOW_INVENTORY,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "url":
                    token,
                "event_occurrences":
                    count,
            }
            for token, count
            in urls.most_common()
        ]
    ).to_csv(
        URL_INVENTORY,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "dom_value":
                    token,
                "event_occurrences":
                    count,
                "detected_process_prefix":
                    classify_dom_identifier(
                        token
                    )
                    or "",
            }
            for token, count
            in dom_values.most_common()
        ]
    ).to_csv(
        DOM_INVENTORY,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "operator":
                    token,
                "event_occurrences":
                    count,
            }
            for token, count
            in operators.most_common()
        ]
    ).to_csv(
        OPERATOR_INVENTORY,
        index=False,
        encoding="utf-8-sig",
    )

    # Compact event-evidence summary by session.
    rows = []

    for session_id in sorted(
        session_totals
    ):
        total = session_totals[
            session_id
        ]

        row = {
            "session_id":
                session_id,
            "total_events":
                total,
        }

        for name, counter_key in [
            (
                "browser_related_events",
                "browser_related_events",
            ),
            (
                "app_switch_events",
                "app_switch_events",
            ),
            (
                "window_title_change_events",
                "window_title_change_events",
            ),
            (
                "clipboard_events",
                "clipboard_events",
            ),
            (
                "desktop_interaction_events",
                "desktop_interaction_events",
            ),
            (
                "screenshot_events",
                "screenshot_events",
            ),
            (
                "form_or_text_input_events",
                "form_or_text_input_events",
            ),
        ]:
            # Recalculate per-session from session_type.
            if name == "browser_related_events":
                values = {
                    "browser_click",
                    "browser_form_input",
                    "browser_navigation",
                    "browser_error",
                    "browser_alert",
                    "browser_tab_event",
                    "extension_connected",
                    "extension_disconnected",
                }
            elif name == "app_switch_events":
                values = {"app_switch"}
            elif name == "window_title_change_events":
                values = {"window_title_change"}
            elif name == "clipboard_events":
                values = {"clipboard_change"}
            elif name == "desktop_interaction_events":
                values = {
                    "keystroke",
                    "mouse_click",
                    "mouse_scroll",
                    "shortcut",
                    "mouse_double_click",
                    "mouse_drag_drop",
                }
            elif name == "screenshot_events":
                values = {"screenshot_smart"}
            else:
                values = {
                    "browser_form_input",
                    "text_input_complete",
                    "keystroke",
                }

            count = sum(
                session_type.get(
                    (
                        session_id,
                        event_type,
                        layer,
                    ),
                    0,
                )
                for event_type in values
                for layer in {
                    "L1",
                    "L2",
                    "L3",
                    "SYSTEM",
                    "",
                }
            )

            row[
                name
            ] = count

            row[
                f"{name}_pct"
            ] = (
                count
                / total
                * 100
                if total
                else 0
            )

        rows.append(
            row
        )

    pd.DataFrame(
        rows
    ).to_csv(
        EVENT_EVIDENCE,
        index=False,
        encoding="utf-8-sig",
    )

    # Global content summary.
    summary_rows = []

    for metric, value in [
        (
            "total_events",
            total_events,
        ),
        (
            "sessions",
            len(session_totals),
        ),
        (
            "chunks",
            len(event_files),
        ),
        (
            "parse_errors",
            parse_errors,
        ),
        (
            "distinct_app_strings",
            len(apps),
        ),
        (
            "distinct_window_strings",
            len(windows),
        ),
        (
            "distinct_urls",
            len(urls),
        ),
        (
            "distinct_dom_values",
            len(dom_values),
        ),
        (
            "distinct_detected_process_prefixes",
            len(dom_prefixes),
        ),
        (
            "distinct_operators",
            len(operators),
        ),
        (
            "browser_related_events",
            content_counters[
                "browser_related_events"
            ],
        ),
        (
            "app_switch_events",
            content_counters[
                "app_switch_events"
            ],
        ),
        (
            "window_title_change_events",
            content_counters[
                "window_title_change_events"
            ],
        ),
        (
            "clipboard_events",
            content_counters[
                "clipboard_events"
            ],
        ),
        (
            "desktop_interaction_events",
            content_counters[
                "desktop_interaction_events"
            ],
        ),
        (
            "screenshot_events",
            content_counters[
                "screenshot_events"
            ],
        ),
        (
            "form_or_text_input_events",
            content_counters[
                "form_or_text_input_events"
            ],
        ),
    ]:
        summary_rows.append(
            {
                "metric":
                    metric,
                "value":
                    value,
            }
        )

    for prefix in (
        "process_prefix:pi",
        "process_prefix:la",
        "process_prefix:ob",
        "process_prefix:si",
        "process_prefix:rt",
    ):
        summary_rows.append(
            {
                "metric":
                    f"dom_identifier_{prefix}",
                "value":
                    dom_prefixes[
                        prefix
                    ],
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        CONTENT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        session_rows
    ).to_csv(
        EVENT_EVIDENCE,
        mode="a",
        index=False,
        encoding="utf-8-sig",
        header=False,
    ) if False else None

    print()
    print("=" * 78)
    print(
        "LAYER 2 RESULT"
    )
    print("=" * 78)

    print(
        f"Total events loaded: {total_events:,}"
    )
    print(
        f"Event files: {len(event_files):,}"
    )
    print(
        f"Parse errors: {parse_errors:,}"
    )

    print()
    print(
        "Top event types:"
    )

    top = sorted(
        [
            (
                event_type,
                layer,
                count,
            )
            for (
                event_type,
                layer,
            ), count
            in event_type_layer.items()
        ],
        key=lambda x: -x[2],
    )[:20]

    for (
        event_type,
        layer,
        count,
    ) in top:
        print(
            f"  {event_type:30s}"
            f"{layer:8s}"
            f"{count:7,d}"
            f"  {count / total_events * 100:6.2f}%"
        )

    print()
    print(
        "DOM process-prefix evidence:"
    )

    for prefix in (
        "pi",
        "la",
        "ob",
        "si",
        "rt",
    ):
        print(
            f"  {prefix}: "
            f"{dom_prefixes['process_prefix:' + prefix]:,}"
        )

    print()
    print(
        "Top applications:"
    )

    for token, count in apps.most_common(15):
        print(
            f"  {token:45s}{count:7,d}"
        )

    print()
    print(
        "Top URLs:"
    )

    for token, count in urls.most_common(15):
        print(
            f"  {token:70s}{count:7,d}"
        )

    print()
    print(
        "Top window identities:"
    )

    for token, count in windows.most_common(15):
        print(
            f"  {token[:70]:70s}{count:7,d}"
        )

    print()
    print(
        "Outputs:"
    )

    for path in (
        EVENT_BY_TYPE,
        SESSION_EVENT,
        CHUNK_EVENT,
        CONTENT_SUMMARY,
        DOM_INVENTORY,
        URL_INVENTORY,
        APP_INVENTORY,
        WINDOW_INVENTORY,
        OPERATOR_INVENTORY,
        EVENT_EVIDENCE,
    ):
        print(path)

    print()
    print(
        "Layer 2 complete. No process ranking was performed."
    )


if __name__ == "__main__":
    main()
