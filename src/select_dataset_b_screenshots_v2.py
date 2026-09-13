
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import shutil

import pandas as pd

from loader import load_all_events, sort_session_events, get_timestamp_ms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "raw_data" / "dataset-downloads"
INVENTORY_FILE = PROJECT_ROOT / "outputs" / "dataset_b_segment_inventory.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SELECTION_FILE = OUTPUT_DIR / "dataset_b_screenshot_selection_v2.csv"
SELECTED_DIR = OUTPUT_DIR / "dataset_b_selected_screenshots_v2"

MAX_SELECTED_PER_SEGMENT = 5
EVENT_NEIGHBOR_SECONDS = 1.5
MIN_SCREENSHOT_SEPARATION_SECONDS = 1.0

HIGH_INFORMATION_EVENTS = {
    "app_switch",
    "browser_navigation",
    "browser_form_input",
    "browser_click",
    "clipboard_change",
    "mouse_click",
    "shortcut",
    "text_input_complete",
    "window_title_change",
    "window_state_change",
    "dialog_opened",
    "dialog_closed",
    "upload_started",
    "upload_completed",
}

MEDIUM_INFORMATION_EVENTS = {
    "keystroke",
    "mouse_scroll",
    "browser_alert",
    "browser_tab_event",
}


def get_event_type(event):
    value = event.get("event_type")
    return value if isinstance(value, str) else ""


def get_event_app(event):
    context = event.get("context") or {}
    active_app = context.get("active_app") or {}
    value = active_app.get("app_name")
    return value.strip() if isinstance(value, str) and value.strip() else "UNKNOWN"


def extract_screenshot_filenames(event):
    payload = event.get("payload") or {}
    ref = payload.get("file_reference")
    if not ref:
        return []

    result = []
    if isinstance(ref, dict):
        filename = ref.get("filename")
        if isinstance(filename, str) and filename.strip():
            result.append(filename.strip())
    elif isinstance(ref, list):
        for item in ref:
            if isinstance(item, dict):
                filename = item.get("filename")
                if isinstance(filename, str) and filename.strip():
                    result.append(filename.strip())
    return result


def prepare_events(events):
    prepared = []

    for event in sort_session_events(events):
        timestamp_ms = get_timestamp_ms(event)
        if timestamp_ms is None:
            continue

        prepared.append(
            {
                "timestamp_ms": int(timestamp_ms),
                "timestamp": pd.to_datetime(
                    int(timestamp_ms), unit="ms", utc=True
                ),
                "event_type": get_event_type(event),
                "app": get_event_app(event),
                "screenshots": extract_screenshot_filenames(event),
            }
        )

    return prepared


def build_screenshot_index():
    print("Indexing screenshot files...")

    index = defaultdict(list)

    for path in DATA_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp"
        }:
            index[path.name].append(path)

    print(f"Indexed unique screenshot filenames: {len(index):,}")
    return index


def event_weight(event_type):
    if event_type in HIGH_INFORMATION_EVENTS:
        return 5.0
    if event_type in MEDIUM_INFORMATION_EVENTS:
        return 2.0
    return 0.0


def collect_segment_candidates(prepared_events, start, end):
    """
    Collect screenshots referenced inside the segment.

    For every screenshot, calculate:
      - nearest high-information event
      - nearest app transition
      - event-proximity score
      - temporal position

    This deliberately avoids treating all screenshots equally.
    """
    candidates = {}

    segment_start_ms = int(start.timestamp() * 1000)
    segment_end_ms = int(end.timestamp() * 1000)

    # First gather app transitions and information-rich events.
    transitions = []
    meaningful_events = []

    previous_app = None

    for event in prepared_events:
        ts = event["timestamp"]
        ts_ms = event["timestamp_ms"]

        if not (
            segment_start_ms <= ts_ms <= segment_end_ms
        ):
            continue

        app = event["app"]

        if (
            previous_app is not None
            and app != previous_app
            and app != "UNKNOWN"
            and previous_app != "UNKNOWN"
        ):
            transitions.append(
                {
                    "timestamp": ts,
                    "from_app": previous_app,
                    "to_app": app,
                }
            )

        if event["event_type"] in (
            HIGH_INFORMATION_EVENTS
            | MEDIUM_INFORMATION_EVENTS
        ):
            meaningful_events.append(event)

        if app != "UNKNOWN":
            previous_app = app

    # Then collect screenshots.
    for event in prepared_events:
        ts = event["timestamp"]

        if not (
            start <= ts <= end
        ):
            continue

        for filename in event["screenshots"]:
            if filename in candidates:
                continue

            best_transition = None
            best_transition_distance = None

            for transition in transitions:
                distance = abs(
                    (
                        ts - transition["timestamp"]
                    ).total_seconds()
                )

                if (
                    best_transition_distance is None
                    or distance < best_transition_distance
                ):
                    best_transition_distance = distance
                    best_transition = transition

            best_event = None
            best_event_distance = None

            for meaningful in meaningful_events:
                distance = abs(
                    (
                        ts - meaningful["timestamp"]
                    ).total_seconds()
                )

                if (
                    best_event_distance is None
                    or distance < best_event_distance
                ):
                    best_event_distance = distance
                    best_event = meaningful

            score = 1.0
            reasons = []

            if (
                best_transition is not None
                and best_transition_distance <= EVENT_NEIGHBOR_SECONDS
            ):
                score += 12.0
                reasons.append(
                    "near app transition "
                    f"{best_transition['from_app']} -> "
                    f"{best_transition['to_app']}"
                )

            if (
                best_event is not None
                and best_event_distance <= EVENT_NEIGHBOR_SECONDS
            ):
                score += (
                    event_weight(
                        best_event["event_type"]
                    )
                )
                reasons.append(
                    "near "
                    f"{best_event['event_type']}"
                )

            duration = max(
                (end - start).total_seconds(), 0.001
            )

            relative_position = (
                ts - start
            ).total_seconds() / duration

            # Mild position bonus only. This is intentionally
            # weaker than event/transition evidence.
            if 0.0 <= relative_position <= 0.20:
                score += 0.5
                reasons.append("early coverage")
            elif 0.40 <= relative_position <= 0.60:
                score += 0.5
                reasons.append("middle coverage")
            elif 0.80 <= relative_position <= 1.0:
                score += 0.5
                reasons.append("late coverage")

            candidates[filename] = {
                "filename": filename,
                "timestamp": ts,
                "app": event["app"],
                "source_event_type": event["event_type"],
                "score": score,
                "reason": "; ".join(reasons) if reasons else "baseline screenshot",
                "nearest_transition": (
                    (
                        best_transition["from_app"]
                        + " -> "
                        + best_transition["to_app"]
                    )
                    if best_transition is not None
                    else ""
                ),
                "nearest_transition_distance_seconds": (
                    round(best_transition_distance, 3)
                    if best_transition_distance is not None
                    else None
                ),
                "nearest_event": (
                    best_event["event_type"]
                    if best_event is not None
                    else ""
                ),
                "nearest_event_distance_seconds": (
                    round(best_event_distance, 3)
                    if best_event_distance is not None
                    else None
                ),
            }

    return list(candidates.values())


def select_candidates(candidates):
    if not candidates:
        return []

    # Highest-value first. Timestamp breaks ties deterministically.
    ranked = sorted(
        candidates,
        key=lambda x: (-x["score"], x["timestamp"])
    )

    selected = []

    def far_enough(candidate):
        return all(
            abs(
                (
                    candidate["timestamp"]
                    - existing["timestamp"]
                ).total_seconds()
            )
            >= MIN_SCREENSHOT_SEPARATION_SECONDS
            for existing in selected
        )

    # --------------------------------------------------------
    # PASS 1: transition-anchored screenshots.
    # One before/at transition and one after where possible.
    # --------------------------------------------------------

    transition_candidates = [
        x
        for x in ranked
        if x["nearest_transition"]
        and x["nearest_transition_distance_seconds"] is not None
        and x["nearest_transition_distance_seconds"]
        <= EVENT_NEIGHBOR_SECONDS
    ]

    for candidate in transition_candidates:
        if len(selected) >= MAX_SELECTED_PER_SEGMENT:
            break

        if far_enough(candidate):
            selected.append(
                {
                    **candidate,
                    "selection_priority":
                        "P1 transition",
                    "selection_reason":
                        (
                            "Highest priority: "
                            + candidate["reason"]
                        ),
                }
            )

    # --------------------------------------------------------
    # PASS 2: other high-information event anchors.
    # --------------------------------------------------------

    if len(selected) < MAX_SELECTED_PER_SEGMENT:

        event_candidates = [
            x
            for x in ranked
            if (
                x not in selected
                and x["nearest_event"]
                and x["nearest_event_distance_seconds"]
                is not None
                and x["nearest_event_distance_seconds"]
                <= EVENT_NEIGHBOR_SECONDS
            )
        ]

        for candidate in event_candidates:

            if len(selected) >= MAX_SELECTED_PER_SEGMENT:
                break

            if far_enough(candidate):
                selected.append(
                    {
                        **candidate,
                        "selection_priority":
                            "P2 interaction",
                        "selection_reason":
                            (
                                "Second priority: "
                                + candidate["reason"]
                            ),
                    }
                )

    # --------------------------------------------------------
    # PASS 3: application diversity.
    # --------------------------------------------------------

    if len(selected) < MAX_SELECTED_PER_SEGMENT:

        app_counts = defaultdict(int)

        for item in selected:
            app_counts[item["app"]] += 1

        for candidate in ranked:

            if len(selected) >= MAX_SELECTED_PER_SEGMENT:
                break

            if not far_enough(candidate):
                continue

            if (
                app_counts[candidate["app"]]
                >= 2
            ):
                continue

            selected.append(
                {
                    **candidate,
                    "selection_priority":
                        "P3 app diversity",
                    "selection_reason":
                        (
                            "Third priority: "
                            "application diversity; "
                            + candidate["reason"]
                        ),
                }
            )

            app_counts[candidate["app"]] += 1

    # --------------------------------------------------------
    # PASS 4: temporal coverage fallback.
    # --------------------------------------------------------

    if len(selected) < MAX_SELECTED_PER_SEGMENT:

        if candidates:

            timestamps = [
                x["timestamp"]
                for x in candidates
            ]

            min_ts = min(timestamps)
            max_ts = max(timestamps)

            targets = [
                (
                    "early fallback",
                    min_ts
                    + (max_ts - min_ts) * 0.15,
                ),
                (
                    "middle fallback",
                    min_ts
                    + (max_ts - min_ts) * 0.50,
                ),
                (
                    "late fallback",
                    min_ts
                    + (max_ts - min_ts) * 0.85,
                ),
            ]

            used_fallback_positions = set()

            for label, target_time in targets:

                if len(selected) >= MAX_SELECTED_PER_SEGMENT:
                    break

                best = None
                best_distance = None

                for candidate in ranked:

                    if not far_enough(candidate):
                        continue

                    distance = abs(
                        (
                            candidate["timestamp"]
                            - target_time
                        ).total_seconds()
                    )

                    if (
                        best_distance is None
                        or distance < best_distance
                    ):
                        best = candidate
                        best_distance = distance

                if best is not None:

                    selected.append(
                        {
                            **best,
                            "selection_priority":
                                "P4 temporal fallback",
                            "selection_reason":
                                (
                                    f"Fallback for {label}; "
                                    + best["reason"]
                                ),
                        }
                    )

    selected.sort(
        key=lambda x: x["timestamp"]
    )

    for i, item in enumerate(selected, start=1):
        item["selection_order"] = i

    return selected


def main():
    print("=" * 70)
    print("DATASET B SCREENSHOT SELECTION V2")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{INVENTORY_FILE}"
        )

    inventory = pd.read_csv(
        INVENTORY_FILE,
        keep_default_na=False,
    )

    inventory["segment_index"] = pd.to_numeric(
        inventory["segment_index"],
        errors="raise",
    ).astype(int)

    inventory["start"] = pd.to_datetime(
        inventory["start"],
        utc=True,
    )

    inventory["end"] = pd.to_datetime(
        inventory["end"],
        utc=True,
    )

    print(
        f"Segments to inspect: "
        f"{len(inventory):,}"
    )

    print()
    print("Loading raw Dataset B sessions...")

    all_sessions = load_all_events(
        DATA_ROOT
    )

    dataset_b_sessions = {}

    for session_id, events in all_sessions.items():
        datasets = {
            event.get("_dataset")
            for event in events
            if event.get("_dataset")
        }

        if "B" in datasets:
            dataset_b_sessions[
                session_id
            ] = events

    print(
        f"Dataset B sessions: "
        f"{len(dataset_b_sessions):,}"
    )

    prepared_sessions = {}

    for session_id in inventory["session_id"].unique():
        if session_id not in dataset_b_sessions:
            raise RuntimeError(
                f"Missing Dataset B session: {session_id}"
            )

        prepared_sessions[
            session_id
        ] = prepare_events(
            dataset_b_sessions[
                session_id
            ]
        )

    screenshot_index = build_screenshot_index()

    if SELECTED_DIR.exists():
        shutil.rmtree(
            SELECTED_DIR
        )

    SELECTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []

    for counter, (_, segment) in enumerate(
        inventory.iterrows(),
        start=1,
    ):
        session_id = segment[
            "session_id"
        ]

        segment_index = int(
            segment["segment_index"]
        )

        start = segment["start"]
        end = segment["end"]

        candidates = collect_segment_candidates(
            prepared_sessions[session_id],
            start,
            end,
        )

        selected = select_candidates(
            candidates
        )

        segment_dir = (
            SELECTED_DIR
            / session_id
            / f"segment_{segment_index:03d}"
        )
        segment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for item in selected:

            source_paths = screenshot_index.get(
                item["filename"],
                []
            )

            copied_paths = []

            for source_path in source_paths:

                destination = (
                    segment_dir
                    / (
                        f"{item['selection_order']:02d}_"
                        + item["filename"]
                    )
                )

                try:
                    shutil.copy2(
                        source_path,
                        destination,
                    )
                    copied_paths.append(
                        str(destination)
                    )
                except OSError:
                    pass

            output_rows.append(
                {
                    "session_id":
                        session_id,

                    "segment_index":
                        segment_index,

                    "segment_start":
                        start,

                    "segment_end":
                        end,

                    "segment_duration_seconds":
                        (
                            end - start
                        ).total_seconds(),

                    "selection_order":
                        item[
                            "selection_order"
                        ],

                    "screenshot_timestamp":
                        item["timestamp"],

                    "seconds_from_segment_start":
                        (
                            item["timestamp"]
                            - start
                        ).total_seconds(),

                    "app":
                        item["app"],

                    "source_event_type":
                        item[
                            "source_event_type"
                        ],

                    "selection_score":
                        round(
                            item["score"],
                            3,
                        ),

                    "selection_priority":
                        item[
                            "selection_priority"
                        ],

                    "selection_reason":
                        item[
                            "selection_reason"
                        ],

                    "nearest_app_transition":
                        item[
                            "nearest_transition"
                        ],

                    "nearest_transition_distance_seconds":
                        item[
                            "nearest_transition_distance_seconds"
                        ],

                    "nearest_event":
                        item[
                            "nearest_event"
                        ],

                    "nearest_event_distance_seconds":
                        item[
                            "nearest_event_distance_seconds"
                        ],

                    "filename":
                        item["filename"],

                    "source_paths":
                        " | ".join(
                            str(p)
                            for p in source_paths
                        ),

                    "copied_paths":
                        " | ".join(
                            copied_paths
                        ),

                    "ambiguous_filename":
                        len(source_paths) > 1,
                }
            )

        if (
            counter % 50 == 0
            or counter == len(inventory)
        ):
            print(
                f"Processed "
                f"{counter}/"
                f"{len(inventory)} segments"
            )

    result = pd.DataFrame(
        output_rows
    )

    result.to_csv(
        SELECTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("SELECTION AUDIT SUMMARY")
    print("=" * 70)

    print(
        f"Segments analyzed: "
        f"{len(inventory):,}"
    )

    print(
        f"Screenshots selected: "
        f"{len(result):,}"
    )

    print(
        f"Mean screenshots/segment: "
        f"{len(result) / len(inventory):.2f}"
    )

    if not result.empty:

        print()
        print("By selection priority:")

        print(
            result[
                "selection_priority"
            ]
            .value_counts()
            .to_string()
        )

        print()
        print("By source app:")

        print(
            result[
                "app"
            ]
            .value_counts()
            .head(15)
            .to_string()
        )

        print()
        print("Screenshots near app transitions:")

        transition_mask = (
            result[
                "nearest_app_transition"
            ].astype(str).str.len()
            > 0
        )

        print(
            f"{int(transition_mask.sum()):,}"
            f" / {len(result):,}"
            f" = {transition_mask.mean():.2%}"
        )

        print()
        print("Top transition types:")

        print(
            result[
                "nearest_app_transition"
            ]
            .loc[transition_mask]
            .value_counts()
            .head(15)
            .to_string()
        )

    print()
    print("=" * 70)
    print("SCREENSHOT SELECTION V2 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Audit CSV:\n"
        f"{SELECTION_FILE}"
    )

    print()
    print(
        f"Selected screenshots:\n"
        f"{SELECTED_DIR}"
    )


if __name__ == "__main__":
    main()
