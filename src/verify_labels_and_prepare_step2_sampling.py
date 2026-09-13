
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_ROOT = (
    PROJECT_ROOT
    / "raw_data"
    / "dataset-downloads"
)

AUTHORITATIVE_SEGMENTS = (
    OUTPUT_DIR / "dataset_b_segments_authoritative.csv"
)
LABEL_AUDIT = (
    OUTPUT_DIR / "dataset_b_segments_label_audit.csv"
)
FEATURES = (
    OUTPUT_DIR / "dataset_b_process_features_with_operator_v2.csv"
)

FINAL_JSONL = (
    PROJECT_ROOT / "segments.jsonl"
)

OUT_COMPARISON = (
    OUTPUT_DIR
    / "dataset_b_label_reconciliation.csv"
)
OUT_OTHER_AUDIT = (
    OUTPUT_DIR
    / "dataset_b_other_segments_audit.csv"
)
OUT_FIXED_SEGMENTS = (
    OUTPUT_DIR
    / "dataset_b_segments_authoritative_v2.csv"
)
OUT_FIXED_JSONL = (
    PROJECT_ROOT / "segments_v2.jsonl"
)
OUT_SAMPLE = (
    OUTPUT_DIR
    / "dataset_b_ocr_sample_75.csv"
)


PREFIXES = (
    "pi",
    "la",
    "ob",
    "si",
    "rt",
)

# Supports both:
#   id=pi-note
#   id=btn-pi-ok
# and the corresponding name/target-field payloads.
PREFIX_PATTERN = re.compile(
    r"(?:^|=)"
    r"(?:btn-)?"
    r"(pi|la|ob|si|rt)"
    r"(?:-note|-ok)"
    r"$",
    re.IGNORECASE,
)


def split_tokens(value):
    text = str(
        value if value is not None else ""
    ).strip()

    if not text or text in {
        "nan",
        "None",
    }:
        return []

    return [
        x.strip()
        for x in text.split("|")
        if x.strip()
    ]


def extract_prefix_counts_from_field_tokens(
    field_token_set
):
    counts = Counter()

    for token in split_tokens(
        field_token_set
    ):
        value = token
        if value.startswith("field:"):
            value = value[len("field:"):]

        for match in PREFIX_PATTERN.finditer(
            value.lower()
        ):
            counts[
                match.group(1).lower()
            ] += 1

    return counts


def get_payload(event):
    value = event.get("payload")
    return value if isinstance(value, dict) else {}


def collect_strings(obj):
    found = []

    if isinstance(
        obj,
        dict,
    ):
        for key, value in obj.items():
            if isinstance(value, str):
                found.append(
                    (str(key), value)
                )
            else:
                found.extend(
                    collect_strings(value)
                )

    elif isinstance(
        obj,
        list,
    ):
        for value in obj:
            found.extend(
                collect_strings(value)
            )

    return found


def event_type(event):
    value = event.get("event_type")
    return (
        value.lower()
        if isinstance(value, str)
        else ""
    )


def extract_prefix_counts_from_raw_event(
    event
):
    et = event_type(event)

    if et not in {
        "browser_click",
        "browser_form_input",
        "keystroke",
    }:
        return Counter()

    counts = Counter()

    for key, value in collect_strings(
        get_payload(event)
    ):
        candidates = (
            f"{key}={value}",
            str(value),
        )

        for candidate in candidates:
            for match in PREFIX_PATTERN.finditer(
                candidate.lower().strip()
            ):
                counts[
                    match.group(1).lower()
                ] += 1

    return counts


def timestamp_ms(event):
    value = event.get(
        "timestamp_ms"
    )

    if value is not None:
        try:
            return int(value)
        except Exception:
            pass

    return None


def load_b_events_for_session(
    session_id
):
    matches = []

    for root in DATA_ROOT.rglob(
        "dataset_b"
    ):
        session_dir = (
            root / session_id
        )

        if session_dir.is_dir():
            matches.append(
                session_dir
            )

    matches = sorted(
        set(matches)
    )

    if not matches:
        raise FileNotFoundError(
            f"Dataset-B session not found: {session_id}"
        )

    # Prefer the candidate with the greatest total event-file size.
    scored = []

    for path in matches:
        event_files = list(
            path.rglob("events.jsonl")
        )

        total_size = sum(
            f.stat().st_size
            for f in event_files
            if f.exists()
        )

        scored.append(
            (
                len(event_files),
                total_size,
                str(path),
                path,
            )
        )

    scored.sort(
        reverse=True
    )

    session_dir = scored[0][-1]

    events = []

    for event_file in sorted(
        session_dir.rglob(
            "events.jsonl"
        )
    ):
        with event_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    event = json.loads(line)
                except Exception:
                    continue

                ts = timestamp_ms(
                    event
                )

                if ts is None:
                    continue

                events.append(
                    (
                        ts,
                        event,
                    )
                )

    events.sort(
        key=lambda x: x[0]
    )

    return events


def application_variant(
    feature_row
):
    has_word = int(
        float(
            feature_row.get(
                "has_word",
                0,
            )
            or 0
        )
    )
    has_notepad = int(
        float(
            feature_row.get(
                "has_notepad",
                0,
            )
            or 0
        )
    )
    has_excel = int(
        float(
            feature_row.get(
                "has_excel",
                0,
            )
            or 0
        )
    )

    if has_word:
        return "word_assisted"

    if has_notepad:
        return "notepad_assisted"

    if has_excel:
        return "excel_assisted"

    return "browser_only"


def choose_prefix(
    counts
):
    if not counts:
        return (
            "other",
            "NONE",
        )

    max_count = max(
        counts.values()
    )

    winners = sorted(
        prefix
        for prefix, count
        in counts.items()
        if count == max_count
    )

    if len(counts) == 1:
        return (
            winners[0],
            "SINGLE_PREFIX",
        )

    return (
        winners[0],
        "MULTI_DOMINANT_PREFIX",
    )


def iso_z(value):
    ts = pd.to_datetime(
        value,
        utc=True,
    )
    return (
        ts.isoformat(
            timespec="milliseconds"
        ).replace(
            "+00:00",
            "Z",
        )
    )


def find_segment_screenshots(
    session_id,
    start,
    end,
):
    """
    Locate screenshot_smart images whose event timestamps fall inside
    the segment. Returns actual file paths where available.
    """
    matches = []

    for root in DATA_ROOT.rglob(
        "dataset_b"
    ):
        session_dir = (
            root / session_id
        )

        if not session_dir.is_dir():
            continue

        for event_file in sorted(
            session_dir.rglob(
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

                        if event.get(
                            "event_type"
                        ) != "screenshot_smart":
                            continue

                        ts_ms = timestamp_ms(
                            event
                        )
                        if ts_ms is None:
                            continue

                        ts = pd.to_datetime(
                            ts_ms,
                            unit="ms",
                            utc=True,
                        )

                        if not (
                            start <= ts <= end
                        ):
                            continue

                        # Inspect common screenshot-reference locations.
                        refs = []

                        for key in (
                            "screenshot_path",
                            "image_path",
                            "file_path",
                            "path",
                            "filename",
                            "file_name",
                        ):
                            value = event.get(
                                key
                            )
                            if isinstance(
                                value,
                                str,
                            ):
                                refs.append(
                                    value
                                )

                        payload = get_payload(
                            event
                        )

                        for key in (
                            "screenshot_path",
                            "image_path",
                            "file_path",
                            "path",
                            "filename",
                            "file_name",
                        ):
                            value = payload.get(
                                key
                            )
                            if isinstance(
                                value,
                                str,
                            ):
                                refs.append(
                                    value
                                )

                        # Resolve referenced path if present.
                        resolved = None

                        for ref in refs:
                            candidate = Path(
                                ref
                            )

                            if candidate.is_absolute():
                                if candidate.exists():
                                    resolved = candidate
                                    break

                            candidate2 = (
                                session_dir
                                / ref
                            )
                            if candidate2.exists():
                                resolved = candidate2
                                break

                        # If no explicit reference is found, search the
                        # session screenshots using the event filename/id.
                        if resolved is None:
                            base = (
                                payload.get(
                                    "screenshot_id"
                                )
                                or payload.get(
                                    "id"
                                )
                                or event.get(
                                    "screenshot_id"
                                )
                            )

                            if isinstance(
                                base,
                                str,
                            ):
                                possible = [
                                    p
                                    for p in session_dir.rglob(
                                        "*"
                                    )
                                    if p.is_file()
                                    and base in p.name
                                ]

                                if possible:
                                    resolved = sorted(
                                        possible
                                    )[0]

                        matches.append(
                            {
                                "screenshot_timestamp":
                                    ts,
                                "screenshot_file":
                                    str(
                                        resolved
                                    )
                                    if resolved
                                    else "",
                            }
                        )

            except OSError:
                continue

        break

    matches.sort(
        key=lambda x:
            x[
                "screenshot_timestamp"
            ]
    )

    return matches


def main():
    print("=" * 70)
    print(
        "STEP 2 LABEL RECONCILIATION + OCR SAMPLE PREPARATION"
    )
    print("=" * 70)

    for path in (
        AUTHORITATIVE_SEGMENTS,
        LABEL_AUDIT,
        FEATURES,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    segments = pd.read_csv(
        AUTHORITATIVE_SEGMENTS,
        keep_default_na=False,
    )

    audit = pd.read_csv(
        LABEL_AUDIT,
        keep_default_na=False,
    )

    features = pd.read_csv(
        FEATURES,
        keep_default_na=False,
    )

    key = [
        "session_id",
        "segment_index",
    ]

    print()
    print(
        f"Authoritative segment rows: {len(segments):,}"
    )
    print(
        f"Audit rows: {len(audit):,}"
    )
    print(
        f"Feature rows: {len(features):,}"
    )

    # --------------------------------------------------------
    # 1. Reconcile existing audit vs export.
    # --------------------------------------------------------

    comparison = segments[
        key + [
            "label",
            "label_resolution",
            "prefix_evidence",
        ]
    ].merge(
        audit[
            key + [
                "label",
                "label_resolution",
                "prefix_evidence",
            ]
        ],
        on=key,
        how="outer",
        suffixes=(
            "_export",
            "_audit",
        ),
        validate="one_to_one",
    )

    comparison[
        "label_changed"
    ] = (
        comparison[
            "label_export"
        ]
        != comparison[
            "label_audit"
        ]
    )

    comparison[
        "resolution_changed"
    ] = (
        comparison[
            "label_resolution_export"
        ]
        != comparison[
            "label_resolution_audit"
        ]
    )

    disagreements = comparison[
        comparison[
            "label_changed"
        ]
    ].copy()

    print()
    print("=" * 70)
    print(
        "LABEL RECONCILIATION"
    )
    print("=" * 70)

    print(
        f"Label disagreements: "
        f"{len(disagreements):,} / {len(comparison):,}"
    )

    if not disagreements.empty:
        print()
        print(
            "Disagreement matrix:"
        )
        print(
            pd.crosstab(
                disagreements[
                    "label_audit"
                ],
                disagreements[
                    "label_export"
                ],
            ).to_string()
        )

        print()
        print(
            "Sample disagreements:"
        )
        print(
            disagreements[
                [
                    "session_id",
                    "segment_index",
                    "label_audit",
                    "label_export",
                    "prefix_evidence_audit",
                    "prefix_evidence_export",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    comparison.to_csv(
        OUT_COMPARISON,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 2. Recompute prefix evidence from the authoritative raw events,
    #    including btn-XX-ok, and create corrected labels.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RECOMPUTING PREFIX EVIDENCE WITH BTN-XX-OK SUPPORT"
    )
    print("=" * 70)

    corrected_rows = []

    session_cache = {}

    for i, session_id in enumerate(
        sorted(
            segments[
                "session_id"
            ].astype(str).unique()
        ),
        start=1,
    ):
        print(
            f"Session {i}/15: {session_id}"
        )

        events = load_b_events_for_session(
            session_id
        )

        session_cache[
            session_id
        ] = events

    for _, row in segments.iterrows():

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

        counts = Counter()

        for ts_ms, event in session_cache[
            session_id
        ]:
            ts = pd.to_datetime(
                ts_ms,
                unit="ms",
                utc=True,
            )

            if (
                ts < start
                or ts > end
            ):
                continue

            counts.update(
                extract_prefix_counts_from_raw_event(
                    event
                )
            )

        label, resolution = choose_prefix(
            counts
        )

        corrected_rows.append(
            {
                "session_id":
                    session_id,
                "segment_index":
                    int(
                        row[
                            "segment_index"
                        ]
                    ),
                "start":
                    start,
                "end":
                    end,
                "duration_seconds":
                    float(
                        row[
                            "duration_seconds"
                        ]
                    ),
                "label":
                    label,
                "label_resolution":
                    resolution,
                "prefix_evidence":
                    json.dumps(
                        dict(
                            sorted(
                                counts.items()
                            )
                        ),
                        ensure_ascii=False,
                    ),
                "event_count":
                    int(
                        row[
                            "event_count"
                        ]
                    ),
            }
        )

    corrected = pd.DataFrame(
        corrected_rows
    )

    print()
    print(
        "Corrected label distribution:"
    )
    print(
        corrected[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Corrected resolution:"
    )
    print(
        corrected[
            "label_resolution"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Compare against exported labels.
    recheck = segments[
        key + [
            "label"
        ]
    ].merge(
        corrected[
            key + [
                "label"
            ]
        ],
        on=key,
        suffixes=(
            "_previous",
            "_corrected",
        ),
        validate="one_to_one",
    )

    changed = recheck[
        recheck[
            "label_previous"
        ]
        != recheck[
            "label_corrected"
        ]
    ]

    print()
    print(
        f"Changes after btn-aware recomputation: "
        f"{len(changed):,} / {len(recheck):,}"
    )

    # --------------------------------------------------------
    # 3. Other-segment diagnostic.
    # --------------------------------------------------------

    other = corrected[
        corrected[
            "label"
        ]
        == "other"
    ].copy()

    print()
    print("=" * 70)
    print(
        "OTHER SEGMENT CHECK"
    )
    print("=" * 70)

    print(
        f"Other segments: "
        f"{len(other):,}"
    )

    if not other.empty:
        print()
        print(
            other[
                [
                    "session_id",
                    "segment_index",
                    "duration_seconds",
                    "event_count",
                    "prefix_evidence",
                ]
            ]
            .sort_values(
                [
                    "event_count",
                    "duration_seconds",
                ],
                ascending=False,
            )
            .head(40)
            .to_string(index=False)
        )

    # Enrich with structural features.
    other_enriched = other.merge(
        features,
        on=[
            "session_id",
            "segment_index",
        ],
        how="left",
        suffixes=(
            "",
            "_feature",
        ),
        validate="one_to_one",
    )

    if not other_enriched.empty:
        other_enriched[
            "application_variant"
        ] = other_enriched.apply(
            application_variant,
            axis=1,
        )

        print()
        print(
            "Other segment activity summary:"
        )

        summary_columns = [
            "event_count",
            "duration_seconds",
            "browser_click_count",
            "browser_form_input_count",
            "browser_navigation_count",
            "clipboard_count",
            "unique_window_identities",
        ]

        available = [
            x
            for x in summary_columns
            if x in other_enriched.columns
        ]

        print(
            other_enriched[
                available
            ]
            .describe()
            .to_string()
        )

    other_enriched.to_csv(
        OUT_OTHER_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 4. Create corrected authoritative CSV + JSONL.
    # --------------------------------------------------------

    corrected = corrected.sort_values(
        [
            "session_id",
            "segment_index",
        ]
    ).reset_index(drop=True)

    corrected.to_csv(
        OUT_FIXED_SEGMENTS,
        index=False,
        encoding="utf-8-sig",
    )

    with OUT_FIXED_JSONL.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for _, row in corrected.iterrows():
            record = {
                "session_id":
                    str(
                        row[
                            "session_id"
                        ]
                    ),
                "start":
                    iso_z(
                        row["start"]
                    ),
                "end":
                    iso_z(
                        row["end"]
                    ),
                "label":
                    str(
                        row["label"]
                    ),
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # 5. Build stratified screenshot sample.
    #
    # Target is about 75 screenshots:
    #   5 families × ~15 screenshots
    #
    # Within each family, prioritize variants and stronger evidence.
    # We initially allocate approximately 5 per observed variant,
    # then fill remaining quota from the strongest evidence.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BUILDING STRATIFIED OCR SAMPLE"
    )
    print("=" * 70)

    feature_subset = features[
        [
            "session_id",
            "segment_index",
            "has_word",
            "has_notepad",
            "has_excel",
            "operator_hashes",
        ]
    ].copy()

    sample_segments = corrected.merge(
        feature_subset,
        on=[
            "session_id",
            "segment_index",
        ],
        how="left",
        validate="one_to_one",
    )

    sample_segments[
        "application_variant"
    ] = sample_segments.apply(
        application_variant,
        axis=1,
    )

    sample_segments[
        "prefix_strength"
    ] = sample_segments[
        "prefix_evidence"
    ].apply(
        lambda x:
            max(
                json.loads(x).values()
            )
            if x and x != "{}"
            else 0
    )

    # Ignore other for business-name sampling.
    candidate_segments = sample_segments[
        sample_segments[
            "label"
        ].isin(
            PREFIXES
        )
    ].copy()

    # Build screenshot inventory.
    screenshot_rows = []

    for _, row in candidate_segments.iterrows():

        screenshots = find_segment_screenshots(
            str(
                row["session_id"]
            ),
            pd.to_datetime(
                row["start"],
                utc=True,
            ),
            pd.to_datetime(
                row["end"],
                utc=True,
            ),
        )

        for screen in screenshots:
            screenshot_rows.append(
                {
                    "label":
                        row["label"],
                    "session_id":
                        row["session_id"],
                    "segment_index":
                        int(
                            row[
                                "segment_index"
                            ]
                        ),
                    "start":
                        row["start"],
                    "end":
                        row["end"],
                    "application_variant":
                        row[
                            "application_variant"
                        ],
                    "prefix_strength":
                        row[
                            "prefix_strength"
                        ],
                    "screenshot_timestamp":
                        screen[
                            "screenshot_timestamp"
                        ],
                    "screenshot_file":
                        screen[
                            "screenshot_file"
                        ],
                }
            )

    screenshot_inventory = pd.DataFrame(
        screenshot_rows
    )

    if screenshot_inventory.empty:
        raise RuntimeError(
            "No screenshot_smart images were found "
            "inside the labeled B segments."
        )

    # Prefer actual resolved files.
    screenshot_inventory[
        "has_file"
    ] = screenshot_inventory[
        "screenshot_file"
    ].astype(str).str.len().gt(0)

    screenshot_inventory = (
        screenshot_inventory
        .sort_values(
            [
                "has_file",
                "prefix_strength",
                "screenshot_timestamp",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "label",
                "session_id",
                "segment_index",
            ],
            keep="first",
        )
    )

    # Target up to 15 segments/screenshots per family.
    # Variant-aware selection: try one strongest segment from each
    # observed variant repeatedly until quota is filled.
    selected = []

    for prefix in PREFIXES:

        family = screenshot_inventory[
            screenshot_inventory[
                "label"
            ]
            == prefix
        ].copy()

        if family.empty:
            continue

        family = family.sort_values(
            [
                "prefix_strength",
                "has_file",
                "screenshot_timestamp",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

        chosen_indices = []

        variants = sorted(
            family[
                "application_variant"
            ].unique()
        )

        # First pass: up to 5 per observed variant.
        for variant in variants:

            subset = family[
                family[
                    "application_variant"
                ]
                == variant
            ]

            take = min(
                5,
                len(subset),
            )

            chosen_indices.extend(
                subset.head(
                    take
                ).index.tolist()
            )

        # Fill remaining family quota with strongest unused samples.
        remaining = (
            family.drop(
                index=chosen_indices,
                errors="ignore",
            )
        )

        need = max(
            0,
            15 - len(chosen_indices),
        )

        if need:
            chosen_indices.extend(
                remaining.head(
                    need
                ).index.tolist()
            )

        for index in chosen_indices:
            selected.append(
                screenshot_inventory.loc[
                    index
                ]
            )

    sample = pd.DataFrame(
        selected
    ).drop_duplicates(
        subset=[
            "label",
            "session_id",
            "segment_index",
        ]
    )

    # Keep strongest 15 per family if oversampled.
    final_parts = []

    for prefix in PREFIXES:
        part = sample[
            sample[
                "label"
            ]
            == prefix
        ].sort_values(
            [
                "prefix_strength",
                "has_file",
                "screenshot_timestamp",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).head(15)

        final_parts.append(
            part
        )

    final_sample = pd.concat(
        final_parts,
        ignore_index=True,
    )

    final_sample[
        [
            "label",
            "session_id",
            "segment_index",
            "application_variant",
            "prefix_strength",
            "screenshot_timestamp",
            "screenshot_file",
        ]
    ].to_csv(
        OUT_SAMPLE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "OCR sample counts:"
    )
    print(
        final_sample[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "OCR sample by family and variant:"
    )

    print(
        pd.crosstab(
            final_sample[
                "label"
            ],
            final_sample[
                "application_variant"
            ],
        ).to_string()
    )

    print()
    print(
        "Sample rows with resolved files:"
    )
    print(
        int(
            final_sample[
                "has_file"
            ].sum()
        ),
        "/",
        len(final_sample),
    )

    print()
    print("=" * 70)
    print(
        "COMPLETE"
    )
    print("=" * 70)

    print(
        f"Label reconciliation:\n  {OUT_COMPARISON}"
    )

    print(
        f"Other-segment audit:\n  {OUT_OTHER_AUDIT}"
    )

    print(
        f"Corrected segments:\n  {OUT_FIXED_SEGMENTS}"
    )

    print(
        f"Corrected JSONL:\n  {OUT_FIXED_JSONL}"
    )

    print(
        f"OCR sample:\n  {OUT_SAMPLE}"
    )

    print()
    print(
        "Do not OCR until the reconciliation and sample counts above "
        "have been reviewed."
    )


if __name__ == "__main__":
    main()
