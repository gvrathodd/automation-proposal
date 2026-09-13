
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from paddleocr import PaddleOCR
except Exception as exc:
    print("Could not import PaddleOCR.")
    print(exc)
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"

SEGMENTS = (
    OUT / "dataset_b_segments_authoritative_v2.csv"
)
FEATURES = (
    OUT / "dataset_b_process_features_with_operator_v2.csv"
)
EXISTING_75 = (
    OUT / "dataset_b_ocr_sample_75_verified_v2.csv"
)

SELECTED = (
    OUT / "dataset_b_ocr_sample_20pct.csv"
)
SELECTED_VERIFIED = (
    OUT / "dataset_b_ocr_sample_20pct_verified.csv"
)
OCR_JSONL = (
    OUT / "dataset_b_ocr_raw_20pct.jsonl"
)
OCR_SUMMARY = (
    OUT / "dataset_b_ocr_summary_20pct.csv"
)

TARGET_FRACTION = 0.20
SEED = 20260912


def parse_timestamp_ms(event):
    value = event.get("timestamp_ms")

    if value is not None:
        try:
            return int(value)
        except Exception:
            pass

    return None


def session_dirs(session_id):
    found = []

    for root in DATA_ROOT.rglob("dataset_b"):
        path = root / session_id

        if path.is_dir():
            found.append(path)

    return sorted(set(found))


def choose_session_dir(session_id):
    candidates = session_dirs(session_id)

    if not candidates:
        raise FileNotFoundError(
            f"Dataset-B session not found: {session_id}"
        )

    scored = []

    for path in candidates:
        screenshots = [
            p
            for p in path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }
        ]

        scored.append(
            (
                len(screenshots),
                sum(
                    p.stat().st_size
                    for p in screenshots
                    if p.exists()
                ),
                str(path),
                path,
            )
        )

    scored.sort(reverse=True)

    return scored[0][-1]


def build_screenshot_inventory():
    """
    Read every Dataset-B screenshot_smart event and resolve its actual
    screenshot file. The reference lives at:

        payload.file_reference.relative_path

    We attach each screenshot to the 540 final segments by timestamp.
    """
    segments = pd.read_csv(
        SEGMENTS,
        keep_default_na=False,
    )

    required = {
        "session_id",
        "segment_index",
        "start",
        "end",
        "label",
    }

    missing = required - set(
        segments.columns
    )

    if missing:
        raise RuntimeError(
            "Segments file missing: "
            + ", ".join(sorted(missing))
        )

    segments["start"] = pd.to_datetime(
        segments["start"],
        utc=True,
    )
    segments["end"] = pd.to_datetime(
        segments["end"],
        utc=True,
    )

    # Session-wise segment lookup.
    by_session = {
        session_id: group.sort_values(
            "start"
        ).reset_index(drop=True)
        for session_id, group
        in segments.groupby(
            "session_id"
        )
    }

    rows = []

    sessions = sorted(
        segments["session_id"]
        .astype(str)
        .unique()
    )

    for pos, session_id in enumerate(
        sessions,
        start=1,
    ):
        print(
            f"Scanning screenshots "
            f"{pos}/{len(sessions)}: "
            f"{session_id}"
        )

        session_dir = choose_session_dir(
            session_id
        )

        group = by_session[
            session_id
        ]

        # Pre-index segments.
        starts_ms = [
            int(
                ts.timestamp() * 1000
            )
            for ts in group["start"]
        ]
        ends_ms = [
            int(
                ts.timestamp() * 1000
            )
            for ts in group["end"]
        ]

        # Resolve screenshot files by basename once.
        image_index = {}

        for image in session_dir.rglob("*"):
            if (
                image.is_file()
                and image.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                }
            ):
                image_index.setdefault(
                    image.name,
                    image,
                )

        for event_file in sorted(
            session_dir.rglob("events.jsonl")
        ):
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
                        event = json.loads(line)
                    except Exception:
                        continue

                    if (
                        event.get(
                            "event_type"
                        )
                        != "screenshot_smart"
                    ):
                        continue

                    ts_ms = parse_timestamp_ms(
                        event
                    )

                    if ts_ms is None:
                        continue

                    # Find segment by timestamp.
                    segment_position = None

                    for i in range(
                        len(group)
                    ):
                        if (
                            starts_ms[i]
                            <= ts_ms
                            <= ends_ms[i]
                        ):
                            segment_position = i
                            break

                    if segment_position is None:
                        continue

                    segment_row = group.iloc[
                        segment_position
                    ]

                    payload = event.get(
                        "payload"
                    )

                    if not isinstance(
                        payload,
                        dict,
                    ):
                        payload = {}

                    file_reference = payload.get(
                        "file_reference"
                    )

                    if not isinstance(
                        file_reference,
                        dict,
                    ):
                        continue

                    relative_path = (
                        file_reference.get(
                            "relative_path"
                        )
                    )

                    filename = (
                        file_reference.get(
                            "filename"
                        )
                    )

                    if not isinstance(
                        relative_path,
                        str,
                    ):
                        relative_path = ""

                    if not isinstance(
                        filename,
                        str,
                    ):
                        filename = ""

                    basename = Path(
                        relative_path
                    ).name

                    if not basename:
                        basename = filename

                    resolved = image_index.get(
                        basename
                    )

                    if resolved is None:
                        continue

                    rows.append(
                        {
                            "session_id":
                                session_id,
                            "segment_index":
                                int(
                                    segment_row[
                                        "segment_index"
                                    ]
                                ),
                            "label":
                                str(
                                    segment_row[
                                        "label"
                                    ]
                                ),
                            "segment_start":
                                segment_row[
                                    "start"
                                ],
                            "segment_end":
                                segment_row[
                                    "end"
                                ],
                            "screenshot_timestamp":
                                pd.to_datetime(
                                    ts_ms,
                                    unit="ms",
                                    utc=True,
                                ),
                            "screenshot_file":
                                str(
                                    relative_path
                                ),
                            "resolved_screenshot_file":
                                str(
                                    resolved.resolve()
                                ),
                            "event_file":
                                str(
                                    event_file
                                ),
                            "event_line":
                                line_number,
                        }
                    )

    inventory = pd.DataFrame(
        rows
    )

    if inventory.empty:
        raise RuntimeError(
            "No Dataset-B screenshots could be resolved."
        )

    return inventory


def add_variants_and_existing_75(
    inventory
):
    features = pd.read_csv(
        FEATURES,
        keep_default_na=False,
    )

    feature_cols = [
        "session_id",
        "segment_index",
        "has_word",
        "has_notepad",
        "has_excel",
    ]

    feature_cols = [
        c
        for c in feature_cols
        if c in features.columns
    ]

    inventory = inventory.merge(
        features[
            feature_cols
        ],
        on=[
            "session_id",
            "segment_index",
        ],
        how="left",
        validate="many_to_one",
    )

    def variant(row):
        if int(float(row.get(
            "has_word", 0
        ) or 0)):
            return "word_assisted"

        if int(float(row.get(
            "has_notepad", 0
        ) or 0)):
            return "notepad_assisted"

        if int(float(row.get(
            "has_excel", 0
        ) or 0)):
            return "excel_assisted"

        return "browser_only"

    inventory[
        "application_variant"
    ] = inventory.apply(
        variant,
        axis=1,
    )

    inventory[
        "is_in_existing_75"
    ] = 0

    if EXISTING_75.exists():
        old = pd.read_csv(
            EXISTING_75,
            keep_default_na=False,
        )

        old_keys = set(
            zip(
                old[
                    "session_id"
                ].astype(str),
                old[
                    "segment_index"
                ].astype(int),
            )
        )

        inventory[
            "is_in_existing_75"
        ] = [
            int(
                (
                    str(row["session_id"]),
                    int(row["segment_index"]),
                )
                in old_keys
            )
            for _, row
            in inventory.iterrows()
        ]

    return inventory


def stratified_sample(
    inventory
):
    """
    Select ~20% of all resolved screenshots.

    Priority:
      1. Keep screenshots from the existing verified 75-sample.
      2. Maintain broad family coverage.
      3. Maintain session coverage within each family.
      4. Cover observed application variants within each family.
      5. Fill remaining quota randomly but deterministically.
    """
    total = len(
        inventory
    )

    target = round(
        total * TARGET_FRACTION
    )

    # At least the verified 75, but never above target.
    existing = inventory[
        inventory[
            "is_in_existing_75"
        ]
        == 1
    ].copy()

    selected_indices = set()

    # Keep existing sample first.
    for index in existing.index:
        if len(selected_indices) >= target:
            break

        selected_indices.add(
            index
        )

    rng = random.Random(
        SEED
    )

    # We want balanced family representation.
    families = [
        x
        for x in (
            "pi",
            "la",
            "ob",
            "si",
            "rt",
        )
        if x
        in set(
            inventory[
                "label"
            ]
        )
    ]

    # Allocate target approximately proportional to screenshot volume,
    # with a minimum floor for each family.
    family_counts = (
        inventory[
            inventory["label"].isin(
                families
            )
        ]
        .groupby("label")
        .size()
    )

    raw_targets = {
        family: max(
            1,
            round(
                target
                * int(
                    family_counts[
                        family
                    ]
                )
                / sum(
                    family_counts.values
                )
            ),
        )
        for family in families
    }

    # Normalize total family targets.
    while sum(raw_targets.values()) > target:
        family = max(
            raw_targets,
            key=raw_targets.get,
        )

        if raw_targets[
            family
        ] > 1:
            raw_targets[
                family
            ] -= 1
        else:
            break

    while sum(raw_targets.values()) < target:
        family = max(
            families,
            key=lambda f:
                int(
                    family_counts[
                        f
                    ]
                )
                - raw_targets[
                    f
                ],
        )
        raw_targets[
            family
        ] += 1

    # First ensure each family reaches its target, while respecting the
    # already-selected verified 75.
    for family in families:

        family_pool = inventory[
            inventory[
                "label"
            ]
            == family
        ].copy()

        already = [
            i
            for i in selected_indices
            if inventory.loc[
                i,
                "label",
            ]
            == family
        ]

        need = max(
            0,
            raw_targets[
                family
            ] - len(already),
        )

        if need <= 0:
            continue

        candidates = (
            family_pool.drop(
                index=selected_indices,
                errors="ignore",
            )
        )

        # Prioritize variants and sessions not yet represented.
        selected_variants = set(
            family_pool.loc[
                already,
                "application_variant",
            ]
            if already
            else []
        )

        selected_sessions = set(
            family_pool.loc[
                already,
                "session_id",
            ]
            if already
            else []
        )

        remaining = candidates.copy()

        ranked = []

        for idx, row in remaining.iterrows():
            score = 0

            if (
                row[
                    "application_variant"
                ]
                not in selected_variants
            ):
                score += 100

            if (
                row[
                    "session_id"
                ]
                not in selected_sessions
            ):
                score += 50

            score += rng.random()

            ranked.append(
                (
                    score,
                    idx,
                )
            )

        ranked.sort(
            reverse=True
        )

        for _, idx in ranked[
            :need
        ]:
            selected_indices.add(
                idx
            )

    # Fill any remaining target.
    if len(selected_indices) < target:
        remaining = [
            idx
            for idx
            in inventory.index
            if idx not in selected_indices
        ]

        rng.shuffle(
            remaining
        )

        selected_indices.update(
            remaining[
                : (
                    target
                    - len(selected_indices)
                )
            ]
        )

    selected = (
        inventory.loc[
            sorted(
                selected_indices
            )
        ]
        .sort_values(
            [
                "label",
                "session_id",
                "segment_index",
                "screenshot_timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return selected, target


def main():
    print("=" * 70)
    print(
        "DATASET-B 20% SCREENSHOT SELECTION + OCR"
    )
    print("=" * 70)

    for path in (
        SEGMENTS,
        FEATURES,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    inventory = build_screenshot_inventory()

    print()
    print(
        f"Resolved Dataset-B screenshots: "
        f"{len(inventory):,}"
    )

    inventory = add_variants_and_existing_75(
        inventory
    )

    selected, target = stratified_sample(
        inventory
    )

    print()
    print(
        f"Target (~20%): {target:,}"
    )
    print(
        f"Selected: {len(selected):,}"
    )

    print()
    print(
        "Selected by family:"
    )
    print(
        selected[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Selected by family × variant:"
    )
    print(
        pd.crosstab(
            selected[
                "label"
            ],
            selected[
                "application_variant"
            ],
        ).to_string()
    )

    print()
    print(
        "Selected sessions:"
    )
    print(
        selected[
            "session_id"
        ]
        .nunique()
    )

    selected[
        [
            "label",
            "session_id",
            "segment_index",
            "application_variant",
            "screenshot_timestamp",
            "screenshot_file",
            "resolved_screenshot_file",
            "is_in_existing_75",
        ]
    ].to_csv(
        SELECTED,
        index=False,
        encoding="utf-8-sig",
    )

    # Every selected file must exist.
    bad = selected[
        ~selected[
            "resolved_screenshot_file"
        ].map(
            lambda x:
                Path(str(x)).exists()
        )
    ]

    if not bad.empty:
        raise RuntimeError(
            f"{len(bad)} selected screenshots "
            "could not be verified on disk."
        )

    selected.to_csv(
        SELECTED_VERIFIED,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print(
        "STARTING PADDLEOCR"
    )
    print("=" * 70)

    ocr = PaddleOCR(
        lang="japan",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    rows = []

    with OCR_JSONL.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as out:

        for i, row in selected.iterrows():

            image_path = Path(
                row[
                    "resolved_screenshot_file"
                ]
            )

            print(
                f"[{i + 1}/{len(selected)}] "
                f"{row['label']} | "
                f"{row['application_variant']} | "
                f"{row['session_id']} | "
                f"segment {int(row['segment_index'])}"
            )

            try:
                prediction = ocr.predict(
                    str(image_path)
                )

                lines = []

                for result in prediction:

                    data = None

                    try:
                        value = result.json

                        if callable(value):
                            value = value()

                        if isinstance(
                            value,
                            str,
                        ):
                            value = json.loads(
                                value
                            )

                        if isinstance(
                            value,
                            dict,
                        ):
                            data = value
                    except Exception:
                        data = None

                    if not data:
                        continue

                    if isinstance(
                        data.get("res"),
                        dict,
                    ):
                        data = data[
                            "res"
                        ]

                    texts = data.get(
                        "rec_texts"
                    )
                    scores = data.get(
                        "rec_scores"
                    )

                    if not isinstance(
                        texts,
                        list,
                    ):
                        continue

                    for j, text in enumerate(
                        texts
                    ):
                        if text is None:
                            continue

                        text = str(
                            text
                        ).strip()

                        if not text:
                            continue

                        score = None

                        if (
                            isinstance(
                                scores,
                                list,
                            )
                            and j
                            < len(scores)
                        ):
                            try:
                                score = float(
                                    scores[j]
                                )
                            except Exception:
                                score = None

                        lines.append(
                            {
                                "text":
                                    text,
                                "score":
                                    score,
                            }
                        )

                valid_scores = [
                    x["score"]
                    for x in lines
                    if x["score"]
                    is not None
                ]

                record = {
                    "label":
                        str(
                            row["label"]
                        ),
                    "session_id":
                        str(
                            row["session_id"]
                        ),
                    "segment_index":
                        int(
                            row[
                                "segment_index"
                            ]
                        ),
                    "application_variant":
                        str(
                            row[
                                "application_variant"
                            ]
                        ),
                    "screenshot_timestamp":
                        str(
                            row[
                                "screenshot_timestamp"
                            ]
                        ),
                    "screenshot_file":
                        str(
                            image_path
                        ),
                    "ocr_line_count":
                        len(lines),
                    "ocr_mean_score":
                        (
                            sum(
                                valid_scores
                            )
                            / len(
                                valid_scores
                            )
                            if valid_scores
                            else None
                        ),
                    "ocr_text":
                        "\n".join(
                            x["text"]
                            for x in lines
                        ),
                    "ocr_lines":
                        lines,
                    "ocr_status":
                        "ok"
                        if lines
                        else "empty_result",
                }

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                rows.append(
                    {
                        "label":
                            record["label"],
                        "session_id":
                            record[
                                "session_id"
                            ],
                        "segment_index":
                            record[
                                "segment_index"
                            ],
                        "application_variant":
                            record[
                                "application_variant"
                            ],
                        "screenshot_timestamp":
                            record[
                                "screenshot_timestamp"
                            ],
                        "ocr_line_count":
                            record[
                                "ocr_line_count"
                            ],
                        "ocr_mean_score":
                            record[
                                "ocr_mean_score"
                            ],
                        "ocr_status":
                            record[
                                "ocr_status"
                            ],
                    }
                )

            except Exception as exc:

                record = {
                    "label":
                        str(
                            row["label"]
                        ),
                    "session_id":
                        str(
                            row["session_id"]
                        ),
                    "segment_index":
                        int(
                            row[
                                "segment_index"
                            ]
                        ),
                    "application_variant":
                        str(
                            row[
                                "application_variant"
                            ]
                        ),
                    "screenshot_timestamp":
                        str(
                            row[
                                "screenshot_timestamp"
                            ]
                        ),
                    "screenshot_file":
                        str(
                            image_path
                        ),
                    "ocr_line_count":
                        0,
                    "ocr_mean_score":
                        None,
                    "ocr_text":
                        "",
                    "ocr_lines":
                        [],
                    "ocr_status":
                        "error",
                    "ocr_error":
                        str(exc),
                }

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                rows.append(
                    {
                        "label":
                            record["label"],
                        "session_id":
                            record[
                                "session_id"
                            ],
                        "segment_index":
                            record[
                                "segment_index"
                            ],
                        "application_variant":
                            record[
                                "application_variant"
                            ],
                        "screenshot_timestamp":
                            record[
                                "screenshot_timestamp"
                            ],
                        "ocr_line_count":
                            0,
                        "ocr_mean_score":
                            None,
                        "ocr_status":
                            "error",
                    }
                )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OCR_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    nonempty = int(
        (
            summary[
                "ocr_line_count"
            ]
            > 0
        ).sum()
    )

    errors = int(
        (
            summary[
                "ocr_status"
            ]
            == "error"
        ).sum()
    )

    print()
    print("=" * 70)
    print(
        "20% OCR COMPLETE"
    )
    print("=" * 70)

    print(
        f"Selected: {len(selected):,}"
    )
    print(
        f"Non-empty OCR: {nonempty:,}"
    )
    print(
        f"Errors: {errors:,}"
    )

    print()
    print(
        "OCR by family:"
    )
    print(
        summary.groupby(
            "label"
        ).agg(
            screenshots=(
                "label",
                "size",
            ),
            nonempty=(
                "ocr_line_count",
                lambda s:
                    int(
                        (
                            s > 0
                        ).sum()
                    ),
            ),
            mean_lines=(
                "ocr_line_count",
                "mean",
            ),
        ).to_string()
    )

    print()
    print(
        f"Selected sample:\n{SELECTED}"
    )
    print(
        f"Verified sample:\n{SELECTED_VERIFIED}"
    )
    print(
        f"Raw OCR JSONL:\n{OCR_JSONL}"
    )
    print(
        f"OCR summary:\n{OCR_SUMMARY}"
    )


if __name__ == "__main__":
    main()
