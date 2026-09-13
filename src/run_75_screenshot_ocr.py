
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

try:
    from paddleocr import PaddleOCR
except Exception as exc:
    print("Could not import PaddleOCR.")
    print(exc)
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SAMPLE_FILE = (
    OUTPUT_DIR
    / "dataset_b_ocr_sample_75_verified_v2.csv"
)

OCR_OUTPUT = (
    OUTPUT_DIR
    / "dataset_b_ocr_raw_75.jsonl"
)

OCR_SUMMARY = (
    OUTPUT_DIR
    / "dataset_b_ocr_summary.csv"
)


def normalize_result_lines(result):
    """
    Handle PaddleOCR 3.x result objects without assuming a single
    exact return shape.
    """
    lines = []

    if result is None:
        return lines

    # New PaddleOCR result object commonly exposes json.
    if hasattr(result, "json"):
        try:
            obj = result.json
            if callable(obj):
                obj = obj()

            if isinstance(obj, str):
                obj = json.loads(obj)

            if isinstance(obj, list):
                for item in obj:
                    lines.extend(
                        normalize_result_lines(item)
                    )
                return lines

            if isinstance(obj, dict):
                result_dict = obj
            else:
                result_dict = None
        except Exception:
            result_dict = None
    else:
        result_dict = None

    if result_dict is None and isinstance(
        result, dict
    ):
        result_dict = result

    if result_dict is None:
        return lines

    rec_texts = result_dict.get(
        "rec_texts"
    )
    rec_scores = result_dict.get(
        "rec_scores"
    )

    if isinstance(rec_texts, list):
        for i, text in enumerate(
            rec_texts
        ):
            score = None

            if (
                isinstance(
                    rec_scores,
                    list,
                )
                and i < len(rec_scores)
            ):
                score = rec_scores[i]

            lines.append(
                {
                    "text": str(text),
                    "score": (
                        float(score)
                        if score is not None
                        else None
                    ),
                }
            )

        return lines

    # Older-style nested PaddleOCR output.
    for value in result_dict.values():
        if isinstance(
            value,
            (list, tuple),
        ):
            for item in value:
                if (
                    isinstance(
                        item,
                        (list, tuple),
                    )
                    and len(item) >= 2
                ):
                    maybe_text = item[1]

                    if isinstance(
                        maybe_text,
                        (list, tuple),
                    ) and len(maybe_text) >= 1:
                        text = maybe_text[0]
                        score = (
                            maybe_text[1]
                            if len(maybe_text) > 1
                            else None
                        )

                        lines.append(
                            {
                                "text": str(text),
                                "score": (
                                    float(score)
                                    if score is not None
                                    else None
                                ),
                            }
                        )

    return lines


def main():
    print("=" * 70)
    print(
        "LOCAL PADDLEOCR — 75 VERIFIED DATASET-B SCREENSHOTS"
    )
    print("=" * 70)

    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(
            f"Verified OCR sample not found:\n{SAMPLE_FILE}"
        )

    sample = pd.read_csv(
        SAMPLE_FILE,
        keep_default_na=False,
    )

    required = {
        "label",
        "session_id",
        "segment_index",
        "application_variant",
        "screenshot_timestamp",
        "resolved_screenshot_file",
        "screenshot_file_exists",
    }

    missing = required - set(
        sample.columns
    )

    if missing:
        raise RuntimeError(
            "Verified sample is missing columns: "
            + ", ".join(sorted(missing))
        )

    bad = sample[
        sample[
            "screenshot_file_exists"
        ].astype(int) != 1
    ]

    if not bad.empty:
        raise RuntimeError(
            f"{len(bad)} screenshot paths are not verified."
        )

    image_paths = sample[
        "resolved_screenshot_file"
    ].astype(str)

    missing_files = [
        p
        for p in image_paths
        if not Path(p).exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Verified paths do not exist on this machine. "
            f"Missing: {len(missing_files)}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Screenshots to OCR: {len(sample):,}"
    )

    print()
    print(
        "Initializing PaddleOCR..."
    )

    # CPU mode. This matches the local setup used for the earlier
    # PaddleOCR work and avoids requiring GPU configuration.
    ocr = PaddleOCR(
        lang="japan",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print(
        "PaddleOCR initialized."
    )

    summary_rows = []

    with OCR_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as out:

        for i, row in sample.iterrows():

            image_path = Path(
                row[
                    "resolved_screenshot_file"
                ]
            )

            print(
                f"[{i + 1:02d}/75] "
                f"{row['label']} | "
                f"{row['session_id']} | "
                f"segment {int(row['segment_index'])} | "
                f"{image_path.name}"
            )

            try:
                result = ocr.predict(
                    str(image_path)
                )

                lines = []

                if result is not None:
                    try:
                        for item in result:
                            lines.extend(
                                normalize_result_lines(
                                    item
                                )
                            )
                    except TypeError:
                        lines.extend(
                            normalize_result_lines(
                                result
                            )
                        )

                text = "\n".join(
                    item["text"]
                    for item in lines
                    if item.get("text")
                )

                scores = [
                    item["score"]
                    for item in lines
                    if item.get("score")
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
                            row["segment_index"]
                        ),
                    "application_variant":
                        str(
                            row[
                                "application_variant"
                            ]
                        ),
                    "prefix_strength":
                        int(
                            row.get(
                                "prefix_strength",
                                0,
                            )
                            or 0
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
                            sum(scores)
                            / len(scores)
                            if scores
                            else None
                        ),
                    "ocr_text":
                        text,
                    "ocr_lines":
                        lines,
                    "ocr_status":
                        "ok",
                }

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                summary_rows.append(
                    {
                        "label":
                            record[
                                "label"
                            ],
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
                            "ok",
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
                            row["segment_index"]
                        ),
                    "application_variant":
                        str(
                            row[
                                "application_variant"
                            ]
                        ),
                    "prefix_strength":
                        int(
                            row.get(
                                "prefix_strength",
                                0,
                            )
                            or 0
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

                summary_rows.append(
                    {
                        "label":
                            record[
                                "label"
                            ],
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

                print(
                    f"  OCR ERROR: {exc}"
                )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OCR_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    ok = int(
        (
            summary[
                "ocr_status"
            ]
            == "ok"
        ).sum()
    )

    errors = len(summary) - ok

    print()
    print("=" * 70)
    print(
        "OCR COMPLETE"
    )
    print("=" * 70)

    print(
        f"Processed: {len(summary):,}"
    )

    print(
        f"Successful: {ok:,}"
    )

    print(
        f"Errors: {errors:,}"
    )

    print()
    print(
        "By process family:"
    )

    print(
        summary.groupby(
            "label"
        ).agg(
            screenshots=(
                "label",
                "size",
            ),
            successful=(
                "ocr_status",
                lambda s:
                    int(
                        (
                            s == "ok"
                        ).sum()
                    ),
            ),
        ).to_string()
    )

    print()
    print(
        "Raw OCR JSONL:"
    )
    print(
        OCR_OUTPUT
    )

    print(
        "OCR summary:"
    )
    print(
        OCR_SUMMARY
    )

    print()
    print(
        "No translation or semantic interpretation was performed."
    )


if __name__ == "__main__":
    main()
