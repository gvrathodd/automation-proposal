
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
    / "dataset_b_ocr_raw_75_v2.jsonl"
)

OCR_SUMMARY = (
    OUTPUT_DIR
    / "dataset_b_ocr_summary_v2.csv"
)


def as_dict(obj):
    if isinstance(obj, dict):
        return obj

    # PaddleOCR Result objects expose json as a property.
    try:
        value = obj.json
        if callable(value):
            value = value()

        if isinstance(value, str):
            return json.loads(value)

        if isinstance(value, dict):
            return value
    except Exception:
        pass

    return None


def extract_result_dict(obj):
    """
    PaddleOCR 3.x commonly returns a Result whose JSON looks like:
      {
        "res": {
          "rec_texts": [...],
          "rec_scores": [...]
        }
      }
    Older/direct variants may expose rec_texts at the top level.
    """
    data = as_dict(obj)

    if data is None:
        return None

    if isinstance(
        data.get("res"),
        dict,
    ):
        return data["res"]

    return data


def extract_lines(result_obj):
    data = extract_result_dict(
        result_obj
    )

    if not data:
        return []

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
        return []

    lines = []

    for i, text in enumerate(
        texts
    ):
        score = None

        if (
            isinstance(
                scores,
                list,
            )
            and i < len(scores)
        ):
            try:
                score = float(
                    scores[i]
                )
            except Exception:
                score = None

        if text is None:
            continue

        text = str(text).strip()

        if not text:
            continue

        lines.append(
            {
                "text": text,
                "score": score,
            }
        )

    return lines


def main():
    print("=" * 70)
    print(
        "LOCAL PADDLEOCR — 75 VERIFIED DATASET-B SCREENSHOTS V2"
    )
    print("=" * 70)

    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(
            SAMPLE_FILE
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
            f"{len(bad)} screenshots are not verified."
        )

    missing_files = [
        str(p)
        for p in sample[
            "resolved_screenshot_file"
        ]
        if not Path(str(p)).exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Resolved screenshot paths do not exist. "
            f"Missing: {len(missing_files)}"
        )

    print(
        f"Screenshots to OCR: {len(sample):,}"
    )

    print()
    print(
        "Initializing PaddleOCR..."
    )

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
    ) as output:

        for i, row in sample.iterrows():

            image_path = Path(
                row[
                    "resolved_screenshot_file"
                ]
            )

            print(
                f"[{i + 1:02d}/75] "
                f"{row['label']} | "
                f"{row['application_variant']} | "
                f"segment {int(row['segment_index'])} | "
                f"{image_path.name}"
            )

            try:
                prediction = ocr.predict(
                    str(image_path)
                )

                lines = []

                for result in prediction:
                    parsed = extract_lines(
                        result
                    )
                    lines.extend(
                        parsed
                    )

                scores = [
                    x["score"]
                    for x in lines
                    if x["score"] is not None
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

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                summary_rows.append(
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

                if not lines:
                    print(
                        "  WARNING: OCR returned no recognized text."
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

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                summary_rows.append(
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

    empty = int(
        (
            summary[
                "ocr_status"
            ]
            == "empty_result"
        ).sum()
    )

    print()
    print("=" * 70)
    print(
        "OCR V2 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Processed: {len(summary):,}"
    )

    print(
        f"Non-empty OCR results: {nonempty:,}"
    )

    print(
        f"Empty OCR results: {empty:,}"
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
        "Raw OCR:"
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

    if nonempty < len(sample):
        print()
        print(
            "WARNING: Not every screenshot produced text. "
            "Inspect the summary before semantic interpretation."
        )


if __name__ == "__main__":
    main()
