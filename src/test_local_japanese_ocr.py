
from __future__ import annotations

from pathlib import Path
import json
import sys
import traceback

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_FILE = (
    ROOT
    / "outputs"
    / "step2_exact_segment_evidence"
    / "screenshot_evidence.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "step2_ocr_test"
)

OUT_CSV = OUT_DIR / "ocr_test.csv"

TEST_COUNT = 5


def safe_value(obj, key):
    """Support dict-like PaddleOCR results and result objects."""
    try:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)
    except Exception:
        return None


def as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def extract_ocr_result(result):
    """
    Normalize PaddleOCR 3.x / compatible result output into
    text + confidence rows.

    PaddleOCR's newer pipeline exposes recognition arrays such
    as rec_texts and rec_scores. Older-style outputs are also
    handled where possible.
    """
    rows = []

    # New-style result object/dict.
    texts = safe_value(result, "rec_texts")
    scores = safe_value(result, "rec_scores")

    if texts is not None:
        texts = as_list(texts)
        scores = as_list(scores)

        for i, text in enumerate(texts):
            if text is None:
                continue

            text = str(text).strip()
            if not text:
                continue

            score = None
            if i < len(scores):
                try:
                    score = float(scores[i])
                except Exception:
                    score = None

            rows.append(
                {
                    "text": text,
                    "confidence": score,
                }
            )

        if rows:
            return rows

    # Try nested OCR result structures.
    for key in ("res", "ocr_res", "result"):
        nested = safe_value(result, key)

        if nested is not None and nested is not result:
            nested_rows = extract_ocr_result(nested)
            if nested_rows:
                return nested_rows

    # Legacy-style list output.
    if isinstance(result, list):
        for item in result:
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
            ):
                # Legacy shape may contain:
                # [box, [text, score]]
                value = item[1]

                if (
                    isinstance(value, (list, tuple))
                    and len(value) >= 2
                ):
                    text = str(value[0]).strip()
                    try:
                        score = float(value[1])
                    except Exception:
                        score = None

                    if text:
                        rows.append(
                            {
                                "text": text,
                                "confidence": score,
                            }
                        )

            elif isinstance(item, dict):
                nested_rows = extract_ocr_result(item)
                rows.extend(nested_rows)

    return rows


def pick_test_images():
    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing screenshot evidence CSV:\n{EVIDENCE_FILE}"
        )

    df = pd.read_csv(
        EVIDENCE_FILE,
        keep_default_na=False,
    )

    required = {
        "family_id",
        "session_id",
        "segment_index",
        "timestamp",
        "filename",
        "copy_path",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    df["segment_index"] = pd.to_numeric(
        df["segment_index"],
        errors="raise",
    ).astype(int)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        [
            "family_id",
            "session_id",
            "segment_index",
            "timestamp",
            "chunk_number",
            "source_line",
            "reference_number",
        ],
        kind="mergesort",
    )

    # Prefer different families/sessions where possible so the
    # first OCR test does not accidentally sample only one UI.
    selected_rows = []
    seen_sessions = set()

    for _, row in df.iterrows():
        path = Path(row["copy_path"])

        if not path.exists():
            continue

        session = row["session_id"]

        if session in seen_sessions:
            continue

        selected_rows.append(row)
        seen_sessions.add(session)

        if len(selected_rows) >= TEST_COUNT:
            break

    # Fill remaining slots if there were not enough unique sessions.
    if len(selected_rows) < TEST_COUNT:
        chosen_keys = {
            (
                row["session_id"],
                row["segment_index"],
                row["filename"],
            )
            for row in selected_rows
        }

        for _, row in df.iterrows():
            key = (
                row["session_id"],
                row["segment_index"],
                row["filename"],
            )

            if key in chosen_keys:
                continue

            path = Path(row["copy_path"])

            if not path.exists():
                continue

            selected_rows.append(row)

            if len(selected_rows) >= TEST_COUNT:
                break

    return pd.DataFrame(selected_rows)


def create_ocr():
    """
    Use the current PaddleOCR pipeline API when available.

    Japanese OCR language code is set to 'japan'.
    """
    from paddleocr import PaddleOCR

    candidates = [
        {
            "lang": "japan",
            "device": "cpu",
        },
        {
            "lang": "japan",
        },
    ]

    last_error = None

    for kwargs in candidates:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not initialize PaddleOCR.\n"
        f"Last error: {last_error}"
    )


def run_one(ocr, image_path):
    """
    Prefer PaddleOCR 3.x predict().
    Fall back to ocr() for older releases.
    """
    if hasattr(ocr, "predict"):
        return ocr.predict(str(image_path))

    if callable(ocr):
        return ocr(str(image_path))

    raise RuntimeError(
        "Installed PaddleOCR object has neither "
        "predict() nor callable OCR interface."
    )


def main():
    print("=" * 70)
    print("LOCAL JAPANESE OCR — 5 IMAGE TEST")
    print("=" * 70)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = pick_test_images()

    print()
    print(
        f"Test screenshots selected: {len(selected)}"
    )

    if selected.empty:
        raise RuntimeError(
            "No usable screenshot files found."
        )

    print()
    print("Selected images:")

    for i, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        print(
            f"{i}. {row['family_id']} | "
            f"{row['session_id']} | "
            f"segment={row['segment_index']} | "
            f"{row['timestamp']} | "
            f"{row['filename']}"
        )
        print(
            f"   {row['copy_path']}"
        )

    print()
    print("Initializing PaddleOCR...")
    ocr = create_ocr()
    print("PaddleOCR initialized.")

    output_rows = []

    print()
    print("Running OCR...")

    for index, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        image_path = Path(
            row["copy_path"]
        )

        print()
        print(
            f"[{index}/{len(selected)}] "
            f"{image_path.name}"
        )

        try:
            raw_output = run_one(
                ocr,
                image_path,
            )

            # New API can return an iterable of result objects.
            if isinstance(
                raw_output,
                (list, tuple),
            ):
                results = list(raw_output)
            else:
                results = [raw_output]

            extracted = []

            for result in results:
                extracted.extend(
                    extract_ocr_result(
                        result
                    )
                )

            texts = [
                item["text"]
                for item in extracted
            ]

            confidences = [
                item["confidence"]
                for item in extracted
                if item["confidence"] is not None
            ]

            joined_text = "\n".join(
                texts
            )

            mean_confidence = (
                sum(confidences)
                / len(confidences)
                if confidences
                else None
            )

            print(
                f"Detected text boxes: "
                f"{len(extracted)}"
            )

            print(
                f"Mean confidence: "
                f"{mean_confidence:.4f}"
                if mean_confidence is not None
                else "Mean confidence: N/A"
            )

            print("--- OCR TEXT ---")

            if joined_text:
                print(
                    joined_text
                )
            else:
                print(
                    "[NO TEXT DETECTED]"
                )

            print(
                "---------------"
            )

            output_rows.append(
                {
                    "family_id":
                        row["family_id"],

                    "session_id":
                        row["session_id"],

                    "segment_index":
                        int(
                            row[
                                "segment_index"
                            ]
                        ),

                    "timestamp":
                        row["timestamp"],

                    "filename":
                        row["filename"],

                    "image_path":
                        str(image_path),

                    "text_box_count":
                        len(extracted),

                    "mean_confidence":
                        mean_confidence,

                    "ocr_text":
                        joined_text,

                    "status":
                        "OK",
                }
            )

        except Exception as exc:

            print(
                "OCR FAILED:"
            )
            print(
                str(exc)
            )
            traceback.print_exc()

            output_rows.append(
                {
                    "family_id":
                        row["family_id"],

                    "session_id":
                        row["session_id"],

                    "segment_index":
                        int(
                            row[
                                "segment_index"
                            ]
                        ),

                    "timestamp":
                        row["timestamp"],

                    "filename":
                        row["filename"],

                    "image_path":
                        str(image_path),

                    "text_box_count":
                        0,

                    "mean_confidence":
                        None,

                    "ocr_text":
                        "",

                    "status":
                        f"ERROR: {exc}",
                }
            )

    result_df = pd.DataFrame(
        output_rows
    )

    result_df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("OCR TEST COMPLETE")
    print("=" * 70)

    print(
        f"Saved:\n{OUT_CSV}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Review the OCR text against the actual screenshots "
        "before we install or run local translation."
    )


if __name__ == "__main__":
    main()
