
from __future__ import annotations

from pathlib import Path
import pandas as pd
import re
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]

OCR_FILE = (
    ROOT
    / "outputs"
    / "step2_ocr_test"
    / "ocr_test.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "step2_translation_test"
)

OUT_FILE = (
    OUT_DIR
    / "translation_test.csv"
)

MODEL_NAME = "Helsinki-NLP/opus-mt-ja-en"

BATCH_SIZE = 4


def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Collapse only excessive whitespace. Preserve Japanese/English
    # punctuation and line structure as much as possible.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def is_probably_translatable(text):
    """
    Avoid sending obviously non-linguistic fragments into the
    translation model. This is only a routing heuristic.
    """
    if not text:
        return False

    # Japanese scripts.
    if re.search(
        r"[\u3040-\u309F\u30A0-\u30FF\u3400-\u9FFF]",
        text,
    ):
        return True

    return False


def split_for_model(text, max_chars=450):
    """
    Split long OCR output into manageable chunks while preserving
    line boundaries where possible.
    """
    text = clean_text(text)

    if len(text) <= max_chars:
        return [text]

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chunks = []
    current = ""

    for line in lines:

        if not current:
            current = line
            continue

        candidate = current + "\n" + line

        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = line

    if current:
        chunks.append(current)

    # If one individual line is still huge, hard-split it.
    final_chunks = []

    for chunk in chunks:

        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue

        for start in range(
            0,
            len(chunk),
            max_chars,
        ):
            final_chunks.append(
                chunk[start:start + max_chars]
            )

    return final_chunks


def load_model():
    from transformers import MarianMTModel, MarianTokenizer

    print()
    print(
        f"Loading local translation model:\n"
        f"{MODEL_NAME}"
    )

    tokenizer = MarianTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = MarianMTModel.from_pretrained(
        MODEL_NAME
    )

    # Translation is intentionally CPU-only here. This keeps the
    # pipeline fully local and avoids assuming a GPU is configured.
    model.eval()

    print("Translation model loaded locally.")

    return tokenizer, model


def translate_text(
    text,
    tokenizer,
    model,
):
    text = clean_text(text)

    if not text:
        return ""

    if not is_probably_translatable(text):
        return ""

    chunks = split_for_model(text)

    translated_chunks = []

    import torch

    with torch.no_grad():

        for chunk in chunks:

            batch = tokenizer(
                [chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            generated = model.generate(
                **batch,
                max_length=512,
            )

            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )

            if decoded:
                translated_chunks.append(
                    decoded[0].strip()
                )

    return "\n".join(
        part
        for part in translated_chunks
        if part
    )


def main():
    print("=" * 70)
    print("LOCAL JAPANESE → ENGLISH TRANSLATION TEST")
    print("=" * 70)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not OCR_FILE.exists():
        raise FileNotFoundError(
            f"OCR test output not found:\n{OCR_FILE}"
        )

    df = pd.read_csv(
        OCR_FILE,
        keep_default_na=False,
    )

    required = {
        "family_id",
        "session_id",
        "segment_index",
        "timestamp",
        "filename",
        "image_path",
        "ocr_text",
        "status",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing columns in OCR CSV:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    print(
        f"OCR rows loaded: {len(df):,}"
    )

    tokenizer, model = load_model()

    output_rows = []

    print()
    print("Translating OCR outputs...")

    for index, row in df.iterrows():

        ocr_text = clean_text(
            row["ocr_text"]
        )

        print()
        print(
            f"[{index + 1}/{len(df)}] "
            f"{row['filename']}"
        )

        if not ocr_text:
            print(
                "No OCR text. Skipping translation."
            )

            output_rows.append(
                {
                    "family_id":
                        row["family_id"],
                    "session_id":
                        row["session_id"],
                    "segment_index":
                        int(
                            row["segment_index"]
                        ),
                    "timestamp":
                        row["timestamp"],
                    "filename":
                        row["filename"],
                    "image_path":
                        row["image_path"],
                    "ocr_text_ja":
                        "",
                    "translation_en":
                        "",
                    "translation_status":
                        "NO_TEXT",
                }
            )

            continue

        if not is_probably_translatable(
            ocr_text
        ):
            print(
                "No Japanese detected in OCR text. "
                "Keeping original without translation."
            )

            output_rows.append(
                {
                    "family_id":
                        row["family_id"],
                    "session_id":
                        row["session_id"],
                    "segment_index":
                        int(
                            row["segment_index"]
                        ),
                    "timestamp":
                        row["timestamp"],
                    "filename":
                        row["filename"],
                    "image_path":
                        row["image_path"],
                    "ocr_text_ja":
                        ocr_text,
                    "translation_en":
                        "",
                    "translation_status":
                        "NO_JAPANESE_DETECTED",
                }
            )

            continue

        try:
            translated = translate_text(
                ocr_text,
                tokenizer,
                model,
            )

            print(
                "--- JAPANESE OCR ---"
            )
            print(
                ocr_text
            )

            print(
                "--- ENGLISH TRANSLATION ---"
            )
            print(
                translated
                if translated
                else "[NO TRANSLATION]"
            )

            output_rows.append(
                {
                    "family_id":
                        row["family_id"],
                    "session_id":
                        row["session_id"],
                    "segment_index":
                        int(
                            row["segment_index"]
                        ),
                    "timestamp":
                        row["timestamp"],
                    "filename":
                        row["filename"],
                    "image_path":
                        row["image_path"],
                    "ocr_text_ja":
                        ocr_text,
                    "translation_en":
                        translated,
                    "translation_status":
                        "OK",
                }
            )

        except Exception as exc:

            print(
                "TRANSLATION FAILED:"
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
                            row["segment_index"]
                        ),
                    "timestamp":
                        row["timestamp"],
                    "filename":
                        row["filename"],
                    "image_path":
                        row["image_path"],
                    "ocr_text_ja":
                        ocr_text,
                    "translation_en":
                        "",
                    "translation_status":
                        f"ERROR: {exc}",
                }
            )

    output = pd.DataFrame(
        output_rows
    )

    output.to_csv(
        OUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("TRANSLATION TEST COMPLETE")
    print("=" * 70)

    print(
        f"Saved:\n{OUT_FILE}"
    )

    print()
    print(
        "The English output is a machine translation, "
        "not yet the final business-process interpretation."
    )

    print(
        "We will validate several translations against the "
        "screenshots before processing the full 137-image set."
    )


if __name__ == "__main__":
    main()
