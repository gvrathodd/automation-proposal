
from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OCR_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_ocr_raw_20pct.jsonl"
)


def is_japanese_char(ch: str) -> bool:
    code = ord(ch)

    return (
        0x3040 <= code <= 0x309F   # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0x3400 <= code <= 0x4DBF   # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF   # CJK Unified Ideographs
        or 0xF900 <= code <= 0xFAFF   # CJK Compatibility Ideographs
        or 0xFF66 <= code <= 0xFF9F   # Half-width Katakana
    )


def main():
    if not OCR_FILE.exists():
        raise FileNotFoundError(
            f"OCR file not found:\n{OCR_FILE}"
        )

    total_records = 0
    total_all_characters = 0
    total_japanese_characters = 0
    total_non_japanese_characters = 0
    records_with_japanese = 0

    japanese_by_family = {}

    with OCR_FILE.open(
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
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at line {line_number}: {exc}"
                )

            total_records += 1

            text = str(
                record.get(
                    "ocr_text",
                    ""
                )
                or ""
            )

            japanese_count = sum(
                1
                for ch in text
                if is_japanese_char(ch)
            )

            total_all_characters += len(text)
            total_japanese_characters += japanese_count
            total_non_japanese_characters += (
                len(text) - japanese_count
            )

            if japanese_count > 0:
                records_with_japanese += 1

            family = str(
                record.get(
                    "label",
                    "unknown",
                )
                or "unknown"
            )

            japanese_by_family[
                family
            ] = (
                japanese_by_family.get(
                    family,
                    0,
                )
                + japanese_count
            )

    print("=" * 72)
    print(
        "DATASET-B OCR JAPANESE CHARACTER COUNT"
    )
    print("=" * 72)

    print(
        f"OCR file: {OCR_FILE}"
    )
    print(
        f"OCR records: {total_records:,}"
    )
    print(
        f"Total OCR characters: {total_all_characters:,}"
    )
    print(
        f"Japanese characters: {total_japanese_characters:,}"
    )
    print(
        f"Non-Japanese characters: "
        f"{total_non_japanese_characters:,}"
    )
    print(
        f"Records containing Japanese: "
        f"{records_with_japanese:,}"
        f" / {total_records:,}"
    )

    print()
    print(
        "Japanese characters by process family:"
    )

    for family, count in sorted(
        japanese_by_family.items()
    ):
        print(
            f"  {family:10s} {count:10,}"
        )

    print()
    print(
        "Google 500,000-character free-tier check:"
    )

    if total_japanese_characters <= 500_000:
        remaining = (
            500_000
            - total_japanese_characters
        )
        print(
            f"  WITHIN 500,000 characters "
            f"(remaining: {remaining:,})"
        )
    else:
        excess = (
            total_japanese_characters
            - 500_000
        )
        print(
            f"  EXCEEDS 500,000 characters "
            f"by {excess:,}"
        )

    print()
    print(
        "Note: this counts Japanese Unicode characters "
        "inside the OCR text. It does not count screenshot files "
        "or pixels, and it does not yet estimate API request overhead."
    )


if __name__ == "__main__":
    main()
