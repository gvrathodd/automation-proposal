
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from google.oauth2 import service_account
from google.cloud import translate_v2 as translate


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OCR_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_ocr_raw_20pct.jsonl"
)

KEY_FILE = Path(
    os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "",
    )
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_ocr_translated_20pct.jsonl"
)

CACHE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_ocr_translation_cache.json"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_b_ocr_translation_summary.json"
)


JAPANESE_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uF900-\uFAFF\uFF66-\uFF9F]"
)


def contains_japanese(text: str) -> bool:
    return bool(
        JAPANESE_RE.search(
            text
        )
    )


def extract_japanese_lines(text: str):
    """
    Keep line boundaries where possible. We only submit OCR lines that
    contain Japanese characters, avoiding unnecessary translation of
    already-English/technical OCR text.
    """
    lines = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if contains_japanese(line):
            lines.append(line)

    return lines


def load_jsonl():
    rows = []

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
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at line {line_number}: {exc}"
                )

    return rows


def load_cache():
    if not CACHE_FILE.exists():
        return {}

    try:
        data = json.loads(
            CACHE_FILE.read_text(
                encoding="utf-8"
            )
        )

        return (
            data
            if isinstance(
                data,
                dict,
            )
            else {}
        )

    except Exception:
        return {}


def save_cache(cache):
    CACHE_FILE.write_text(
        json.dumps(
            cache,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_client():
    if not KEY_FILE.exists():
        raise FileNotFoundError(
            "Google service-account JSON key not found.\n"
            "Set GOOGLE_APPLICATION_CREDENTIALS first, e.g.:\n"
            '$env:GOOGLE_APPLICATION_CREDENTIALS="C:\\path\\to\\key.json"'
        )

    # Reading the JSON here lets us fail early with a useful error.
    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(KEY_FILE)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Google service-account key: {exc}"
        )

    # Cloud Translation Basic accepts authenticated client requests.
    client = translate.Client(
        credentials=credentials
    )

    return client


def translate_one(
    client,
    text,
    attempts=4,
):
    """
    Translate one Japanese OCR line.
    We deliberately use literal NMT translation, not LLM interpretation.
    """
    last_error = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            result = client.translate(
                text,
                target_language="en",
                source_language="ja",
                model="nmt",
            )

            return str(
                result.get(
                    "translatedText",
                    "",
                )
            )

        except Exception as exc:
            last_error = exc

            if attempt < attempts:
                sleep_seconds = (
                    2 ** (attempt - 1)
                )
                time.sleep(
                    sleep_seconds
                )

    raise RuntimeError(
        f"Translation failed after {attempts} attempts: "
        f"{last_error}"
    )


def main():
    print("=" * 78)
    print(
        "DATASET-B FULL OCR JAPANESE → ENGLISH TRANSLATION"
    )
    print("=" * 78)

    if not OCR_FILE.exists():
        raise FileNotFoundError(
            OCR_FILE
        )

    rows = load_jsonl()

    print(
        f"OCR records: {len(rows):,}"
    )

    # Build unique Japanese-line inventory.
    unique_lines = []
    seen = set()

    total_japanese_chars = 0
    records_with_japanese = 0

    for record in rows:
        text = str(
            record.get(
                "ocr_text",
                "",
            )
            or ""
        )

        lines = extract_japanese_lines(
            text
        )

        if lines:
            records_with_japanese += 1

        for line in lines:
            total_japanese_chars += sum(
                1
                for ch in line
                if contains_japanese(ch)
            )

            if line not in seen:
                seen.add(line)
                unique_lines.append(
                    line
                )

    print(
        f"Records containing Japanese: "
        f"{records_with_japanese:,}"
    )

    print(
        f"Unique Japanese OCR lines: "
        f"{len(unique_lines):,}"
    )

    print(
        f"Japanese characters to translate: "
        f"{total_japanese_chars:,}"
    )

    cache = load_cache()

    cached = sum(
        1
        for line in unique_lines
        if line in cache
        and cache[line]
    )

    pending = [
        line
        for line in unique_lines
        if not cache.get(line)
    ]

    print(
        f"Already cached translations: "
        f"{cached:,}"
    )

    print(
        f"New lines requiring API calls: "
        f"{len(pending):,}"
    )

    if total_japanese_chars > 500_000:
        print(
            "WARNING: Japanese character count exceeds "
            "the 500,000-character free allowance."
        )

    if pending:
        print()
        print(
            "Initializing Google Cloud Translation..."
        )

        client = create_client()

        print(
            "Google Translation client initialized."
        )

        for i, line in enumerate(
            pending,
            start=1,
        ):
            if (
                i == 1
                or i % 25 == 0
                or i == len(pending)
            ):
                print(
                    f"[{i}/{len(pending)}] "
                    f"Translating unique Japanese OCR lines..."
                )

            try:
                translated = translate_one(
                    client,
                    line,
                )

                cache[line] = translated

            except Exception as exc:
                print(
                    f"ERROR translating line: {line}"
                )
                print(
                    f"  {exc}"
                )

                # Stop rather than silently producing a partial
                # semantic dataset. Cache remains checkpointed.
                save_cache(cache)
                raise

            # Checkpoint every 25 translations.
            if i % 25 == 0:
                save_cache(cache)

        save_cache(cache)

    # Build the full translated OCR JSONL while preserving the original OCR.
    translated_records = []

    translation_missing = 0
    translated_line_count = 0

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as out:

        for record in rows:
            original_ocr = str(
                record.get(
                    "ocr_text",
                    "",
                )
                or ""
            )

            japanese_lines = extract_japanese_lines(
                original_ocr
            )

            translated_lines = []

            for line in japanese_lines:
                translation = cache.get(
                    line,
                    "",
                )

                if translation:
                    translated_lines.append(
                        {
                            "japanese":
                                line,
                            "english":
                                translation,
                        }
                    )
                    translated_line_count += 1
                else:
                    translation_missing += 1

            enriched = dict(
                record
            )

            enriched[
                "japanese_ocr_lines"
            ] = [
                {
                    "japanese":
                        item[
                            "japanese"
                        ],
                    "english":
                        item[
                            "english"
                        ],
                }
                for item
                in translated_lines
            ]

            enriched[
                "english_translation"
            ] = "\n".join(
                item[
                    "english"
                ]
                for item
                in translated_lines
            )

            enriched[
                "translation_status"
            ] = (
                "complete"
                if (
                    not japanese_lines
                    or len(translated_lines)
                    == len(japanese_lines)
                )
                else "partial"
            )

            out.write(
                json.dumps(
                    enriched,
                    ensure_ascii=False,
                )
                + "\n"
            )

            translated_records.append(
                enriched
            )

    # Summary
    records_with_translation = sum(
        1
        for r in translated_records
        if r.get(
            "english_translation",
            "",
        ).strip()
    )

    summary = {
        "source_ocr_file":
            str(OCR_FILE),
        "output_translated_file":
            str(OUTPUT_FILE),
        "ocr_records":
            len(rows),
        "records_containing_japanese":
            records_with_japanese,
        "unique_japanese_lines":
            len(unique_lines),
        "japanese_characters_processed":
            total_japanese_chars,
        "cached_translation_lines_before_run":
            cached,
        "new_translation_api_calls":
            len(pending),
        "translation_cache_size":
            len(cache),
        "translated_japanese_lines":
            translated_line_count,
        "missing_translation_lines":
            translation_missing,
        "records_with_english_translation":
            records_with_translation,
        "translation_complete":
            translation_missing == 0,
        "source_language":
            "ja",
        "target_language":
            "en",
        "translation_engine":
            "Google Cloud Translation Basic / NMT",
        "note":
            "Japanese originals are preserved; English is an analytical translation only.",
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "TRANSLATION COMPLETE"
    )
    print("=" * 78)

    print(
        f"Records: {len(rows):,}"
    )
    print(
        f"Records containing Japanese: "
        f"{records_with_japanese:,}"
    )
    print(
        f"Unique Japanese lines: "
        f"{len(unique_lines):,}"
    )
    print(
        f"Japanese characters processed: "
        f"{total_japanese_chars:,}"
    )
    print(
        f"New translation API calls: "
        f"{len(pending):,}"
    )
    print(
        f"Translated lines: "
        f"{translated_line_count:,}"
    )
    print(
        f"Missing translations: "
        f"{translation_missing:,}"
    )
    print(
        f"Records with English translation: "
        f"{records_with_translation:,}"
    )

    print()
    print(
        "Outputs:"
    )
    print(
        OUTPUT_FILE
    )
    print(
        CACHE_FILE
    )
    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
