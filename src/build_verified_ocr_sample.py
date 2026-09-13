
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"

SAMPLE_FILE = OUT / "dataset_b_ocr_sample_75.csv"
OUT_VERIFIED = OUT / "dataset_b_ocr_sample_75_verified.csv"
OUT_MISSING = OUT / "dataset_b_ocr_sample_missing.csv"


def session_dirs(session_id):
    dirs = []

    for root in DATA_ROOT.rglob("dataset_b"):
        p = root / session_id
        if p.is_dir():
            dirs.append(p)

    return sorted(set(dirs))


def build_screenshot_index(session_id):
    """
    Build a basename -> actual screenshot path index for one B session.
    The event schema stores references as:
        payload.file_reference.relative_path
    e.g.
        screenshots/scr_smart_....jpg
    """
    dirs = session_dirs(session_id)

    if not dirs:
        return {}

    scored = []

    for session_dir in dirs:
        screenshots = [
            p
            for p in session_dir.rglob("*")
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
                str(session_dir),
                session_dir,
                screenshots,
            )
        )

    scored.sort(reverse=True)

    screenshots = scored[0][-1]

    index = {}

    for path in screenshots:
        index.setdefault(
            path.name,
            path,
        )

    return index


def main():
    print("=" * 70)
    print(
        "VERIFYING DATASET-B OCR SAMPLE SCREENSHOT FILES"
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
        "screenshot_file",
    }

    missing = required - set(
        sample.columns
    )

    if missing:
        raise RuntimeError(
            "OCR sample is missing columns: "
            + ", ".join(sorted(missing))
        )

    print(
        f"Sample rows: {len(sample):,}"
    )

    # Cache session screenshot indexes.
    index_cache = {}

    verified_rows = []
    missing_rows = []

    for i, row in sample.iterrows():

        session_id = str(
            row["session_id"]
        )

        if session_id not in index_cache:
            print(
                f"Indexing screenshots for "
                f"{session_id}"
            )
            index_cache[
                session_id
            ] = build_screenshot_index(
                session_id
            )
            print(
                f"  screenshots indexed: "
                f"{len(index_cache[session_id]):,}"
            )

        index = index_cache[
            session_id
        ]

        candidate = str(
            row["screenshot_file"]
        ).strip()

        resolved = None

        if candidate:
            p = Path(candidate)

            # Absolute path.
            if p.is_absolute() and p.exists():
                resolved = p

            # Path already relative to project/session.
            if resolved is None:
                for base in session_dirs(
                    session_id
                ):
                    candidate_path = (
                        base / candidate
                    )

                    if candidate_path.exists():
                        resolved = candidate_path
                        break

            # Basename lookup from the screenshot index.
            if resolved is None:
                resolved = index.get(
                    p.name
                )

        verified = row.to_dict()

        verified[
            "resolved_screenshot_file"
        ] = (
            str(resolved.resolve())
            if resolved is not None
            else ""
        )

        verified[
            "screenshot_file_exists"
        ] = int(
            resolved is not None
            and resolved.exists()
        )

        if resolved is not None:
            verified_rows.append(
                verified
            )
        else:
            missing_rows.append(
                verified
            )

    verified_df = pd.DataFrame(
        verified_rows
    )
    missing_df = pd.DataFrame(
        missing_rows
    )

    # Preserve the full sample, adding resolution status.
    all_rows = pd.DataFrame(
        [
            r
            for r in (
                verified_rows
                + missing_rows
            )
        ]
    )

    # Restore original sample order using segment/time ordering.
    if not all_rows.empty:
        all_rows = all_rows.sort_values(
            [
                "label",
                "session_id",
                "segment_index",
                "screenshot_timestamp",
            ]
        ).reset_index(drop=True)

    all_rows.to_csv(
        OUT_VERIFIED,
        index=False,
        encoding="utf-8-sig",
    )

    if missing_df.empty:
        # Still create a valid empty file with the same columns.
        pd.DataFrame(
            columns=all_rows.columns
        ).to_csv(
            OUT_MISSING,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        missing_df.to_csv(
            OUT_MISSING,
            index=False,
            encoding="utf-8-sig",
        )

    print()
    print("=" * 70)
    print(
        "VERIFICATION RESULT"
    )
    print("=" * 70)

    print(
        f"Resolved: "
        f"{len(verified_df):,} / {len(sample):,}"
    )

    print(
        f"Missing: "
        f"{len(missing_df):,} / {len(sample):,}"
    )

    print()
    print(
        "Resolved by process family:"
    )

    if not verified_df.empty:
        print(
            verified_df[
                "label"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )
    else:
        print("none")

    print()
    print(
        "Resolved by application variant:"
    )

    if not verified_df.empty:
        print(
            pd.crosstab(
                verified_df["label"],
                verified_df[
                    "application_variant"
                ],
            ).to_string()
        )
    else:
        print("none")

    print()
    print(
        "Sample resolved file examples:"
    )

    if not verified_df.empty:
        print(
            verified_df[
                [
                    "label",
                    "session_id",
                    "segment_index",
                    "application_variant",
                    "screenshot_file",
                    "resolved_screenshot_file",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print()
    print(
        "Outputs:"
    )
    print(
        OUT_VERIFIED
    )
    print(
        OUT_MISSING
    )

    if len(verified_df) == len(sample):
        print()
        print(
            "STATUS: READY FOR OCR"
        )
    else:
        print()
        print(
            "STATUS: NOT READY FOR OCR"
        )
        print(
            "Inspect the missing rows before running OCR."
        )


if __name__ == "__main__":
    main()
