
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"

SAMPLE_FILE = OUT / "dataset_b_ocr_sample_75.csv"
OUT_VERIFIED = OUT / "dataset_b_ocr_sample_75_verified_v2.csv"
OUT_MISSING = OUT / "dataset_b_ocr_sample_missing_v2.csv"


def session_dirs(session_id):
    candidates = []

    for root in DATA_ROOT.rglob("dataset_b"):
        candidate = root / session_id
        if candidate.is_dir():
            candidates.append(candidate)

    return sorted(set(candidates))


def choose_session_dir(session_id):
    candidates = session_dirs(session_id)

    if not candidates:
        raise FileNotFoundError(
            f"Dataset B session not found: {session_id}"
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

        total_size = sum(
            p.stat().st_size
            for p in screenshots
            if p.exists()
        )

        scored.append(
            (
                len(screenshots),
                total_size,
                str(path),
                path,
            )
        )

    scored.sort(reverse=True)
    return scored[0][-1]


def build_screenshot_index(session_id):
    session_dir = choose_session_dir(
        session_id
    )

    index = {}

    for path in session_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            continue

        index.setdefault(
            path.name,
            path,
        )

    return session_dir, index


def main():
    print("=" * 70)
    print(
        "FIXED DATASET-B OCR SAMPLE RESOLUTION V2"
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
        "prefix_strength",
        "screenshot_timestamp",
        "screenshot_file",
    }

    missing = required - set(
        sample.columns
    )

    if missing:
        raise RuntimeError(
            "Sample is missing columns: "
            + ", ".join(sorted(missing))
        )

    print(
        f"Sample rows: {len(sample):,}"
    )

    session_cache = {}
    rows = []

    for _, row in sample.iterrows():
        session_id = str(
            row["session_id"]
        )

        if session_id not in session_cache:
            print()
            print(
                f"Indexing screenshots for {session_id}"
            )

            session_dir, index = (
                build_screenshot_index(
                    session_id
                )
            )

            session_cache[
                session_id
            ] = (
                session_dir,
                index,
            )

            print(
                f"  directory: {session_dir}"
            )
            print(
                f"  screenshots indexed: "
                f"{len(index):,}"
            )

        session_dir, index = (
            session_cache[session_id]
        )

        original_ref = str(
            row["screenshot_file"]
        ).strip()

        # The previous sample-builder may have produced an empty reference.
        # The authoritative event schema gives us:
        # payload.file_reference.filename
        # and
        # payload.file_reference.relative_path
        #
        # Since the sample CSV contains the screenshot timestamp, locate
        # the exact screenshot deterministically from its expected filename.
        #
        # The screenshot naming convention is:
        # scr_smart_<timestamp_ms>_monitor_1_post.jpg
        ts = pd.to_datetime(
            row["screenshot_timestamp"],
            utc=True,
        )

        timestamp_ms = int(
            ts.timestamp() * 1000
        )

        expected_names = [
            f"scr_smart_{timestamp_ms}_monitor_1_post.jpg",
            f"scr_smart_{timestamp_ms}_monitor_1_post.jpeg",
            f"scr_smart_{timestamp_ms}_monitor_1_post.png",
            f"scr_smart_{timestamp_ms}_monitor_1_post.webp",
        ]

        resolved = None

        # First use exact expected event-generated filename.
        for name in expected_names:
            if name in index:
                resolved = index[name]
                break

        # Then use the basename from the old reference if it exists.
        if resolved is None and original_ref:
            candidate_name = Path(
                original_ref
            ).name

            if candidate_name in index:
                resolved = index[
                    candidate_name
                ]

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

        verified[
            "resolution_method"
        ] = (
            "timestamp_filename"
            if resolved is not None
            and Path(
                str(resolved)
            ).name
            in expected_names
            else (
                "existing_sample_reference"
                if resolved is not None
                else "unresolved"
            )
        )

        rows.append(
            verified
        )

    verified_df = pd.DataFrame(rows)

    resolved_mask = (
        verified_df[
            "screenshot_file_exists"
        ]
        .astype(int)
        .eq(1)
    )

    resolved_df = verified_df[
        resolved_mask
    ].copy()

    missing_df = verified_df[
        ~resolved_mask
    ].copy()

    verified_df.to_csv(
        OUT_VERIFIED,
        index=False,
        encoding="utf-8-sig",
    )

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
        f"{len(resolved_df):,} / "
        f"{len(verified_df):,}"
    )

    print(
        f"Missing: "
        f"{len(missing_df):,} / "
        f"{len(verified_df):,}"
    )

    if not resolved_df.empty:
        print()
        print(
            "Resolved by process family:"
        )
        print(
            resolved_df[
                "label"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()
        print(
            "Resolved by application variant:"
        )
        print(
            pd.crosstab(
                resolved_df[
                    "label"
                ],
                resolved_df[
                    "application_variant"
                ],
            ).to_string()
        )

        print()
        print(
            "First resolved files:"
        )
        print(
            resolved_df[
                [
                    "label",
                    "session_id",
                    "segment_index",
                    "screenshot_timestamp",
                    "resolved_screenshot_file",
                    "resolution_method",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    if not missing_df.empty:
        print()
        print(
            "Unresolved rows:"
        )
        print(
            missing_df[
                [
                    "label",
                    "session_id",
                    "segment_index",
                    "screenshot_timestamp",
                    "screenshot_file",
                ]
            ]
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

    if len(resolved_df) == len(verified_df):
        print()
        print(
            "STATUS: READY FOR OCR"
        )
    else:
        print()
        print(
            "STATUS: NOT READY FOR OCR"
        )


if __name__ == "__main__":
    main()
