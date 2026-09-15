
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SEGMENTS = ROOT / "segments_v2.jsonl"
EVIDENCE = OUT / "step2_layer3_authoritative_segment_evidence.csv"
TAXONOMY = OUT / "step2_layer5_conservative_process_taxonomy_v1.csv"

FAMILY_OUT = OUT / "step2_layer7_all540_family_quantification.csv"
CASETYPE_OUT = OUT / "step2_layer7_observed_case_type_quantification.csv"


FAMILIES = ["pi", "la", "ob", "si", "rt", "other"]


def load_segments():
    rows = []
    with SEGMENTS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    if len(df) != 540:
        raise RuntimeError(
            f"Expected 540 segments; found {len(df)}"
        )

    df["segment_index"] = (
        df.groupby("session_id").cumcount() + 1
    )

    df["segment_key"] = (
        df["session_id"].astype(str)
        + "::"
        + df["segment_index"].astype(str)
    )

    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)

    return df


def load_evidence():
    df = pd.read_csv(
        EVIDENCE,
        keep_default_na=False,
    )

    df["segment_key"] = (
        df["session_id"].astype(str)
        + "::"
        + df["segment_index"].astype(str)
    )

    return df


def load_taxonomy():
    if not TAXONOMY.exists():
        return pd.DataFrame()

    return pd.read_csv(
        TAXONOMY,
        keep_default_na=False,
    )


def split_set(value):
    return {
        x.strip()
        for x in str(value or "").split("|")
        if x.strip()
    }


def contains_app(value, needles):
    text = str(value or "").lower()
    return any(
        needle.lower() in text
        for needle in needles
    )


def has_excel(value):
    return contains_app(
        value,
        ["excel", "excel.exe", "xlsx", ".xls"],
    )


def has_word(value):
    return contains_app(
        value,
        ["word", "winword", ".doc", ".docx"],
    )


def has_notepad(value):
    return contains_app(
        value,
        ["notepad", ".txt"],
    )


def has_browser(value):
    text = str(value or "").lower()
    return any(
        x in text
        for x in [
            "msedge",
            "microsoft edge",
            "browser",
        ]
    )


def browser_events(row):
    cols = [
        "browser_event_count",
        "browser_click_count",
        "browser_form_input_count",
        "browser_navigation_count",
    ]
    return sum(
        int(float(row.get(c, 0) or 0))
        for c in cols
        if c in row
    )


def semantic_case_map(taxonomy):
    """
    Return family -> case type -> direct OCR segment count.
    These counts are NOT extrapolated to all 540 segments.
    """
    if taxonomy.empty:
        return {}

    if "classification" not in taxonomy.columns:
        return {}

    usable = taxonomy[
        taxonomy["classification"]
        == "OBSERVED_CASE_TYPE"
    ].copy()

    result = {}

    for family, group in usable.groupby("family"):
        result[family] = {
            str(row["candidate_process"]):
                int(row["direct_ocr_segments"])
            for _, row in group.iterrows()
        }

    return result


def case_segments_from_taxonomy():
    """
    Use the conservative evidence file if available to compute exact observed
    direct-OCR segment membership for each case type.
    """
    evidence_file = (
        OUT / "step2_layer5_conservative_process_evidence_v1.csv"
    )

    if not evidence_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        evidence_file,
        keep_default_na=False,
    )

    return df


def main():
    print("=" * 80)
    print(
        "STEP 2 — LAYER 7 QUANTIFY ALL 540 SEGMENTS"
    )
    print("=" * 80)

    segments = load_segments()
    evidence = load_evidence()
    taxonomy = load_taxonomy()
    case_evidence = case_segments_from_taxonomy()

    df = segments.merge(
        evidence,
        on=[
            "segment_key",
            "session_id",
            "segment_index",
            "label",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(df) != 540:
        raise RuntimeError(
            "Layer-7 join did not preserve all 540 segments."
        )

    # Normalize important numeric columns.
    numeric_cols = [
        "event_count",
        "duration_seconds",
        "clipboard_change_count",
        "browser_click_count",
        "browser_form_input_count",
        "browser_navigation_count",
        "app_switch_count",
        "window_title_change_count",
        "screenshot_count",
        "browser_event_count",
        "dom_value_count",
        "url_count",
        "app_count",
        "window_count",
        "operator_count",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    # Exact family-level workload metrics across every 540 segment.
    rows = []
    total_duration = float(
        df["duration_seconds"].sum()
    )

    for family in FAMILIES:
        fam = df[
            df["label"] == family
        ].copy()

        if fam.empty:
            continue

        # Application involvement counts are segment-level:
        # a segment may involve multiple tools, so these do not sum to 1.
        word_segments = int(
            fam["apps"].map(has_word).sum()
        )
        excel_segments = int(
            fam["apps"].map(has_excel).sum()
        )
        notepad_segments = int(
            fam["apps"].map(has_notepad).sum()
        )
        browser_app_segments = int(
            fam["apps"].map(has_browser).sum()
        )

        browser_interacting_segments = 0
        if "browser_event_count" in fam.columns:
            browser_interacting_segments = int(
                (
                    fam["browser_event_count"]
                    > 0
                ).sum()
            )

        clipboard_segments = int(
            (
                fam["clipboard_change_count"]
                > 0
            ).sum()
        )

        field_evidence_segments = int(
            (
                fam.get(
                    "dom_value_count",
                    pd.Series(
                        0,
                        index=fam.index,
                    ),
                )
                > 0
            ).sum()
        )

        rows.append(
            {
                "process_family": family,
                "execution_count_segments": len(fam),
                "percent_of_all_540_segments":
                    len(fam) / len(df) * 100,
                "total_duration_seconds":
                    fam["duration_seconds"].sum(),
                "relative_time_share_pct":
                    (
                        fam["duration_seconds"].sum()
                        / total_duration
                        * 100
                        if total_duration
                        else 0
                    ),
                "median_duration_seconds":
                    fam["duration_seconds"].median(),
                "mean_duration_seconds":
                    fam["duration_seconds"].mean(),
                "mean_events_per_segment":
                    fam["event_count"].mean(),
                "median_events_per_segment":
                    fam["event_count"].median(),
                "sessions":
                    fam["session_id"].nunique(),
                "operators":
                    (
                        sum(
                            len(
                                split_set(x)
                            )
                            for x
                            in fam.get(
                                "operators",
                                pd.Series(
                                    "",
                                    index=fam.index,
                                ),
                            )
                            if str(x).strip()
                        )
                    ),
                "segments_with_word":
                    word_segments,
                "word_segment_pct":
                    word_segments / len(fam) * 100,
                "segments_with_excel":
                    excel_segments,
                "excel_segment_pct":
                    excel_segments / len(fam) * 100,
                "segments_with_notepad":
                    notepad_segments,
                "notepad_segment_pct":
                    notepad_segments / len(fam) * 100,
                "segments_with_browser_app":
                    browser_app_segments,
                "browser_app_segment_pct":
                    browser_app_segments / len(fam) * 100,
                "segments_with_browser_interaction":
                    browser_interacting_segments,
                "browser_interaction_pct":
                    browser_interacting_segments / len(fam) * 100,
                "total_browser_events":
                    fam.apply(
                        browser_events,
                        axis=1,
                    ).sum(),
                "segments_with_clipboard":
                    clipboard_segments,
                "clipboard_segment_pct":
                    clipboard_segments / len(fam) * 100,
                "total_clipboard_changes":
                    fam["clipboard_change_count"].sum(),
                "segments_with_dom_field_evidence":
                    field_evidence_segments,
                "dom_field_evidence_pct":
                    field_evidence_segments / len(fam) * 100,
                "mean_dom_values_per_segment":
                    fam["dom_value_count"].mean(),
                "mean_url_count_per_segment":
                    fam["url_count"].mean(),
                "mean_window_count_per_segment":
                    fam["window_count"].mean(),
                "mean_app_count_per_segment":
                    fam["app_count"].mean(),
            }
        )

    family_df = pd.DataFrame(rows)

    # Direct OCR case-type metrics.
    case_rows = []

    if not case_evidence.empty:
        for (
            family,
            candidate,
        ), group in case_evidence.groupby(
            [
                "family",
                "candidate_process",
            ]
        ):
            keys = sorted(
                set(
                    group[
                        "segment_key"
                    ].astype(str)
                )
            )

            # Recover full 540 metrics for those directly observed OCR
            # segments. Do NOT extrapolate to the rest.
            subset = df[
                df["segment_key"].isin(keys)
            ].copy()

            case_rows.append(
                {
                    "family":
                        family,
                    "observed_case_type":
                        candidate,
                    "direct_ocr_segment_count":
                        len(keys),
                    "share_of_family_all_segments_pct":
                        (
                            len(keys)
                            / len(
                                df[
                                    df["label"]
                                    == family
                                ]
                            )
                            * 100
                        ),
                    "share_of_family_ocr_observed_segments_pct":
                        (
                            len(keys)
                            / max(
                                1,
                                len(
                                    df[
                                        (
                                            df[
                                                "label"
                                            ]
                                            == family
                                        )
                                        & (
                                            df[
                                                "segment_key"
                                            ].isin(
                                                set(
                                                    case_evidence[
                                                        case_evidence[
                                                            "family"
                                                        ]
                                                        == family
                                                    ][
                                                        "segment_key"
                                                    ]
                                                )
                                            )
                                        )
                                    ]
                                ),
                            )
                            * 100
                        ),
                    "sessions":
                        subset[
                            "session_id"
                        ].nunique(),
                    "total_duration_seconds":
                        subset[
                            "duration_seconds"
                        ].sum(),
                    "median_duration_seconds":
                        subset[
                            "duration_seconds"
                        ].median(),
                    "mean_events_per_segment":
                        subset[
                            "event_count"
                        ].mean(),
                    "word_pct":
                        (
                            subset[
                                "apps"
                            ]
                            .map(has_word)
                            .mean()
                            * 100
                        ),
                    "excel_pct":
                        (
                            subset[
                                "apps"
                            ]
                            .map(has_excel)
                            .mean()
                            * 100
                        ),
                    "notepad_pct":
                        (
                            subset[
                                "apps"
                            ]
                            .map(has_notepad)
                            .mean()
                            * 100
                        ),
                    "clipboard_pct":
                        (
                            (
                                subset[
                                    "clipboard_change_count"
                                ]
                                > 0
                            ).mean()
                            * 100
                        ),
                    "browser_interaction_pct":
                        (
                            (
                                subset[
                                    "browser_event_count"
                                ]
                                > 0
                            ).mean()
                            * 100
                        ),
                    "field_evidence_pct":
                        (
                            (
                                subset[
                                    "dom_value_count"
                                ]
                                > 0
                            ).mean()
                            * 100
                        ),
                    "representative_segments":
                        " | ".join(
                            keys[:15]
                        ),
                }
            )

    case_df = pd.DataFrame(
        case_rows
    )

    # Sort for readability.
    if not family_df.empty:
        family_df = family_df.sort_values(
            "execution_count_segments",
            ascending=False,
        )

    if not case_df.empty:
        case_df = case_df.sort_values(
            [
                "family",
                "direct_ocr_segment_count",
            ],
            ascending=[
                True,
                False,
            ],
        )

    family_df.to_csv(
        FAMILY_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    case_df.to_csv(
        CASETYPE_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print(
        "ALL-540 FAMILY QUANTIFICATION"
    )
    print("=" * 80)

    print(
        family_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "DIRECT-OCR OBSERVED CASE-TYPE QUANTIFICATION"
    )
    print("=" * 80)

    if case_df.empty:
        print(
            "No observed case types available."
        )
    else:
        print(
            case_df.to_string(
                index=False
            )
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Family metrics use all 540 segments."
    )
    print(
        "Observed case-type metrics use only segments with direct OCR evidence."
    )
    print(
        "No case-type prevalence is extrapolated to OCR-free segments."
    )

    print()
    print(
        "Outputs:"
    )
    print(
        FAMILY_OUT
    )
    print(
        CASETYPE_OUT
    )

    print()
    print(
        "Layer 7 complete."
    )


if __name__ == "__main__":
    main()
