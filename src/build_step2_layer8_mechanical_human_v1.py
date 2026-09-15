
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SEGMENT_EVIDENCE = OUT / "step2_layer3_authoritative_segment_evidence.csv"
SEMANTIC = OUT / "step2_layer6_semantic_segment_profiles.csv"

SEGMENT_OUT = OUT / "step2_layer8_segment_work_decomposition.csv"
FAMILY_OUT = OUT / "step2_layer8_family_work_decomposition.csv"
DECISION_OUT = OUT / "step2_layer8_decision_point_evidence.csv"


FAMILIES = [
    "pi",
    "la",
    "ob",
    "si",
    "rt",
    "other",
]


def norm(x):
    return re.sub(
        r"\s+",
        " ",
        str(x or ""),
    ).strip()


def split_values(value):
    text = str(value or "").strip()

    if not text:
        return []

    # Handle JSON/list-like columns.
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return [
                    str(k)
                    for k in parsed.keys()
                ]

            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict):
                        result.extend(
                            str(k)
                            for k in item.keys()
                        )
                    else:
                        result.append(str(item))
                return result
        except Exception:
            pass

    return [
        x.strip()
        for x in re.split(
            r"\s*\|\s*|\s*;\s*|\s*,\s*",
            text,
        )
        if x.strip()
    ]


def numeric(row, candidates):
    for col in candidates:
        if col in row.index:
            try:
                return float(row[col] or 0)
            except Exception:
                pass
    return 0.0


def find_column(df, candidates):
    lowered = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[
                candidate.lower()
            ]

    return None


def contains_any(text, terms):
    text = str(text or "").lower()
    return any(
        term.lower() in text
        for term in terms
    )


MECHANICAL_EVENT_TYPES = {
    "browser_navigation",
    "browser_form_input",
    "browser_click",
    "mouse_click",
    "keystroke",
    "shortcut",
    "clipboard_change",
    "app_switch",
    "window_title_change",
    "mouse_scroll",
}

MECHANICAL_ACTION_TERMS = {
    "navigation",
    "copy",
    "paste",
    "enter",
    "input",
    "typing",
    "form input",
    "document",
    "save",
    "open",
    "close",
    "search",
}

JUDGMENT_TERMS = {
    "approve": [
        "approve",
        "approval",
        "承認",
    ],
    "hold": [
        "hold",
        "保留",
    ],
    "return_for_correction": [
        "return for correction",
        "reject back",
        "差戻し",
    ],
    "exception": [
        "exception",
        "異常",
        "例外",
        "要確認",
        "要対応",
    ],
    "manual_review": [
        "manual review",
        "review",
        "確認",
        "審査",
        "照合",
    ],
    "ambiguous_interpretation": [
        "ambiguous",
        "判断",
        "判定",
        "interpretation",
    ],
}


def load_inputs():
    if not SEGMENT_EVIDENCE.exists():
        raise FileNotFoundError(
            SEGMENT_EVIDENCE
        )

    if not SEMANTIC.exists():
        raise FileNotFoundError(
            SEMANTIC
        )

    evidence = pd.read_csv(
        SEGMENT_EVIDENCE,
        keep_default_na=False,
    )
    semantic = pd.read_csv(
        SEMANTIC,
        keep_default_na=False,
    )

    for df in (
        evidence,
        semantic,
    ):
        if "segment_key" not in df.columns:
            df["segment_key"] = (
                df["session_id"].astype(str)
                + "::"
                + df["segment_index"].astype(str)
            )

    # Semantic file should be the canonical 540-row representation.
    if len(semantic) != 540:
        raise RuntimeError(
            f"Expected 540 semantic profiles; got {len(semantic)}"
        )

    joined = semantic.merge(
        evidence,
        on="segment_key",
        how="left",
        suffixes=(
            "_semantic",
            "_evidence",
        ),
        validate="one_to_one",
    )

    if len(joined) != 540:
        raise RuntimeError(
            "Layer-8 join did not preserve all 540 segments."
        )

    return joined


def detect_mechanical_evidence(row):
    event_types = set(
        x.lower()
        for x in split_values(
            row.get(
                "event_types",
                "",
            )
        )
    )

    # If Layer-3 provides explicit event-type counts as JSON, collect them.
    event_type_counts = {}
    for col in row.index:
        col_s = str(col)

        if (
            "event_type" in col_s.lower()
            and "count" in col_s.lower()
        ):
            value = row[col]
            try:
                count = float(value or 0)
            except Exception:
                continue

            if count > 0:
                event_type_counts[
                    col_s
                ] = count

    browser_form = numeric(
        row,
        [
            "browser_form_input_count",
            "browser_form_input",
        ],
    )

    browser_click = numeric(
        row,
        [
            "browser_click_count",
            "browser_click",
        ],
    )

    browser_navigation = numeric(
        row,
        [
            "browser_navigation_count",
            "browser_navigation",
        ],
    )

    clipboard = numeric(
        row,
        [
            "clipboard_change_count",
            "clipboard_changes",
        ],
    )

    app_switch = numeric(
        row,
        [
            "app_switch_count",
            "app_switch",
        ],
    )

    keystrokes = numeric(
        row,
        [
            "keystroke_count",
            "keystrokes",
        ],
    )

    mechanical_categories = []

    if browser_navigation > 0:
        mechanical_categories.append(
            "navigation"
        )

    if browser_form > 0:
        mechanical_categories.append(
            "form_entry"
        )

    if clipboard > 0:
        mechanical_categories.append(
            "copy_paste"
        )

    apps = str(
        row.get(
            "apps",
            "",
        )
    ).lower()

    windows = str(
        row.get(
            "windows",
            "",
        )
    ).lower()

    if (
        "word" in apps
        or "word" in windows
        or ".doc" in windows
    ):
        mechanical_categories.append(
            "document_preparation"
        )

    if (
        "notepad" in apps
        or "notepad" in windows
        or ".txt" in windows
    ):
        mechanical_categories.append(
            "text_note_preparation"
        )

    if (
        "excel" in apps
        or "excel" in windows
        or ".xls" in windows
        or ".xlsx" in windows
    ):
        mechanical_categories.append(
            "spreadsheet_handling"
        )

    if browser_click > 0:
        mechanical_categories.append(
            "repetitive_browser_interaction"
        )

    if (
        keystrokes > 0
        or "keystroke" in event_types
    ):
        mechanical_categories.append(
            "typing_entry"
        )

    if (
        app_switch > 0
        or "app_switch" in event_types
    ):
        mechanical_categories.append(
            "cross_application_navigation"
        )

    return {
        "categories":
            sorted(
                set(
                    mechanical_categories
                )
            ),
        "browser_form_inputs":
            browser_form,
        "browser_clicks":
            browser_click,
        "browser_navigations":
            browser_navigation,
        "clipboard_changes":
            clipboard,
        "app_switches":
            app_switch,
        "keystrokes":
            keystrokes,
        "event_type_counts":
            event_type_counts,
    }


def detect_judgment_evidence(row):
    """
    Deliberately conservative.

    A decision label/button alone is NOT enough to claim that the preceding
    work is human judgment or that the preceding work is non-automatable.

    We record decision-point evidence only and require contextual signals:
      - action/status evidence in OCR/semantic text, AND
      - some case/business evidence (business term, field, or route).

    This still does not reveal the worker's internal reasoning.
    """
    text = " ".join(
        [
            str(
                row.get(
                    "english_translation",
                    "",
                )
            ),
            str(
                row.get(
                    "actions_status",
                    "",
                )
            ),
            str(
                row.get(
                    "business_terms",
                    "",
                )
            ),
            str(
                row.get(
                    "fields",
                    "",
                )
            ),
            str(
                row.get(
                    "raw_english",
                    "",
                )
            ),
        ]
    ).lower()

    evidence = []

    for category, terms in JUDGMENT_TERMS.items():
        hits = [
            term
            for term in terms
            if term.lower() in text
        ]

        if hits:
            evidence.append(
                {
                    "category":
                        category,
                    "terms":
                        sorted(
                            set(
                                hits
                            )
                        ),
                }
            )

    business_context = bool(
        str(
            row.get(
                "business_terms",
                "",
            )
        ).strip()
        or str(
            row.get(
                "fields",
                "",
            )
        ).strip()
        or str(
            row.get(
                "routes",
                "",
            )
        ).strip()
    )

    contextual = (
        bool(evidence)
        and business_context
    )

    return {
        "decision_evidence":
            evidence,
        "business_context_present":
            business_context,
        "contextual_decision_evidence":
            contextual,
    }


def build_segment_output(df):
    rows = []

    for _, row in df.iterrows():
        mechanical = (
            detect_mechanical_evidence(
                row
            )
        )
        judgment = (
            detect_judgment_evidence(
                row
            )
        )

        mechanical_categories = (
            mechanical[
                "categories"
            ]
        )

        judgment_categories = [
            item[
                "category"
            ]
            for item
            in judgment[
                "decision_evidence"
            ]
        ]

        # Evidence status rather than a fabricated automation percentage.
        if judgment["contextual_decision_evidence"]:
            human_status = (
                "decision_point_evidence"
            )
        else:
            human_status = (
                "no_direct_judgment_evidence"
            )

        if mechanical_categories:
            mechanical_status = (
                "mechanical_activity_evidence"
            )
        else:
            mechanical_status = (
                "no_direct_mechanical_evidence"
            )

        # Important interpretation:
        # mechanical evidence does not mean fully automatable.
        # judgment evidence does not mean the whole segment is human.
        interpretation = []

        if mechanical_categories:
            interpretation.append(
                "repetitive/mechanical activity is directly observed"
            )

        if judgment_categories:
            interpretation.append(
                "a decision/review point is directly evidenced"
            )

        if not interpretation:
            interpretation.append(
                "insufficient evidence to classify work mode"
            )

        if (
            mechanical_categories
            and judgment_categories
        ):
            interpretation.append(
                "segment contains both mechanical activity and a decision point"
            )

        rows.append(
            {
                "segment_key":
                    row["segment_key"],
                "session_id":
                    row.get(
                        "session_id_semantic",
                        row.get(
                            "session_id",
                            "",
                        ),
                    ),
                "segment_index":
                    row.get(
                        "segment_index_semantic",
                        row.get(
                            "segment_index",
                            "",
                        ),
                    ),
                "family":
                    row.get(
                        "family",
                        row.get(
                            "label",
                            "",
                        ),
                    ),
                "business_terms":
                    row.get(
                        "business_terms",
                        "",
                    ),
                "routes":
                    row.get(
                        "routes_semantic",
                        row.get(
                            "routes",
                            "",
                        ),
                    ),
                "application_variant":
                    row.get(
                        "application_variant",
                        "",
                    ),
                "mechanical_categories":
                    " | ".join(
                        mechanical_categories
                    ),
                "mechanical_status":
                    mechanical_status,
                "browser_form_inputs":
                    mechanical[
                        "browser_form_inputs"
                    ],
                "browser_clicks":
                    mechanical[
                        "browser_clicks"
                    ],
                "browser_navigations":
                    mechanical[
                        "browser_navigations"
                    ],
                "clipboard_changes":
                    mechanical[
                        "clipboard_changes"
                    ],
                "app_switches":
                    mechanical[
                        "app_switches"
                    ],
                "keystrokes":
                    mechanical[
                        "keystrokes"
                    ],
                "document_tools":
                    row.get(
                        "documents",
                        "",
                    ),
                "human_judgment_evidence":
                    " | ".join(
                        judgment_categories
                    ),
                "human_status":
                    human_status,
                "decision_evidence_detail":
                    json.dumps(
                        judgment[
                            "decision_evidence"
                        ],
                        ensure_ascii=False,
                    ),
                "business_context_present":
                    int(
                        judgment[
                            "business_context_present"
                        ]
                    ),
                "interpretation":
                    " ; ".join(
                        interpretation
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_family_output(
    segment_df,
    original,
):
    rows = []

    for family in FAMILIES:
        fam = segment_df[
            segment_df[
                "family"
            ]
            == family
        ]

        if fam.empty:
            continue

        original_fam = original[
            original[
                "label"
            ]
            == family
        ]

        mechanical_segments = int(
            (
                fam[
                    "mechanical_status"
                ]
                == "mechanical_activity_evidence"
            ).sum()
        )

        judgment_segments = int(
            (
                fam[
                    "human_status"
                ]
                == "decision_point_evidence"
            ).sum()
        )

        both_segments = int(
            (
                fam[
                    "interpretation"
                ]
                .str.contains(
                    "both mechanical activity",
                    na=False,
                )
            ).sum()
        )

        categories = Counter()

        for value in fam[
            "mechanical_categories"
        ]:
            for item in split_values(
                value
            ):
                categories[item] += 1

        judgment_categories = Counter()

        for value in fam[
            "human_judgment_evidence"
        ]:
            for item in split_values(
                value
            ):
                judgment_categories[item] += 1

        rows.append(
            {
                "family":
                    family,
                "all_segments":
                    len(fam),
                "segments_with_mechanical_evidence":
                    mechanical_segments,
                "mechanical_evidence_pct":
                    mechanical_segments
                    / len(fam)
                    * 100,
                "segments_with_decision_point_evidence":
                    judgment_segments,
                "decision_point_evidence_pct":
                    judgment_segments
                    / len(fam)
                    * 100,
                "segments_with_both":
                    both_segments,
                "both_pct":
                    both_segments
                    / len(fam)
                    * 100,
                "mechanical_categories":
                    " | ".join(
                        f"{k}:{v}"
                        for k, v
                        in categories.most_common()
                    ),
                "decision_categories":
                    " | ".join(
                        f"{k}:{v}"
                        for k, v
                        in judgment_categories.most_common()
                    ),
                "median_duration_seconds":
                    original_fam[
                        "duration_seconds"
                    ].median()
                    if "duration_seconds"
                    in original_fam.columns
                    else None,
                "mean_events_per_segment":
                    original_fam[
                        "event_count"
                    ].mean()
                    if "event_count"
                    in original_fam.columns
                    else None,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_decision_evidence(
    segment_df,
    original,
):
    rows = []

    for _, row in segment_df.iterrows():
        if (
            row[
                "human_status"
            ]
            != "decision_point_evidence"
        ):
            continue

        original_row = original[
            original[
                "segment_key"
            ]
            == row[
                "segment_key"
            ]
        ]

        if original_row.empty:
            continue

        original_row = original_row.iloc[
            0
        ]

        rows.append(
            {
                "segment_key":
                    row["segment_key"],
                "family":
                    row["family"],
                "business_terms":
                    row["business_terms"],
                "routes":
                    row["routes"],
                "application_variant":
                    row[
                        "application_variant"
                    ],
                "decision_evidence":
                    row[
                        "human_judgment_evidence"
                    ],
                "decision_evidence_detail":
                    row[
                        "decision_evidence_detail"
                    ],
                "ocr_english_evidence":
                    str(
                        original_row.get(
                            "english_translation",
                            "",
                        )
                    )[:2000],
                "ocr_japanese_evidence":
                    str(
                        original_row.get(
                            "japanese_ocr",
                            "",
                        )
                    )[:2000],
                "interpretation":
                    (
                        "Decision/review evidence exists, but this does NOT establish "
                        "that the preceding activity could not be automated."
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print("=" * 80)
    print(
        "STEP 2 — LAYER 8 MECHANICAL vs HUMAN WORK DECOMPOSITION"
    )
    print("=" * 80)

    df = load_inputs()

    segment_df = build_segment_output(
        df
    )

    family_df = build_family_output(
        segment_df,
        df,
    )

    decision_df = build_decision_evidence(
        segment_df,
        df,
    )

    segment_df.to_csv(
        SEGMENT_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    family_df.to_csv(
        FAMILY_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    decision_df.to_csv(
        DECISION_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print(
        "FAMILY WORK DECOMPOSITION"
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
        "SEGMENT-LEVEL EVIDENCE COUNTS"
    )
    print("=" * 80)

    print(
        segment_df[
            [
                "family",
                "mechanical_status",
                "human_status",
            ]
        ]
        .value_counts()
        .rename(
            "segments"
        )
        .reset_index()
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "DECISION-POINT EVIDENCE"
    )
    print("=" * 80)

    print(
        f"Segments with contextual decision evidence: "
        f"{len(decision_df):,}"
    )

    if not decision_df.empty:
        print(
            decision_df[
                [
                    "segment_key",
                    "family",
                    "business_terms",
                    "decision_evidence",
                    "interpretation",
                ]
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    print()
    print(
        "IMPORTANT INTERPRETATION RULES"
    )
    print(
        "1. Mechanical evidence means repetitive activity was observed."
    )
    print(
        "2. Decision-point evidence means review/approval/hold/exception language was observed with business context."
    )
    print(
        "3. An approval button alone is NOT treated as proof that preceding work is automatable or human."
    )
    print(
        "4. A segment can contain both mechanical activity and a human decision point."
    )
    print(
        "5. No automation percentage is claimed by this layer."
    )

    print()
    print("Outputs:")
    print(SEGMENT_OUT)
    print(FAMILY_OUT)
    print(DECISION_OUT)

    print()
    print(
        "Layer 8 complete."
    )
    print(
        "No segment labels or Step-1 artifacts were changed."
    )


if __name__ == "__main__":
    main()
