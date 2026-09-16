
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SEMANTIC_FILE = OUT / "step2_layer6_semantic_segment_profiles.csv"
OCR_FILE = OUT / "dataset_b_ocr_translated_20pct.jsonl"
SEGMENT_EVIDENCE_FILE = OUT / "step2_layer3_authoritative_segment_evidence.csv"

CANDIDATE_OUT = OUT / "step2_layer5_subprocess_candidates_v2.csv"
EVIDENCE_OUT = OUT / "step2_layer5_subprocess_candidate_evidence_v2.csv"
VARIANT_OUT = OUT / "step2_layer5_family_variants_v2.csv"


FAMILIES = ["pi", "la", "ob", "si", "rt"]

# Business-level concepts supported by the translated OCR we have already seen.
# Related OCR terms are deliberately consolidated into one candidate where
# they describe the same business purpose.
CONCEPT_RULES = {
    "pi": {
        "expense_reimbursement": [
            "expense reimbursement",
            "expense settlement",
        ],
        "transportation_expense": [
            "transportation expense",
        ],
        "payroll_change": [
            "payroll change",
        ],
        "dependent_deduction": [
            "dependent deduction",
        ],
    },
    "la": {
        "attendance_leave": [
            "attendance",
            "timekeeping",
            "leave",
            "vacation",
        ],
        "paid_leave": [
            "paid leave",
        ],
        "flexible_schedule": [
            "flexible working hours",
        ],
        "overtime_holiday": [
            "overtime",
            "holiday",
        ],
    },
    "ob": {
        "onboarding": [
            "onboarding",
            "new hire",
            "joining",
            "joining date",
        ],
        "new_hire_reconciliation": [
            "new-hire reconciliation",
            "reconciliation",
        ],
        "onboarding_insurance": [
            "employment insurance",
            "health insurance",
        ],
        "onboarding_allowances": [
            "commuting allowance",
            "family allowance",
        ],
        "my_number": [
            "my number",
        ],
    },
    "si": {
        "social_insurance": [
            "social insurance",
            "social insurance premiums",
        ],
        "maternity_leave": [
            "maternity leave",
        ],
        "childcare_leave": [
            "childcare leave",
        ],
        "caregiver_leave": [
            "caregiver leave",
        ],
        "insurance_exemption": [
            "exemption",
        ],
    },
    "rt": {
        "resident_tax": [
            "resident tax",
        ],
        "special_collection": [
            "special collection",
        ],
        "tax_payment": [
            "tax payment",
        ],
    },
}

# Structural dimensions used to describe each candidate, not to invent new
# business processes.
ROUTE_COLORS = {
    "payroll": "/payroll-items",
    "leave": "/leave-applications",
    "onboarding": "/onboarding",
    "social_insurance": "/social-insurance",
    "resident_tax": "/resident-tax",
}


def norm(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def load_profiles():
    if not SEMANTIC_FILE.exists():
        raise FileNotFoundError(SEMANTIC_FILE)

    df = pd.read_csv(
        SEMANTIC_FILE,
        keep_default_na=False,
    )

    if len(df) != 540:
        raise RuntimeError(
            f"Expected 540 semantic segment profiles; got {len(df)}"
        )

    # Keep only columns we need and normalize.
    for col in [
        "family",
        "session_id",
        "segment_key",
        "routes",
        "documents",
        "application_variant",
        "business_terms",
        "fields",
        "actions_status",
        "japanese_ocr",
        "english_translation",
        "ocr_status",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "ocr_screenshot_count" in df.columns:
        df["ocr_screenshot_count"] = pd.to_numeric(
            df["ocr_screenshot_count"],
            errors="coerce",
        ).fillna(0).astype(int)
    else:
        df["ocr_screenshot_count"] = 0

    if "ocr_mean_confidence" in df.columns:
        df["ocr_mean_confidence"] = pd.to_numeric(
            df["ocr_mean_confidence"],
            errors="coerce",
        ).fillna(0)
    else:
        df["ocr_mean_confidence"] = 0.0

    df["ocr_direct"] = (
        df["ocr_screenshot_count"] > 0
    )

    return df


def load_raw_ocr_records():
    if not OCR_FILE.exists():
        raise FileNotFoundError(OCR_FILE)

    rows = []

    with OCR_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            sid = str(
                obj.get(
                    "session_id",
                    "",
                )
            )
            idx = obj.get(
                "segment_index"
            )

            if not sid or idx is None:
                continue

            segment_key = (
                f"{sid}::{int(idx)}"
            )

            jp = []

            for item in (
                obj.get(
                    "japanese_ocr_lines",
                    []
                )
                or []
            ):
                if isinstance(item, dict):
                    j = norm(
                        item.get(
                            "japanese",
                            "",
                        )
                    )
                    e = norm(
                        item.get(
                            "english",
                            "",
                        )
                    )

                    if j:
                        jp.append(
                            f"{j} = {e}"
                        )

            rows.append(
                {
                    "segment_key":
                        segment_key,
                    "ocr_text":
                        norm(
                            obj.get(
                                "ocr_text",
                                "",
                            )
                        ),
                    "english_translation":
                        norm(
                            obj.get(
                                "english_translation",
                                "",
                            )
                        ),
                    "japanese_translation_pairs":
                        " | ".join(
                            jp
                        ),
                    "ocr_mean_score":
                        (
                            float(
                                obj[
                                    "ocr_mean_score"
                                ]
                            )
                            if obj.get(
                                "ocr_mean_score"
                            )
                            not in (
                                None,
                                "",
                            )
                            else 0.0
                        ),
                }
            )

    return pd.DataFrame(rows)


def load_segment_evidence():
    if not SEGMENT_EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            SEGMENT_EVIDENCE_FILE
        )

    df = pd.read_csv(
        SEGMENT_EVIDENCE_FILE,
        keep_default_na=False,
    )

    df["segment_key"] = (
        df["session_id"].astype(str)
        + "::"
        + df["segment_index"].astype(str)
    )

    return df


def attach_auxiliary_ocr(
    profiles,
    raw_ocr,
):
    # A segment may have multiple OCR screenshots. Aggregate them to one row.
    if raw_ocr.empty:
        return profiles

    grouped = (
        raw_ocr.groupby("segment_key")
        .agg(
            raw_ocr_records=(
                "segment_key",
                "size",
            ),
            raw_english=(
                "english_translation",
                lambda x:
                    "\n".join(
                        x.astype(str)
                    ),
            ),
            raw_japanese_pairs=(
                "japanese_translation_pairs",
                lambda x:
                    "\n".join(
                        x.astype(str)
                    ),
            ),
            raw_ocr_confidence=(
                "ocr_mean_score",
                "mean",
            ),
        )
        .reset_index()
    )

    profiles = profiles.drop(
        columns=[
            c
            for c in [
                "raw_ocr_records",
                "raw_english",
                "raw_japanese_pairs",
                "raw_ocr_confidence",
            ]
            if c in profiles.columns
        ],
        errors="ignore",
    )

    return profiles.merge(
        grouped,
        on="segment_key",
        how="left",
        validate="one_to_one",
    )


def concept_hits(
    text,
    family,
):
    result = {}

    for concept, patterns in (
        CONCEPT_RULES.get(
            family,
            {}
        ).items()
    ):
        hits = [
            pattern
            for pattern in patterns
            if pattern.lower() in text
        ]

        if hits:
            result[concept] = hits

    return result


def best_route(
    route_text,
    family,
):
    routes = [
        x.strip().lower()
        for x in str(
            route_text or ""
        ).split("|")
        if x.strip()
    ]

    return routes


def classify_variant(row):
    apps = str(
        row.get(
            "application_variant",
            "",
        )
    )

    documents = str(
        row.get(
            "documents",
            "",
        )
    )

    if apps == "word_assisted" or "Word/document" in documents:
        return "word_assisted"

    if apps == "notepad_assisted" or "Notepad/text" in documents:
        return "notepad_assisted"

    if apps == "excel_assisted" or "Excel/spreadsheet" in documents:
        return "excel_assisted"

    return "browser_only"


def main():
    print("=" * 80)
    print(
        "STEP 2 — LAYER 5 SUBPROCESS DISCOVERY V2"
    )
    print("=" * 80)

    profiles = load_profiles()
    raw_ocr = load_raw_ocr_records()
    raw_evidence = load_segment_evidence()

    profiles = attach_auxiliary_ocr(
        profiles,
        raw_ocr,
    )

    # Use Layer-3 structural evidence to recover routes/apps/documents when
    # these were not carried through to the translated OCR records.
    merge_cols = [
        "segment_key",
        "apps",
        "windows",
        "urls",
        "clipboard_change_count",
        "browser_form_input_count",
        "browser_click_count",
        "app_switch_count",
        "window_title_change_count",
    ]

    merge_cols = [
        c
        for c in merge_cols
        if c in raw_evidence.columns
    ]

    profiles = profiles.drop(
        columns=[
            c
            for c in merge_cols
            if c != "segment_key"
            and c in profiles.columns
        ],
        errors="ignore",
    ).merge(
        raw_evidence[
            merge_cols
        ],
        on="segment_key",
        how="left",
        validate="one_to_one",
    )

    candidates = []
    evidence_rows = []

    for family in FAMILIES:
        fam = profiles[
            profiles[
                "family"
            ]
            == family
        ].copy()

        direct = fam[
            fam["ocr_direct"]
        ]

        if direct.empty:
            continue

        concept_segments = defaultdict(set)
        concept_sessions = defaultdict(set)
        concept_operators = defaultdict(set)
        concept_routes = defaultdict(Counter)
        concept_actions = defaultdict(Counter)
        concept_fields = defaultdict(Counter)
        concept_documents = defaultdict(Counter)
        concept_variants = defaultdict(Counter)

        for _, row in direct.iterrows():

            combined_text = " ".join(
                [
                    str(
                        row.get(
                            "english_translation",
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

            hits = concept_hits(
                combined_text,
                family,
            )

            if not hits:
                continue

            routes = best_route(
                row.get(
                    "urls",
                    row.get(
                        "routes",
                        "",
                    ),
                ),
                family,
            )

            route = (
                routes[0]
                if routes
                else ""
            )

            variant = classify_variant(
                row
            )

            operators = [
                x.strip()
                for x
                in str(
                    row.get(
                        "operators",
                        "",
                    )
                ).split("|")
                if x.strip()
            ]

            fields = [
                x.strip()
                for x
                in str(
                    row.get(
                        "fields",
                        "",
                    )
                ).split("|")
                if x.strip()
            ]

            actions = [
                x.strip()
                for x
                in str(
                    row.get(
                        "actions_status",
                        "",
                    )
                ).split("|")
                if x.strip()
            ]

            docs = [
                x.strip()
                for x
                in str(
                    row.get(
                        "documents",
                        "",
                    )
                ).split("|")
                if x.strip()
            ]

            for concept, supporting_terms in hits.items():
                concept_segments[
                    concept
                ].add(
                    row["segment_key"]
                )

                concept_sessions[
                    concept
                ].add(
                    row["session_id"]
                )

                for operator in operators:
                    concept_operators[
                        concept
                    ].add(
                        operator
                    )

                if route:
                    concept_routes[
                        concept
                    ][route] += 1

                for action in actions:
                    concept_actions[
                        concept
                    ][action] += 1

                for field in fields:
                    concept_fields[
                        concept
                    ][field] += 1

                for doc in docs:
                    concept_documents[
                        concept
                    ][doc] += 1

                concept_variants[
                    concept
                ][variant] += 1

                evidence_rows.append(
                    {
                        "family":
                            family,
                        "candidate_concept":
                            concept,
                        "segment_key":
                            row[
                                "segment_key"
                            ],
                        "session_id":
                            row[
                                "session_id"
                            ],
                        "supporting_ocr_terms":
                            " | ".join(
                                supporting_terms
                            ),
                        "japanese_evidence":
                            str(
                                row.get(
                                    "raw_japanese_pairs",
                                    row.get(
                                        "japanese_ocr",
                                        "",
                                    ),
                                )
                            )[:1500],
                        "english_evidence":
                            str(
                                row.get(
                                    "raw_english",
                                    row.get(
                                        "english_translation",
                                        "",
                                    ),
                                )
                            )[:1500],
                        "route":
                            route,
                        "fields":
                            " | ".join(
                                fields
                            ),
                        "actions_status":
                            " | ".join(
                                actions
                            ),
                        "documents":
                            " | ".join(
                                docs
                            ),
                        "application_variant":
                            variant,
                        "ocr_confidence":
                            row.get(
                                "ocr_mean_confidence",
                                row.get(
                                    "raw_ocr_confidence",
                                    0,
                                ),
                            ),
                    }
                )

        # Recurrence floor chosen BEFORE looking at final candidates:
        # >=5 OCR-covered segments, >=2 sessions, >=2 operators.
        for concept in sorted(
            concept_segments
        ):
            n_segments = len(
                concept_segments[
                    concept
                ]
            )
            n_sessions = len(
                concept_sessions[
                    concept
                ]
            )
            n_operators = len(
                concept_operators[
                    concept
                ]
            )

            if (
                n_segments < 5
                or n_sessions < 2
                or n_operators < 2
            ):
                continue

            top_route = (
                concept_routes[
                    concept
                ]
                .most_common(3)
            )

            top_fields = (
                concept_fields[
                    concept
                ]
                .most_common(10)
            )

            top_actions = (
                concept_actions[
                    concept
                ]
                .most_common(8)
            )

            top_docs = (
                concept_documents[
                    concept
                ]
                .most_common(8)
            )

            variants = (
                concept_variants[
                    concept
                ]
            )

            # Distinct subprocess vs variant:
            # A business concept with coherent route and recurring evidence
            # is a subprocess candidate. Differences in application/document
            # handling are reported separately as variants.
            route_values = {
                x[0]
                for x
                in top_route
            }

            classification = (
                "PROCESS_CANDIDATE"
                if len(
                    route_values
                ) <= 2
                else "REVIEW"
            )

            reason = (
                "Recurring business-purpose evidence across "
                ">=5 segments, >=2 sessions and >=2 operators. "
                "Tooling differences are treated as variants."
            )

            if classification == "REVIEW":
                reason += (
                    " Multiple routes occur inside the candidate; "
                    "manual screenshot review is required before splitting."
                )

            candidates.append(
                {
                    "family":
                        family,
                    "candidate_subprocess":
                        concept,
                    "classification":
                        classification,
                    "ocr_direct_segments":
                        n_segments,
                    "segment_share_of_family_ocr_pct":
                        n_segments
                        / len(
                            direct
                        )
                        * 100,
                    "sessions":
                        n_sessions,
                    "operators":
                        n_operators,
                    "top_routes":
                        " | ".join(
                            f"{x}:{y}"
                            for x, y
                            in top_route
                        ),
                    "key_fields":
                        " | ".join(
                            f"{x}:{y}"
                            for x, y
                            in top_fields
                        ),
                    "actions_status":
                        " | ".join(
                            f"{x}:{y}"
                            for x, y
                            in top_actions
                        ),
                    "document_patterns":
                        " | ".join(
                            f"{x}:{y}"
                            for x, y
                            in top_docs
                        ),
                    "variants":
                        " | ".join(
                            f"{x}:{y}"
                            for x, y
                            in variants.most_common()
                        ),
                    "representative_segments":
                        " | ".join(
                            sorted(
                                concept_segments[
                                    concept
                                ]
                            )[:15]
                        ),
                    "reason":
                        reason,
                }
            )

    candidates_df = pd.DataFrame(
        candidates
    )

    evidence_df = pd.DataFrame(
        evidence_rows
    )

    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            [
                "family",
                "classification",
                "ocr_direct_segments",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

    candidates_df.to_csv(
        CANDIDATE_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    evidence_df.to_csv(
        EVIDENCE_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    # Family-level handling variants across the direct OCR sample.
    variant_rows = []

    for family in FAMILIES:
        fam = profiles[
            (
                profiles["family"]
                == family
            )
            & profiles["ocr_direct"]
        ]

        counts = Counter(
            classify_variant(
                row
            )
            for _, row
            in fam.iterrows()
        )

        for variant_name, count in counts.most_common():
            variant_rows.append(
                {
                    "family":
                        family,
                    "application_variant":
                        variant_name,
                    "ocr_segments":
                        count,
                    "share_within_family_ocr_pct":
                        count
                        / max(
                            1,
                            len(fam),
                        )
                        * 100,
                }
            )

    pd.DataFrame(
        variant_rows
    ).to_csv(
        VARIANT_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print(
        "LAYER 5 CANDIDATE SUBPROCESSES"
    )
    print("=" * 80)

    if candidates_df.empty:
        print(
            "No candidate subprocess met the pre-declared recurrence floor."
        )
    else:
        print(
            candidates_df.to_string(
                index=False
            )
        )

    print()
    print(
        "Recurrence floor: >=5 OCR-covered segments, >=2 sessions, >=2 operators."
    )
    print(
        "This is a candidate-generation rule, not a final process classification."
    )

    print()
    print(
        "Outputs:"
    )
    print(
        CANDIDATE_OUT
    )
    print(
        EVIDENCE_OUT
    )
    print(
        VARIANT_OUT
    )

    print()
    print(
        "Layer 5 candidate discovery complete."
    )
    print(
        "Manual confirmation is still required for PROCESS_CANDIDATE/REVIEW rows."
    )
    print(
        "segments_v2.jsonl was not modified."
    )


if __name__ == "__main__":
    main()
