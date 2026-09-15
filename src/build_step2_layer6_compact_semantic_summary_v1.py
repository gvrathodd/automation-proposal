
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

OCR_FILE = OUT / "dataset_b_ocr_translated_20pct.jsonl"

SUMMARY_OUT = OUT / "step2_layer6_compact_semantic_summary.csv"
EXAMPLES_OUT = OUT / "step2_layer6_representative_ocr_examples.csv"


FAMILIES = ["pi", "la", "ob", "si", "rt"]

# Business semantics are extracted from the translated OCR, but the original
# Japanese is always retained in the evidence tables.
BUSINESS_TERMS = {
    "pi": [
        "expense reimbursement",
        "expense settlement",
        "payroll change",
        "transportation expense",
        "dependent deduction",
        "payroll",
        "salary",
        "deduction",
        "payment",
    ],
    "la": [
        "attendance",
        "timekeeping",
        "leave",
        "vacation",
        "paid leave",
        "flexible working hours",
        "overtime",
        "holiday",
    ],
    "ob": [
        "new hire",
        "onboarding",
        "joining",
        "new-hire reconciliation",
        "joining date",
        "employment category",
        "department",
        "my number",
        "employment insurance",
        "health insurance",
        "commuting allowance",
        "family allowance",
        "reconciliation",
    ],
    "si": [
        "social insurance",
        "maternity leave",
        "childcare leave",
        "caregiver leave",
        "exemption",
        "social insurance premiums",
    ],
    "rt": [
        "resident tax",
        "tax payment",
        "special collection",
        "tax",
    ],
}

ACTION_PATTERNS = {
    "approval": ["approve", "approval"],
    "hold": ["hold"],
    "return_for_correction": [
        "return for correction",
        "reject back",
    ],
    "reconciliation_complete": [
        "reconciliation complete",
    ],
    "processing_complete": [
        "processing complete",
    ],
    "unconfirmed": ["unconfirmed"],
    "awaiting_processing": [
        "awaiting processing",
    ],
    "unprocessed": ["unprocessed"],
    "confirm_registration": [
        "confirm registration",
    ],
}

FIELD_TERMS = [
    "employee id",
    "full name",
    "name",
    "amount",
    "status",
    "processing comment",
    "category",
    "type",
    "start date",
    "end date",
    "department",
    "joining date",
    "employment category",
    "my number",
    "employment insurance",
    "health insurance",
    "commuting allowance",
    "family allowance",
    "social insurance",
    "resident tax",
    "paid leave",
    "attendance",
]


def norm(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def route_from_value(value: str) -> list[str]:
    routes = []

    for raw in str(value or "").split("|"):
        raw = raw.strip()
        if not raw:
            continue

        m = re.search(
            r"/#(/[^/?#\s]+)",
            raw,
            flags=re.I,
        )

        if m:
            routes.append(
                m.group(1).lower()
            )

    return sorted(set(routes))


def application_variant(apps: str) -> str:
    text = str(apps or "").lower()

    if "word" in text or "winword" in text:
        return "word_assisted"
    if "notepad" in text:
        return "notepad_assisted"
    if "excel" in text:
        return "excel_assisted"

    return "browser_only"


def document_pattern(
    apps: str,
    windows: str,
) -> list[str]:
    text = (
        str(apps or "")
        + " | "
        + str(windows or "")
    ).lower()

    result = []

    if (
        "word" in text
        or "winword" in text
        or ".doc" in text
    ):
        result.append("Word/document")

    if (
        "notepad" in text
        or ".txt" in text
    ):
        result.append("Notepad/text")

    if (
        "excel" in text
        or ".xls" in text
        or ".xlsx" in text
    ):
        result.append("Excel/spreadsheet")

    return sorted(set(result))


def extract_japanese_items(record):
    items = []

    for item in (
        record.get(
            "japanese_ocr_lines",
            []
        )
        or []
    ):
        if isinstance(item, dict):
            jp = norm(
                item.get(
                    "japanese",
                    "",
                )
            )
            en = norm(
                item.get(
                    "english",
                    "",
                )
            )

            if jp:
                items.append(
                    (
                        jp,
                        en,
                    )
                )

    return items


def main():
    print("=" * 80)
    print(
        "STEP 2 — LAYER 6 COMPACT OCR SEMANTIC SUMMARY"
    )
    print("=" * 80)

    if not OCR_FILE.exists():
        raise FileNotFoundError(
            OCR_FILE
        )

    records = []

    with OCR_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            record = json.loads(
                line
            )

            records.append(
                record
            )

    print(
        f"OCR records loaded: {len(records):,}"
    )

    # One summary record per OCR screenshot/segment observation.
    observations = []

    for record in records:
        family = str(
            record.get(
                "label",
                "",
            )
        )

        if family not in FAMILIES:
            continue

        segment_key = (
            str(
                record.get(
                    "session_id",
                    "",
                )
            )
            + "::"
            + str(
                int(
                    record.get(
                        "segment_index",
                        0,
                    )
                )
            )
        )

        english = norm(
            record.get(
                "english_translation",
                "",
            )
        ).lower()

        japanese_items = (
            extract_japanese_items(
                record
            )
        )

        business_terms = [
            term
            for term in BUSINESS_TERMS[
                family
            ]
            if term in english
        ]

        actions = [
            action
            for action, patterns
            in ACTION_PATTERNS.items()
            if any(
                pattern in english
                for pattern in patterns
            )
        ]

        fields = [
            field
            for field in FIELD_TERMS
            if field in english
        ]

        jp_terms = [
            jp
            for jp, _ in japanese_items
        ]

        translated_terms = [
            f"{jp} = {en}"
            for jp, en
            in japanese_items
            if en
        ]

        routes = route_from_value(
            record.get(
                "urls",
                "",
            )
        )

        apps = str(
            record.get(
                "apps",
                "",
            )
        )

        windows = str(
            record.get(
                "windows",
                "",
            )
        )

        observations.append(
            {
                "segment_key":
                    segment_key,
                "family":
                    family,
                "session_id":
                    record.get(
                        "session_id",
                        "",
                    ),
                "segment_index":
                    record.get(
                        "segment_index",
                        "",
                    ),
                "business_terms":
                    business_terms,
                "fields":
                    fields,
                "actions":
                    actions,
                "routes":
                    routes,
                "documents":
                    document_pattern(
                        apps,
                        windows,
                    ),
                "application_variant":
                    application_variant(
                        apps
                    ),
                "japanese_evidence":
                    " | ".join(
                        sorted(
                            set(
                                jp_terms
                            )
                        )
                    ),
                "translated_evidence":
                    " | ".join(
                        sorted(
                            set(
                                translated_terms
                            )
                        )
                    ),
                "ocr_mean_score":
                    record.get(
                        "ocr_mean_score"
                    ),
                "ocr_text":
                    norm(
                        record.get(
                            "ocr_text",
                            "",
                        )
                    ),
                "english_translation":
                    norm(
                        record.get(
                            "english_translation",
                            "",
                        )
                    ),
            }
        )

    obs = pd.DataFrame(
        observations
    )

    if obs.empty:
        raise RuntimeError(
            "No family OCR observations found."
        )

    # Family semantic summary.
    summary_rows = []

    for family in FAMILIES:
        fam = obs[
            obs["family"] == family
        ].copy()

        if fam.empty:
            continue

        business_counter = Counter()
        field_counter = Counter()
        action_counter = Counter()
        route_counter = Counter()
        document_counter = Counter()
        variant_counter = Counter()

        for _, row in fam.iterrows():
            for x in row["business_terms"]:
                business_counter[x] += 1

            for x in row["fields"]:
                field_counter[x] += 1

            for x in row["actions"]:
                action_counter[x] += 1

            for x in row["routes"]:
                route_counter[x] += 1

            for x in row["documents"]:
                document_counter[x] += 1

            variant_counter[
                row["application_variant"]
            ] += 1

        def render(counter, limit=12):
            return " | ".join(
                f"{k} ({v})"
                for k, v
                in counter.most_common(
                    limit
                )
            )

        summary_rows.append(
            {
                "family":
                    family,
                "ocr_records":
                    len(fam),
                "unique_segments_with_ocr":
                    fam[
                        "segment_key"
                    ].nunique(),
                "sessions":
                    fam[
                        "session_id"
                    ].nunique(),
                "mean_ocr_confidence":
                    fam[
                        "ocr_mean_score"
                    ].mean(),
                "recurring_business_concepts":
                    render(
                        business_counter
                    ),
                "recurring_fields":
                    render(
                        field_counter
                    ),
                "workflow_actions_statuses":
                    render(
                        action_counter
                    ),
                "dominant_routes":
                    render(
                        route_counter
                    ),
                "document_patterns":
                    render(
                        document_counter
                    ),
                "application_variants":
                    render(
                        variant_counter
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # Representative examples: highest-confidence screenshots with the
    # strongest amount of semantic information per family.
    obs["semantic_signal_count"] = (
        obs["business_terms"].map(len)
        + obs["fields"].map(len)
        + obs["actions"].map(len)
        + obs["routes"].map(len)
        + obs["documents"].map(len)
    )

    examples = (
        obs[
            obs["semantic_signal_count"] > 0
        ]
        .sort_values(
            [
                "family",
                "semantic_signal_count",
                "ocr_mean_score",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .groupby(
            "family"
        )
        .head(5)
        [
            [
                "family",
                "segment_key",
                "session_id",
                "segment_index",
                "business_terms",
                "fields",
                "actions",
                "routes",
                "documents",
                "application_variant",
                "japanese_evidence",
                "translated_evidence",
                "ocr_mean_score",
            ]
        ]
        .copy()
    )

    # Render list columns as readable strings.
    for col in [
        "business_terms",
        "fields",
        "actions",
        "routes",
        "documents",
    ]:
        examples[col] = examples[
            col
        ].map(
            lambda x:
                " | ".join(
                    x
                )
            if isinstance(
                x,
                list,
            )
            else str(x)
        )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    examples.to_csv(
        EXAMPLES_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print(
        "COMPACT FAMILY SEMANTIC SUMMARY"
    )
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "REPRESENTATIVE OCR EVIDENCE"
    )
    print("=" * 80)

    print(
        examples[
            [
                "family",
                "segment_key",
                "business_terms",
                "fields",
                "actions",
                "routes",
                "documents",
                "japanese_evidence",
                "translated_evidence",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Outputs:"
    )
    print(
        SUMMARY_OUT
    )
    print(
        EXAMPLES_OUT
    )

    print()
    print(
        "Layer 6 packaging complete."
    )
    print(
        "Japanese OCR is retained; English translation is analytical enrichment."
    )
    print(
        "No subprocess classification, propagation, or segment-label changes were performed."
    )


if __name__ == "__main__":
    main()
