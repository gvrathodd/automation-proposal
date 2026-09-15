
from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

L7 = OUT / "step2_layer7_all540_family_quantification.csv"
L8 = OUT / "step2_layer8_family_work_decomposition.csv"
L9 = OUT / "step2_layer9_feasibility_risk_matrix_v2.csv"
L5 = OUT / "step2_layer5_conservative_process_taxonomy_v1.csv"

DECISION = OUT / "step2_layer10_automation_candidate_decision_table_v2.csv"
SUPPORT = OUT / "step2_layer10_case_type_support_by_family_v2.csv"
RATIONALE = OUT / "step2_layer10_priority_rationale_v2.csv"


FAMILY_ORDER = ["pi", "la", "ob", "si", "rt", "other"]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def n(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def row_for(df, key_col, value):
    rows = df[df[key_col].astype(str) == str(value)]
    return rows.iloc[0] if not rows.empty else None


def pct(v):
    return f"{n(v):.1f}%"


def application_variants(row):
    parts = []
    mapping = [
        ("Word", "word_segment_pct"),
        ("Excel", "excel_segment_pct"),
        ("Notepad", "notepad_segment_pct"),
    ]
    for name, col in mapping:
        if col in row.index and n(row[col]) > 0:
            parts.append(f"{name} {pct(row[col])}")
    parts.append(
        f"Browser {pct(row.get('browser_interaction_pct', 0))}"
    )
    return " | ".join(parts)


def choose_priority(row):
    """
    Qualitative decision rules. No weighted composite score.

    P1: largest/near-largest workload + strong mechanical evidence +
        strong transfer pattern + acceptable implementation risk.
    P2: good opportunity but smaller workload or stronger constraints.
    P3: meaningful opportunity, but pilot/validation should precede production.
    DEFER: insufficient evidence or low value.
    """
    family = row["family"]
    time_share = n(row["relative_time_share_pct"])
    seg_pct = n(row["percent_of_all_540_segments"])
    mechanical = n(row["mechanical_evidence_pct"])
    clipboard = n(row["clipboard_segment_pct"])
    decision = n(row["decision_point_evidence_pct"])
    ui_risk = str(row["ui_selector_risk"])
    cross_risk = str(row["cross_application_risk"])
    gov_risk = str(row["data_governance_risk"])

    # "Implementation acceptable" here means the evidence supports a
    # technically plausible pilot. It does NOT mean production is approved.
    acceptable_implementation = (
        ui_risk == "MEDIUM"
        and cross_risk in {"MEDIUM", "LOW-MEDIUM"}
    )

    if family == "other":
        return (
            "P3 — DEFER / SUPPORT ONLY",
            "Limited business-process evidence and only 2.95% of total segment time; "
            "keep as supporting work until its purpose is better established.",
        )

    # P1: high-volume, high-time, strongly mechanical. pi wins because it is
    # materially larger than all other families in total time and segment volume.
    if (
        family == "pi"
        and time_share >= 30
        and mechanical >= 95
        and clipboard >= 80
    ):
        return (
            "P1 — PRIMARY CANDIDATE",
            "Largest workload and time share, universal mechanical activity, "
            "and frequent cross-application/data-transfer behavior justify a "
            "focused human-in-the-loop automation pilot.",
        )

    # P2: high-value family with strong repetitive signal.
    if (
        time_share >= 20
        and mechanical >= 95
        and clipboard >= 85
    ):
        return (
            "P2 — STRONG SECOND",
            "Material workload with pervasive repetitive activity and frequent "
            "data transfer; suitable for a scoped automation pilot after access "
            "and governance validation.",
        )

    # P2 for mid-sized family with exceptionally strong browser/clipboard pattern.
    if (
        time_share >= 10
        and mechanical >= 95
        and clipboard >= 90
        and acceptable_implementation
    ):
        return (
            "P2 — GOOD CANDIDATE",
            "Meaningful workload and very strong repetitive/data-transfer signal; "
            "good candidate for scoped automation with human oversight.",
        )

    # Sensitive domains or higher implementation uncertainty should be pilot-first.
    if gov_risk == "HIGH" or ui_risk.startswith("HIGH") or cross_risk == "HIGH":
        return (
            "P3 — PILOT / VALIDATE",
            "Automation opportunity exists, but governance and/or implementation "
            "uncertainty requires sandbox validation before production rollout.",
        )

    if (
        time_share >= 8
        and mechanical >= 80
    ):
        return (
            "P3 — LIMITED / PILOT",
            "Repetitive work is present, but workload or feasibility evidence is "
            "not strong enough to make this the first target.",
        )

    return (
        "P3 — LIMITED / PILOT",
        "Some repetitive activity exists, but evidence does not justify priority over larger opportunities.",
    )


def main():
    print("=" * 80)
    print("STEP 2 — LAYER 10 AUTOMATION DECISION TABLE V2")
    print("=" * 80)

    l7 = read_csv(L7)
    l8 = read_csv(L8)
    l9 = read_csv(L9)
    l5 = read_csv(L5)

    # ------------------------------------------------------------
    # One decision row per authoritative process family.
    # The family is the only unit with exact workload coverage across
    # all 540 segments. OCR-observed case types are supporting evidence,
    # not separate workload rows.
    # ------------------------------------------------------------
    decision_rows = []

    for family in FAMILY_ORDER:
        f7 = row_for(l7, "process_family", family)
        f8 = row_for(l8, "family", family)
        f9 = row_for(l9, "family", family)

        if f7 is None:
            continue

        all_segments = n(
            f7["execution_count_segments"]
        )

        segment_share = n(
            f7["percent_of_all_540_segments"]
        )

        time_share = n(
            f7["relative_time_share_pct"]
        )

        median_duration = n(
            f7["median_duration_seconds"]
        )

        mean_events = n(
            f7["mean_events_per_segment"]
        )

        operators = int(
            n(f7["operators"])
        )

        mechanical = n(
            f8["mechanical_evidence_pct"]
            if f8 is not None
            else 0
        )

        judgment = n(
            f8["decision_point_evidence_pct"]
            if f8 is not None
            else 0
        )

        clipboard = n(
            f7["clipboard_segment_pct"]
        )

        browser = n(
            f7["browser_interaction_pct"]
        )

        word = n(
            f7["word_segment_pct"]
        )

        excel = n(
            f7["excel_segment_pct"]
        )

        notepad = n(
            f7["notepad_segment_pct"]
        )

        implementation = (
            str(
                f9["overall_feasibility"]
            )
            if f9 is not None
            else "UNKNOWN"
        )

        ui_risk = (
            str(
                f9["ui_selector_risk"]
            )
            if f9 is not None
            else "UNKNOWN"
        )

        cross_risk = (
            str(
                f9["cross_application_risk"]
            )
            if f9 is not None
            else "UNKNOWN"
        )

        gov_risk = (
            str(
                f9["data_governance_risk"]
            )
            if f9 is not None
            else "UNKNOWN"
        )

        # Business purpose: family-level names are deliberately conservative.
        purposes = {
            "pi": "Payroll/expense-item processing and related employee payroll changes",
            "la": "Attendance and leave application processing",
            "ob": "Employee onboarding and new-hire processing",
            "si": "Social-insurance related employee processing",
            "rt": "Resident-tax related processing",
            "other": "Supporting / non-family-specific activity",
        }

        # Structured data evidence.
        if family == "other":
            structured = "PARTIALLY STRUCTURED"
            structured_reason = (
                "Some browser/DOM activity exists, but a stable business-purpose "
                "schema is not established."
            )
        else:
            structured = "STRUCTURED UI INPUTS"
            structured_reason = (
                "Family-specific DOM identifiers and browser form interactions "
                "show structured UI fields; underlying API/schema is unknown."
            )

        # Human decision interpretation.
        if judgment >= 30:
            human = (
                "High decision-point evidence; keep approval/hold/review/exception "
                "with a human."
            )
        elif judgment >= 15:
            human = (
                "Moderate decision-point evidence; retain human review for decisions "
                "and exceptions."
            )
        else:
            human = (
                "Lower direct decision-point evidence; do not infer that the "
                "workflow is decision-free."
            )

        repetitive = (
            "read → copy/transfer → transform/enter → navigate → prepare documents → submit"
            if mechanical >= 90
            else
            "repetitive form entry/navigation/data-transfer work"
        )

        cross_app = (
            f"Clipboard activity in {clipboard:.1f}% of segments; "
            f"browser interaction in {browser:.1f}%."
        )

        variants = application_variants(
            f7
        )

        # We are deliberately not calling these "confirmed subprocesses".
        if family == "pi":
            business_branch = (
                "OCR directly observes expense reimbursement, payroll change, "
                "dependent deduction, and transportation-expense case types; "
                "these are branches/case types within the family, not separate "
                "540-segment workloads."
            )
        elif family == "la":
            business_branch = (
                "OCR directly observes attendance/leave and paid-leave case types; "
                "do not treat their counts as additive workloads."
            )
        elif family == "ob":
            business_branch = (
                "OCR directly observes onboarding, insurance, allowances, and "
                "My Number themes; several co-occur on the same screens."
            )
        elif family == "si":
            business_branch = (
                "OCR directly observes social insurance, maternity leave, "
                "childcare leave, and insurance-exemption themes."
            )
        elif family == "rt":
            business_branch = (
                "No sufficiently recurrent OCR case-type split was established; "
                "retain the family-level process."
            )
        else:
            business_branch = (
                "No semantic subprocess split is supported."
            )

        priority, priority_reason = choose_priority(
            {
                "family": family,
                "relative_time_share_pct": time_share,
                "percent_of_all_540_segments": segment_share,
                "mechanical_evidence_pct": mechanical,
                "clipboard_segment_pct": clipboard,
                "decision_point_evidence_pct": judgment,
                "ui_selector_risk": ui_risk,
                "cross_application_risk": cross_risk,
                "data_governance_risk": gov_risk,
            }
        )

        scope = (
            "Automate the repetitive read/copy/transfer/form-entry/document-preparation "
            "path; retain human approval/review/exception handling."
            if priority.startswith(
                ("P1", "P2")
            )
            else
            "Pilot only the stable repetitive substeps; retain human decisions and "
            "exceptions until access, rules, and governance are validated."
        )

        integration_assumptions = (
            "Browser/UI integration is evidenced. "
            "API availability, authentication mechanism, service accounts, "
            "and production permissions are unknown and require validation."
        )

        rows = {
            "process":
                f"{family}_family_level_workflow",
            "family":
                family,
            "business_purpose":
                purposes[family],
            "frequency_segments":
                int(all_segments),
            "percentage_of_540_segments":
                round(
                    segment_share,
                    2,
                ),
            "relative_time_share_pct":
                round(
                    time_share,
                    2,
                ),
            "median_duration_seconds":
                round(
                    median_duration,
                    2,
                ),
            "mean_events_per_segment":
                round(
                    mean_events,
                    2,
                ),
            "operators":
                operators,
            "variants":
                variants,
            "repetitive_work":
                repetitive,
            "mechanical_work_evidence_pct":
                round(
                    mechanical,
                    2,
                ),
            "human_judgment":
                human,
            "structured_data":
                structured,
            "cross_app_transfer":
                cross_app,
            "implementation_complexity":
                (
                    "HIGH"
                    if (
                        ui_risk.startswith("HIGH")
                        or cross_risk == "HIGH"
                    )
                    else
                    "MEDIUM"
                ),
            "integration_assumptions":
                integration_assumptions,
            "governance_risk":
                gov_risk,
            "prototype_feasibility":
                implementation,
            "recommended_scope":
                scope,
            "priority":
                priority,
            "priority_reason":
                priority_reason,
            "observed_case_type_branches":
                business_branch,
            "word_involvement_pct":
                round(
                    word,
                    2,
                ),
            "excel_involvement_pct":
                round(
                    excel,
                    2,
                ),
            "notepad_involvement_pct":
                round(
                    notepad,
                    2,
                ),
            "ui_selector_risk":
                ui_risk,
            "cross_application_risk":
                cross_risk,
            "structured_data_reason":
                structured_reason,
        }

        decision_rows.append(
            rows
        )

    decision_df = pd.DataFrame(
        decision_rows
    )

    # ------------------------------------------------------------
    # Supporting case-type table: direct OCR evidence only.
    # This is where the finer semantic information belongs.
    # ------------------------------------------------------------
    support_rows = []

    if not l5.empty:
        obs = l5[
            l5["classification"].astype(str)
            == "OBSERVED_CASE_TYPE"
        ]

        for _, row in obs.iterrows():
            family = str(
                row["family"]
            )

            support_rows.append(
                {
                    "family":
                        family,
                    "observed_case_type":
                        row[
                            "candidate_process"
                        ],
                    "direct_ocr_segments":
                        int(
                            n(
                                row[
                                    "direct_ocr_segments"
                                ]
                            )
                        ),
                    "sessions":
                        int(
                            n(
                                row[
                                    "sessions"
                                ]
                            )
                        ),
                    "evidence_terms":
                        row[
                            "evidence_terms"
                        ],
                    "routes":
                        row[
                            "routes"
                        ],
                    "key_fields":
                        row[
                            "key_fields"
                        ],
                    "actions_status":
                        row[
                            "actions_status"
                        ],
                    "document_pattern":
                        row[
                            "document_pattern"
                        ],
                    "handling_variants":
                        row[
                            "handling_variants"
                        ],
                    "classification":
                        "OBSERVED CASE TYPE — NOT SEPARATE WORKLOAD",
                    "reason":
                        (
                            "Direct OCR shows this business theme. "
                            "Counts may overlap with other case types shown on the same screen, "
                            "so they are not additive and are not used for family workload priority."
                        ),
                }
            )

    support_df = pd.DataFrame(
        support_rows
    )

    # ------------------------------------------------------------
    # Priority rationale output.
    # ------------------------------------------------------------
    rationale_rows = []

    for _, row in decision_df.iterrows():
        rationale_rows.append(
            {
                "process":
                    row["process"],
                "priority":
                    row["priority"],
                "why":
                    row["priority_reason"],
                "workload_evidence":
                    (
                        f"{row['frequency_segments']} segments / "
                        f"{row['relative_time_share_pct']:.2f}% of total time"
                    ),
                "mechanical_evidence":
                    f"{row['mechanical_work_evidence_pct']:.2f}% of family segments",
                "human_decision_evidence":
                    row["human_judgment"],
                "feasibility_constraint":
                    row["prototype_feasibility"],
                "key_risks":
                    (
                        f"UI={row['ui_selector_risk']}; "
                        f"cross-app={row['cross_application_risk']}; "
                        f"governance={row['governance_risk']}"
                    ),
                "decision_principle":
                    (
                        "High workload + substantial repetitive/mechanical work + "
                        "structured inputs + feasible implementation + acceptable risk."
                    ),
            }
        )

    rationale_df = pd.DataFrame(
        rationale_rows
    )

    decision_df.to_csv(
        DECISION,
        index=False,
        encoding="utf-8-sig",
    )

    support_df.to_csv(
        SUPPORT,
        index=False,
        encoding="utf-8-sig",
    )

    rationale_df.to_csv(
        RATIONALE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("FINAL FAMILY-LEVEL AUTOMATION DECISION TABLE")
    print("=" * 80)

    print(
        decision_df[
            [
                "process",
                "family",
                "business_purpose",
                "frequency_segments",
                "percentage_of_540_segments",
                "relative_time_share_pct",
                "operators",
                "variants",
                "repetitive_work",
                "mechanical_work_evidence_pct",
                "human_judgment",
                "structured_data",
                "cross_app_transfer",
                "implementation_complexity",
                "integration_assumptions",
                "governance_risk",
                "prototype_feasibility",
                "recommended_scope",
                "priority",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("DECISION PRINCIPLE")
    print("=" * 80)
    print(
        "No composite numerical score was used."
    )
    print(
        "The priority considers workload + repetitive/mechanical activity + "
        "structured UI + implementation feasibility + cross-app transfer + "
        "human judgment + governance risk together."
    )

    print()
    print(
        "Important: observed OCR case types are supporting branch evidence, "
        "not independent 540-segment workloads."
    )

    print()
    print("Outputs:")
    print(DECISION)
    print(SUPPORT)
    print(RATIONALE)

    print()
    print(
        "Layer 10 V2 complete."
    )


if __name__ == "__main__":
    main()
