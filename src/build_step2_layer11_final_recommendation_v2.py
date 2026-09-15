
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

L7 = OUT / "step2_layer7_all540_family_quantification.csv"
L8 = OUT / "step2_layer8_family_work_decomposition.csv"
L9 = OUT / "step2_layer9_feasibility_risk_matrix_v2.csv"
L9_APP = OUT / "step2_layer9_segment_application_evidence_v2.csv"
L3 = OUT / "step2_layer3_authoritative_segment_evidence.csv"

FINAL_REC = OUT / "step2_layer11_final_recommendation_v2.csv"
STEP3_SCOPE = OUT / "step2_layer11_step3_automation_scope_v2.csv"
DEFERRED = OUT / "step2_layer11_deferred_processes_v2.csv"


def read(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def num(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def row_for(df, col, value):
    rows = df[df[col].astype(str) == str(value)]
    return rows.iloc[0] if not rows.empty else None


def true_operator_count(l3, family):
    fam = l3[l3["label"].astype(str) == str(family)].copy()

    for col in ["operator_hash", "operator", "operators"]:
        if col not in fam.columns:
            continue

        vals = set()
        for value in fam[col].astype(str):
            if not value or value.lower() == "nan":
                continue
            for item in value.replace(";", "|").split("|"):
                item = item.strip()
                if item:
                    vals.add(item)

        if vals:
            return len(vals), col

    for col in ["operator_count", "operators_count"]:
        if col in fam.columns:
            vals = pd.to_numeric(
                fam[col],
                errors="coerce",
            ).dropna()
            if not vals.empty:
                return int(vals.max()), col

    return "NOT_AVAILABLE", "not_available"


def exact_app_variants(l9_app, segment_keys):
    subset = l9_app[
        l9_app["segment_key"].astype(str).isin(
            {str(x) for x in segment_keys}
        )
    ].copy()

    if subset.empty:
        return "NOT_AVAILABLE"

    parts = []

    for name, col in [
        ("Word", "word_active"),
        ("Excel", "excel_active"),
        ("Notepad", "notepad_active"),
        ("Browser", "browser_active"),
    ]:
        if col not in subset.columns:
            continue

        pct = (
            pd.to_numeric(
                subset[col],
                errors="coerce",
            )
            .fillna(0)
            .mean()
            * 100
        )

        if pct > 0:
            parts.append(f"{name} {pct:.1f}%")

    return " | ".join(parts) if parts else "No exact active-app evidence"


def main():
    print("=" * 80)
    print("STEP 2 — LAYER 11 FINAL RECOMMENDATION V2")
    print("=" * 80)

    l7 = read(L7)
    l8 = read(L8)
    l9 = read(L9)
    l9_app = read(L9_APP)
    l3 = read(L3)

    operator_map = {}
    for family in l7["process_family"].astype(str).unique():
        operator_map[family] = true_operator_count(
            l3,
            family,
        )

    priority_rows = []

    for family, priority, title in [
        ("pi", "1", "PRIMARY BUILD TARGET"),
        ("la", "2", "SECOND OPPORTUNITY"),
    ]:
        f7 = row_for(l7, "process_family", family)
        f8 = row_for(l8, "family", family)
        f9 = row_for(l9, "family", family)

        if f7 is None:
            continue

        fam_keys = set(
            l3[
                l3["label"].astype(str) == family
            ].apply(
                lambda r: f"{r['session_id']}::{int(r['segment_index'])}",
                axis=1,
            )
        )

        app_variants = exact_app_variants(
            l9_app,
            fam_keys,
        )

        op_count, op_source = operator_map.get(
            family,
            ("NOT_AVAILABLE", "not_available"),
        )

        if family == "pi":
            purpose = (
                "Payroll/expense-item processing and related employee payroll changes"
            )
            why = (
                "pi is the largest family by both workload and time: 209/540 segments "
                "(38.7%) and 35.8% of total segment time. It also has pervasive mechanical "
                "activity (100%), frequent clipboard transfer (88.0%), and frequent browser "
                "interaction (90.0%). The opportunity is large enough to justify first build, "
                "while the 34.45% decision-point evidence means the build must retain a human gate."
            )
            scope = (
                "Automate repetitive retrieval, copy/transfer, transformation, structured "
                "form entry, predictable navigation, document preparation, and submission "
                "preparation. Keep approval/hold/review/exception decisions human."
            )
        else:
            purpose = "Attendance and leave application processing"
            why = (
                "la has 99/540 segments (18.3%) but consumes 24.2% of total segment time, "
                "with 100% mechanical activity, 93.9% clipboard activity, and 92.9% browser "
                "interaction. It is the strongest follow-on opportunity once the pi pilot "
                "proves the reusable automation pattern."
            )
            scope = (
                "Reuse the validated automation pattern for repetitive attendance/leave "
                "transfer and form-entry work, while retaining human review and exception handling."
            )

        priority_rows.append(
            {
                "priority": priority,
                "family": family,
                "title": title,
                "business_purpose": purpose,
                "segments": int(
                    num(
                        f7["execution_count_segments"]
                    )
                ),
                "segment_share_pct": round(
                    num(
                        f7["percent_of_all_540_segments"]
                    ),
                    2,
                ),
                "time_share_pct": round(
                    num(
                        f7["relative_time_share_pct"]
                    ),
                    2,
                ),
                "median_duration_seconds": round(
                    num(
                        f7["median_duration_seconds"]
                    ),
                    2,
                ),
                "operator_count": op_count,
                "operator_source": op_source,
                "application_variants_exact_active_app": app_variants,
                "mechanical_evidence_pct": round(
                    num(
                        f8["mechanical_evidence_pct"]
                    ) if f8 is not None else 0,
                    2,
                ),
                "decision_point_evidence_pct": round(
                    num(
                        f8["decision_point_evidence_pct"]
                    ) if f8 is not None else 0,
                    2,
                ),
                "clipboard_segment_pct": round(
                    num(
                        f7["clipboard_segment_pct"]
                    ),
                    2,
                ),
                "browser_interaction_pct": round(
                    num(
                        f7["browser_interaction_pct"]
                    ),
                    2,
                ),
                "implementation_feasibility": (
                    f9["overall_feasibility"]
                    if f9 is not None
                    else "UNKNOWN"
                ),
                "governance_risk": (
                    f9["data_governance_risk"]
                    if f9 is not None
                    else "UNKNOWN"
                ),
                "why_selected": why,
                "rollout_recommendation": scope,
            }
        )

    priority_df = pd.DataFrame(priority_rows)

    deferred_rows = []

    reasons = {
        "ob": (
            "Substantial workload (14.3% of segments / 16.8% of time) and strong "
            "mechanical activity, but higher governance sensitivity makes it better "
            "as a later controlled pilot after the primary pattern is validated."
        ),
        "si": (
            "Meaningful repetitive workload (12.6% of segments / 11.7% of time), "
            "but higher governance sensitivity and decision involvement favor a "
            "controlled pilot after the first two opportunities."
        ),
        "rt": (
            "Lower workload (10.2% of segments / 8.6% of time) and high governance "
            "sensitivity; defer until higher-value families are validated."
        ),
        "other": (
            "Only 5.9% of segments and 3.0% of time with no stable business-family "
            "meaning; not a sensible first automation target."
        ),
    }

    names = {
        "ob": "Deferred — onboarding",
        "si": "Deferred — social insurance",
        "rt": "Deferred — resident tax",
        "other": "Deferred — supporting/non-family activity",
    }

    for family in ["ob", "si", "rt", "other"]:
        f7 = row_for(l7, "process_family", family)
        if f7 is None:
            continue

        deferred_rows.append(
            {
                "family": family,
                "title": names[family],
                "segments": int(
                    num(
                        f7["execution_count_segments"]
                    )
                ),
                "segment_share_pct": round(
                    num(
                        f7["percent_of_all_540_segments"]
                    ),
                    2,
                ),
                "time_share_pct": round(
                    num(
                        f7["relative_time_share_pct"]
                    ),
                    2,
                ),
                "decision": "DEFER",
                "reason": reasons[family],
            }
        )

    deferred_df = pd.DataFrame(deferred_rows)

    scope_rows = [
        {
            "scope_area": "Target",
            "value": "pi family-level workflow",
            "decision": "BUILD FIRST",
            "evidence": (
                "Largest family: 38.7% of segments and 35.8% of total time, "
                "with pervasive mechanical activity."
            ),
        },
        {
            "scope_area": "Automate",
            "value": (
                "Read/retrieve visible case data; copy/transfer data; "
                "transform/format data; populate structured browser fields; "
                "prepare repetitive documents; predictable navigation; "
                "repetitive submission preparation."
            ),
            "decision": "IN SCOPE",
            "evidence": (
                "Layer 7/8 show pervasive browser, clipboard, typing, form-entry, "
                "navigation, and document activity."
            ),
        },
        {
            "scope_area": "Human gate",
            "value": (
                "Approval, hold/rejection, policy interpretation, ambiguous cases, "
                "exception handling, and final business release."
            ),
            "decision": "OUT OF SCOPE",
            "evidence": (
                "Layer 8 found contextual decision-point evidence; a button label "
                "alone is not treated as proof that preceding work is automatable."
            ),
        },
        {
            "scope_area": "Automation form",
            "value": (
                "Human-in-the-loop desktop/browser workflow assistant with an explicit review gate."
            ),
            "decision": "RECOMMENDED",
            "evidence": (
                "Browser/data-transfer work is directly evidenced; API/authentication/"
                "business-rule availability remains unknown."
            ),
        },
        {
            "scope_area": "Pre-production gate",
            "value": (
                "Validate authentication, permissions, selector stability, business rules, "
                "exceptions, auditability, and data-handling controls in a sandbox."
            ),
            "decision": "REQUIRED",
            "evidence": (
                "These production facts are not established by Dataset B and remain Layer-9 unknowns."
            ),
        },
        {
            "scope_area": "Case-type branching",
            "value": (
                "Do not create separate autonomous branches from OCR-observed case types yet; "
                "treat them as branch evidence within pi."
            ),
            "decision": "DEFER",
            "evidence": (
                "Layer 5 established observed business case types but no confirmed "
                "workflow-level subprocess split."
            ),
        },
    ]

    scope_df = pd.DataFrame(scope_rows)

    priority_df.to_csv(
        FINAL_REC,
        index=False,
        encoding="utf-8-sig",
    )
    scope_df.to_csv(
        STEP3_SCOPE,
        index=False,
        encoding="utf-8-sig",
    )
    deferred_df.to_csv(
        DEFERRED,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("FINAL PRIORITIES")
    print("=" * 80)
    print(
        priority_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("DEFERRED")
    print("=" * 80)
    print(
        deferred_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("STEP-3 SCOPE")
    print("=" * 80)
    print(
        scope_df.to_string(
            index=False
        )
    )

    print()
    print("Outputs:")
    print(FINAL_REC)
    print(STEP3_SCOPE)
    print(DEFERRED)

    print()
    print(
        "Layer 11 V2 complete."
    )
    print(
        "Application variants now come from corrected Layer-9 exact active-app evidence."
    )
    print(
        "segments_v2.jsonl was not modified."
    )


if __name__ == "__main__":
    main()
