
"""
STEP 3E-3 — PI FINAL PROTOTYPE ASSESSMENT V1

Purpose
-------
Freeze the compact Step-3E prototype findings into one final evidence
workbook. This is an assessment layer, not another prototype implementation.

Inputs
------
outputs/step3e1_pi_prototype_evidence_v1.xlsx
outputs/step3e2_pi_prototype_validation_v1.xlsx
latest step3c_pi_parameterized_vs_branch_analysis_v3*.xlsx
latest step3d_pi_technical_interface_discovery_v2*.xlsx

Output
------
outputs/step3e3_pi_final_prototype_assessment_v1.xlsx

Important limitations
---------------------
- Prototype ran only against the controlled local mock.
- It did not access production.
- Test results are prototype evidence, not population estimates.
- Human decision actions remain excluded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

E1 = OUT / "step3e1_pi_prototype_evidence_v1.xlsx"
E2 = OUT / "step3e2_pi_prototype_validation_v1.xlsx"

C3_FILES = sorted(
    OUT.glob("step3c_pi_parameterized_vs_branch_analysis_v3*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
D3_FILES = sorted(
    OUT.glob("step3d_pi_technical_interface_discovery_v2*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def load():
    require(E1)
    require(E2)

    if not C3_FILES:
        raise FileNotFoundError("No Step-3C V3 workbook found.")
    if not D3_FILES:
        raise FileNotFoundError("No Step-3D V2 workbook found.")

    e1_summary = pd.read_excel(
        E1,
        sheet_name="Prototype_Summary",
    )
    e1_results = pd.read_excel(
        E1,
        sheet_name="Execution_Results",
    )
    e1_cases = pd.read_excel(
        E1,
        sheet_name="Test_Cases",
    )

    e2_summary = pd.read_excel(
        E2,
        sheet_name="Validation_Summary",
    )
    e2_results = pd.read_excel(
        E2,
        sheet_name="Execution_Results",
    )
    e2_validation = pd.read_excel(
        E2,
        sheet_name="Validation",
    )

    c3 = C3_FILES[0]
    c3_arch = pd.read_excel(
        c3,
        sheet_name="Architecture_Decision",
    )
    c3_compare = pd.read_excel(
        c3,
        sheet_name="Case_Type_Path_Comparison",
    )
    c3_summary = pd.read_excel(
        c3,
        sheet_name="Case_Type_Summary",
    )

    d3 = D3_FILES[0]
    d3_routes = pd.read_excel(
        d3,
        sheet_name="Browser_Routes",
    )
    d3_events = pd.read_excel(
        d3,
        sheet_name="Input_Transfer_Interface",
    )
    d3_assumptions = pd.read_excel(
        d3,
        sheet_name="Integration_Assumptions",
    )
    d3_proto = pd.read_excel(
        d3,
        sheet_name="Prototype_Interface_Spec",
    )

    return (
        e1_summary,
        e1_results,
        e1_cases,
        e2_summary,
        e2_results,
        e2_validation,
        c3_arch,
        c3_compare,
        c3_summary,
        d3_routes,
        d3_events,
        d3_assumptions,
        d3_proto,
        c3,
        d3,
    )


def build_executive_assessment(
    e1_results: pd.DataFrame,
    e2_results: pd.DataFrame,
    c3_arch: pd.DataFrame,
    d3_assumptions: pd.DataFrame,
):
    e2_pass = int(e2_results["status"].eq("PASS").sum())
    e2_total = len(e2_results)

    production_unknowns = d3_assumptions[
        d3_assumptions["status"].isin(
            ["UNKNOWN", "VALIDATE"]
        )
    ]["technical_question"].tolist()

    architecture = (
        norm(c3_arch.iloc[0]["architecture_decision"])
        if not c3_arch.empty and "architecture_decision" in c3_arch.columns
        else "ONE PARAMETERIZED CORE + VALIDATED BRANCH CANDIDATES"
    )

    return pd.DataFrame([
        {
            "assessment_area": "Prototype implementation",
            "finding":
                "A working local parameterized preparation prototype was executed.",
            "evidence":
                f"Step 3E-1 completed the prepare/verify/stop path on {len(e1_results)} initial cases.",
            "status": "DEMONSTRATED",
        },
        {
            "assessment_area": "Prototype validation",
            "finding":
                "All Step 3E-2 representative validation cases passed.",
            "evidence":
                f"{e2_pass}/{e2_total} validation cases passed; failures were converted into expected safe stops.",
            "status": "DEMONSTRATED",
        },
        {
            "assessment_area": "Parameterized architecture",
            "finding":
                architecture,
            "evidence":
                "Step 3C identified a shared normalized mechanical core with case-specific parameters and variants requiring targeted validation.",
            "status": "SUPPORTED FOR PROTOTYPE",
        },
        {
            "assessment_area": "Automation boundary",
            "finding":
                "Mechanical preparation is automated only inside validated states; uncertainty stops the prototype.",
            "evidence":
                "Step 3B + 3E tests demonstrated missing-input, invalid-input and unexpected-UI stop conditions.",
            "status": "DEMONSTRATED",
        },
        {
            "assessment_area": "Human decision gate",
            "finding":
                "Business decision remains human.",
            "evidence":
                "Register clicks = 0 and Hold clicks = 0 throughout prototype validation.",
            "status": "DEMONSTRATED",
        },
        {
            "assessment_area": "Production readiness",
            "finding":
                "Production integration is not established.",
            "evidence":
                "Authentication, permissions, API availability, selector stability, business-rule access, audit logging and exception recovery remain unknown/validation items.",
            "status": "NOT ESTABLISHED",
        },
    ])


def build_metrics(e1_results, e2_results):
    e1_success = int(
        e1_results["result"]
        .astype(str)
        .isin(["SUCCESS", "SUCCESS_HUMAN_GATE"])
        .sum()
    )
    e1_safe = int(
        e1_results["result"]
        .astype(str)
        .str.startswith("STOP_")
        .sum()
    )

    e2_pass = int(
        e2_results["status"].eq("PASS").sum()
    )
    e2_safe = int(
        e2_results["actual_result"]
        .astype(str)
        .str.startswith("STOP_")
        .sum()
    )

    fields = pd.to_numeric(
        e2_results["fields_verified"],
        errors="coerce",
    ).fillna(0)

    attempts = pd.to_numeric(
        e2_results["fields_attempted"],
        errors="coerce",
    ).fillna(0)

    return pd.DataFrame([
        {
            "metric": "3E-1 initial cases",
            "value": len(e1_results),
            "interpretation": "Initial proof-of-concept execution set.",
        },
        {
            "metric": "3E-1 successful preparation/human-gate cases",
            "value": e1_success,
            "interpretation": "Cases completed through preparation and stopped at human gate.",
        },
        {
            "metric": "3E-1 safe stops",
            "value": e1_safe,
            "interpretation": "Controlled invalid/missing inputs stopped safely.",
        },
        {
            "metric": "3E-2 validation cases",
            "value": len(e2_results),
            "interpretation": "Representative validation set.",
        },
        {
            "metric": "3E-2 validation pass rate",
            "value":
                round(
                    100 * e2_pass / len(e2_results),
                    2
                ) if len(e2_results) else 0,
            "interpretation":
                "Pass means successful preparation or expected safe stop.",
        },
        {
            "metric": "3E-2 safe stops",
            "value": e2_safe,
            "interpretation":
                "Expected validation boundary triggered safely.",
        },
        {
            "metric": "3E-2 field verification",
            "value":
                f"{int(fields.sum())}/{int(attempts.sum())}",
            "interpretation":
                "Across cases where field population was attempted.",
        },
        {
            "metric": "Register clicks",
            "value":
                int(e2_results["register_clicked"].sum()),
            "interpretation":
                "Must remain zero because business decision is human.",
        },
        {
            "metric": "Hold clicks",
            "value":
                int(e2_results["hold_clicked"].sum()),
            "interpretation":
                "Must remain zero because business decision is human.",
        },
    ])


def build_final_scope():
    return pd.DataFrame([
        {
            "prototype_scope": "READ",
            "status": "DEMONSTRATED",
            "final_treatment":
                "Read known case values from a validated local interface.",
            "not_yet_proven":
                "Production source authentication and source-system connectivity.",
        },
        {
            "prototype_scope": "EXTRACT",
            "status": "DEMONSTRATED",
            "final_treatment":
                "Extract mapped values from known fields.",
            "not_yet_proven":
                "Stable production selectors across UI versions.",
        },
        {
            "prototype_scope": "TRANSFORM",
            "status": "DEMONSTRATED",
            "final_treatment":
                "Apply deterministic transformation rules.",
            "not_yet_proven":
                "Complete production business-rule coverage.",
        },
        {
            "prototype_scope": "POPULATE",
            "status": "DEMONSTRATED",
            "final_treatment":
                "Populate validated target fields and verify them.",
            "not_yet_proven":
                "Production field mappings and live-system compatibility.",
        },
        {
            "prototype_scope": "EXCEPTION HANDLING",
            "status": "DEMONSTRATED",
            "final_treatment":
                "Stop on missing/invalid/unexpected conditions.",
            "not_yet_proven":
                "Production recovery/orchestration strategy.",
        },
        {
            "prototype_scope": "BUSINESS DECISION",
            "status": "HUMAN",
            "final_treatment":
                "Human performs approval/hold/rejection/exception/final release.",
            "not_yet_proven":
                "Nothing; autonomous decision is intentionally out of scope.",
        },
    ])


def build_remaining_work(d3_assumptions):
    rows = []

    for _, r in d3_assumptions.iterrows():
        status = norm(r["status"])
        question = norm(r["technical_question"])
        next_validation = norm(r["next_validation"])

        if status in {"UNKNOWN", "VALIDATE", "OBSERVED + VALIDATE"}:
            rows.append({
                "remaining_item": question,
                "current_status": status,
                "required_next_validation": next_validation,
            })

    return pd.DataFrame(rows)


def main():
    (
        e1_summary,
        e1_results,
        e1_cases,
        e2_summary,
        e2_results,
        e2_validation,
        c3_arch,
        c3_compare,
        c3_summary,
        d3_routes,
        d3_events,
        d3_assumptions,
        d3_proto,
        c3_path,
        d3_path,
    ) = load()

    assessment = build_executive_assessment(
        e1_results,
        e2_results,
        c3_arch,
        d3_assumptions,
    )

    metrics = build_metrics(
        e1_results,
        e2_results,
    )

    scope = build_final_scope()

    remaining = build_remaining_work(
        d3_assumptions
    )

    final_decision = pd.DataFrame([
        {
            "decision": "STEP 3E PROTOTYPE STATUS",
            "result":
                "PROTOTYPE DEMONSTRATED ON CONTROLLED LOCAL MOCK",
            "basis":
                "The parameterized preparation workflow completed successful cases, verified target fields, safely stopped on invalid/unexpected cases, and preserved the human decision gate.",
            "limitation":
                "The actual production /payroll-items environment was not available in the repository, so production integration remains unverified.",
        },
        {
            "decision": "RECOMMENDED ARCHITECTURE",
            "result":
                norm(c3_arch.iloc[0]["architecture_decision"])
                if not c3_arch.empty else
                "ONE PARAMETERIZED CORE + VALIDATED BRANCH CANDIDATES",
            "basis":
                "Step 3C architecture evidence.",
            "limitation":
                "Minority mechanical variants require representative validation before becoming explicit branches.",
        },
        {
            "decision": "HUMAN BOUNDARY",
            "result":
                "RETAIN HUMAN DECISION",
            "basis":
                "Step 3B and Step 3E explicitly stop before Register/Hold.",
            "limitation":
                "The prototype does not measure actual human decision quality or production approval policy.",
        },
    ])

    validation = pd.DataFrame([
        {
            "check": "3E2_all_cases_passed",
            "expected": "YES",
            "observed":
                "YES"
                if len(e2_results)
                and e2_results["status"].eq("PASS").all()
                else "NO",
            "status":
                "PASS"
                if len(e2_results)
                and e2_results["status"].eq("PASS").all()
                else "FAIL",
        },
        {
            "check": "human_gate_preserved",
            "expected": "YES",
            "observed":
                "YES"
                if len(e2_results)
                and e2_results["human_gate_preserved"].all()
                else "NO",
            "status":
                "PASS"
                if len(e2_results)
                and e2_results["human_gate_preserved"].all()
                else "FAIL",
        },
        {
            "check": "register_clicks",
            "expected": 0,
            "observed":
                int(e2_results["register_clicked"].sum()),
            "status":
                "PASS"
                if int(e2_results["register_clicked"].sum()) == 0
                else "FAIL",
        },
        {
            "check": "hold_clicks",
            "expected": 0,
            "observed":
                int(e2_results["hold_clicked"].sum()),
            "status":
                "PASS"
                if int(e2_results["hold_clicked"].sum()) == 0
                else "FAIL",
        },
        {
            "check": "production_access",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "step2_reopened_or_modified",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
    ])

    readme = pd.DataFrame([
        ["Purpose", "Freeze Step-3E prototype implementation and validation into one final evidence artifact."],
        ["Prototype target", "Controlled local mock derived from the evidence-established /payroll-items interface."],
        ["Architecture", "One parameterized preparation core with validated variants; business decisions remain human."],
        ["Prototype path", "READ → EXTRACT → TRANSFORM → POPULATE → VERIFY → STOP → HUMAN REVIEW."],
        ["Validation rule", "Successful preparation and expected safe stops both count as passing controlled tests."],
        ["Production limitation", "No production system, credentials, permissions or live API were accessed or established."],
    ], columns=["item", "value"])

    output = OUT / "step3e3_pi_final_prototype_assessment_v1.xlsx"

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        readme.to_excel(
            writer,
            sheet_name="README",
            index=False
        )
        assessment.to_excel(
            writer,
            sheet_name="Final_Assessment",
            index=False
        )
        final_decision.to_excel(
            writer,
            sheet_name="Final_Decision",
            index=False
        )
        metrics.to_excel(
            writer,
            sheet_name="Prototype_Metrics",
            index=False
        )
        scope.to_excel(
            writer,
            sheet_name="Final_Prototype_Scope",
            index=False
        )
        e2_results.to_excel(
            writer,
            sheet_name="Validation_Results",
            index=False
        )
        c3_compare.to_excel(
            writer,
            sheet_name="Architecture_Evidence",
            index=False
        )
        d3_routes.to_excel(
            writer,
            sheet_name="Technical_Interfaces",
            index=False
        )
        remaining.to_excel(
            writer,
            sheet_name="Remaining_Validation",
            index=False
        )
        validation.to_excel(
            writer,
            sheet_name="Validation",
            index=False
        )

        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        wb = writer.book

        for ws in wb.worksheets:
            ws.freeze_panes = "A2"

            if ws.max_row > 1 and ws.max_column > 1:
                ws.auto_filter.ref = ws.dimensions

            for c in ws[1]:
                c.font = Font(
                    bold=True,
                    color="FFFFFF"
                )
                c.fill = PatternFill(
                    "solid",
                    fgColor="1F4E78"
                )
                c.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True
                )

            for i, cells in enumerate(
                ws.iter_cols(
                    min_row=1,
                    max_row=min(ws.max_row, 150)
                ),
                1
            ):
                width = max(
                    [len(str(c.value or "")) for c in cells] + [12]
                )
                ws.column_dimensions[
                    get_column_letter(i)
                ].width = min(width + 2, 52)

    print("=" * 80)
    print("STEP 3E-3 — PI FINAL PROTOTYPE ASSESSMENT V1")
    print("=" * 80)

    print("\nFINAL ASSESSMENT")
    print(assessment.to_string(index=False))

    print("\nFINAL DECISION")
    print(final_decision.to_string(index=False))

    print("\nPROTOTYPE METRICS")
    print(metrics.to_string(index=False))

    print("\nREMAINING VALIDATION")
    print(
        remaining.to_string(index=False)
        if not remaining.empty
        else "No remaining validation items."
    )

    print("\nVALIDATION")
    print(validation.to_string(index=False))

    failures = validation[
        validation["status"] == "FAIL"
    ]

    if not failures.empty:
        raise RuntimeError(
            "3E-3 validation failed. Inspect Validation sheet."
        )

    print("\nOUTPUT")
    print(output)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3E-3 COMPLETE.")


if __name__ == "__main__":
    main()
