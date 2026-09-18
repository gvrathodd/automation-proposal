
from __future__ import annotations
import pandas as pd
from la_common import OUT, excel_write

A3 = OUT / "step3a3_la_ocr_semantic_evidence_v1.xlsx"
A4 = OUT / "step3a4_la_workflow_map_v1.xlsx"
OUTPUT = OUT / "step3c_la_parameterized_vs_branch_analysis_v1.xlsx"

def main():
    sem = pd.read_excel(A3, sheet_name="LA_Semantic_Profiles")
    seq = pd.read_excel(A4, sheet_name="Segment_Workflow_Sequences")

    observed = []
    if not sem.empty:
        for _, r in sem.iterrows():
            observed.append({
                "session_id":r["session_id"],"segment_index":int(r["segment_index"]),
                "case_family": "attendance_leave",
                "business_terms": r.get("business_terms",""),
                "request_type": r.get("request_type",""),
                "field_names": r.get("field_names",""),
                "decision_options": r.get("decision_options","")
            })
    evidence = pd.DataFrame(observed)

    architecture = pd.DataFrame([
        ["Architecture","ONE PARAMETERIZED CORE + VALIDATED BRANCHES"],
        ["Parameterization rule","Same mechanical skeleton + different employee/leave/attendance data = parameter"],
        ["Optional variant","Supporting document or desktop-tool variation does not automatically create a branch"],
        ["Branch rule","Create a branch only when a meaningfully different mechanical skeleton has repeatable support"],
        ["Human gate","Different approval/return/hold state is a human gate, not an automation branch"],
        ["No extrapolation","OCR-observed case types are branch evidence only; no prevalence extrapolation"],
    ], columns=["item","value"])

    comparison = pd.DataFrame([
        ["attendance_leave","/leave-applications","attendance | leave | paid_leave | flexible_schedule",
         "Shared browser/transfer/form core; use parameterized case data first."],
    ], columns=["case_type","route","semantic_evidence","architecture_interpretation"])

    excel_write(OUTPUT, {"README":architecture,"Case_Evidence":evidence,"Case_Type_Path_Comparison":comparison,
                         "Observed_Workflow_Sequences":seq})
    print(OUTPUT)

if __name__ == "__main__":
    main()
