
from __future__ import annotations
from pathlib import Path
import pandas as pd
from la_common import OUT, excel_write

A1_SEG = OUT / "step3a1_la_authoritative_segment_evidence_v1.csv"
A2 = OUT / "step3a2_la_chronological_workflow_evidence_v1.xlsx"
A3 = OUT / "step3a3_la_ocr_semantic_evidence_v1.xlsx"
OUTPUT = OUT / "step3a4_la_workflow_map_v1.xlsx"

CORE_THRESHOLD = 0.50
ORDER_THRESHOLD = 0.70

def main():
    seg = pd.read_csv(A1_SEG)
    seq = pd.read_excel(A2, sheet_name="Segment_Workflow_Sequences")
    sem = pd.read_excel(A3, sheet_name="LA_Semantic_Profiles")
    total = len(seg)

    phases = ["BROWSER_NAVIGATION","BROWSER_WORK","DATA_TRANSFER","APPLICATION_SWITCH","DESKTOP_INPUT","DOCUMENT_WORK"]
    rows=[]
    for p in phases:
        present = int(seq["clean_phase_sequence"].fillna("").str.contains(p, regex=False).sum()) if not seq.empty else 0
        rows.append({
            "phase":p, "segment_count":present,
            "segment_coverage_pct":round(100*present/total,2) if total else 0,
            "classification":"COMMON_CORE" if present/total >= CORE_THRESHOLD else ("FREQUENT" if present/total >= .25 else "OPTIONAL")
        })
    workflow_map = pd.DataFrame(rows)

    order_rows=[]
    phase_sets = seq.set_index(["session_id","segment_index"])["clean_phase_sequence"].fillna("").apply(lambda x:[t.strip() for t in str(x).split("→") if t.strip()])
    for a in phases:
        for b in phases:
            if a==b: continue
            common=not_a=0
            for arr in phase_sets:
                if a in arr and b in arr:
                    if arr.index(a) < arr.index(b):
                        common += 1
                    else:
                        not_a += 1
            support = common / (common+not_a) if common+not_a else 0
            if common+not_a >= 10:
                order_rows.append({"first_phase":a,"second_phase":b,"common":common,"not_common":not_a,"support_pct":round(100*support,2),
                                    "classification":"COMMON_ORDER" if support>=ORDER_THRESHOLD else ("FREQUENT_ORDER" if support>=.55 else "OTHER")})
    orders = pd.DataFrame(order_rows).sort_values("support_pct", ascending=False)

    summary = pd.DataFrame([
        ["Population", total],
        ["Expected Step-2 LA family", "99"],
        ["Direct OCR semantic segments", int(sem["direct_ocr_present"].sum()) if not sem.empty else 0],
        ["Route", "/leave-applications"],
        ["Business interpretation", "Attendance / leave application family; observed OCR case types include attendance, leave, paid leave, flexible scheduling."],
        ["Decision caution", "Approve/return/hold controls are evidence only; business judgement stays human."]
    ], columns=["item","value"])

    excel_write(OUTPUT, {"README":summary,"Workflow_Map":workflow_map,"Common_Order_Evidence":orders,
                         "Segment_Workflow_Sequences":seq,"Semantic_By_Segment":sem})
    print(OUTPUT)

if __name__ == "__main__":
    main()
