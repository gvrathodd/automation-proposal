
from __future__ import annotations
import pandas as pd
from la_common import OUT, excel_write

A1 = OUT / "step3a1_la_authoritative_segment_evidence_v1.csv"
A4 = OUT / "step3a4_la_workflow_map_v1.xlsx"
OUTPUT = OUT / "step3b_la_automation_boundary_v1.xlsx"

def main():
    seg = pd.read_csv(A1)
    phases = pd.read_excel(A4, sheet_name="Segment_Workflow_Sequences")
    rows=[]
    treatments = {
        "BROWSER_NAVIGATION":"AUTOMATE",
        "BROWSER_WORK":"AUTOMATE + FLAG",
        "APPLICATION_SWITCH":"AUTOMATE",
        "DATA_TRANSFER":"AUTOMATE + FLAG",
        "FORM_ENTRY":"AUTOMATE + FLAG",
        "DOCUMENT_WORK":"AUTOMATE + FLAG",
        "DESKTOP_INPUT":"VALIDATE",
    }
    for p, treatment in treatments.items():
        present = int(phases["clean_phase_sequence"].fillna("").str.contains(p, regex=False).sum()) if not phases.empty else 0
        rows.append({
            "phase":p,"segments_affected":present,
            "treatment":treatment,
            "rationale":{
                "BROWSER_NAVIGATION":"Predictable navigation into the observed leave-application UI.",
                "BROWSER_WORK":"Automate expected states, stop on unexpected UI.",
                "APPLICATION_SWITCH":"Deterministic context movement where needed.",
                "DATA_TRANSFER":"Strong clipboard/transfer evidence; validate target values.",
                "FORM_ENTRY":"Structured target field population; validate required fields.",
                "DOCUMENT_WORK":"Only where document requirements are deterministic.",
                "DESKTOP_INPUT":"Source-side evidence varies; validate at runtime."
            }.get(p,"")
        })
    human = pd.DataFrame([
        ["review","HUMAN","Review prepared attendance/leave record"],
        ["approve","HUMAN","Business decision"],
        ["return_for_correction","HUMAN","Exception/correction decision"],
        ["hold","HUMAN","Business hold decision"],
        ["policy_interpretation","HUMAN","Interpretation cannot be inferred safely from UI interaction alone"],
        ["final_release","HUMAN","Final business action remains outside automation"],
    ], columns=["action","treatment","reason"])
    summary = pd.DataFrame([
        ["Core scope","READ → TRANSFER → TRANSFORM → FORM ENTRY → VERIFY → HUMAN GATE"],
        ["Out of scope","Approve / return / hold / policy / ambiguity / exception resolution / final release"],
        ["Family","LA — Attendance / Leave Application"],
        ["Observed route","/leave-applications"],
    ], columns=["item","value"])
    excel_write(OUTPUT, {"README":summary,"Automation_Boundary":pd.DataFrame(rows),"Human_Gate_Evidence":human})
    print(OUTPUT)

if __name__ == "__main__":
    main()
