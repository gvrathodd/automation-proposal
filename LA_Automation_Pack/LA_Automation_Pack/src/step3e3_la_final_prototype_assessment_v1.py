
from __future__ import annotations
from pathlib import Path
import pandas as pd
from la_common import OUT, excel_write

E2=OUT/"step3e2_la_prototype_validation_v1.xlsx"
C3=OUT/"step3c_la_parameterized_vs_branch_analysis_v1.xlsx"
D3=OUT/"step3d_la_technical_interface_discovery_v1.xlsx"
OUTPUT=OUT/"step3e3_la_final_prototype_assessment_v1.xlsx"

def main():
    r=pd.read_excel(E2,sheet_name="Execution_Results")
    summary=pd.DataFrame([
        ["Validation cases",len(r)],
        ["Passes",int((r["status"]=="PASS").sum()) if not r.empty else 0],
        ["Pass rate",round(100*(r["status"]=="PASS").mean(),2) if not r.empty else 0],
        ["Production access",False],
        ["Human decisions automated",False],
        ["Architecture","ONE PARAMETERIZED CORE + VALIDATED BRANCHES"],
        ["Safety invariant","No Approve / Return / Hold clicks by automation"],
    ],columns=["metric","value"])
    assessment=pd.DataFrame([
        ["Verified","Local mechanical preparation against reconstructed interface"],
        ["Verified","Missing/invalid input causes safe stop"],
        ["Verified","Human decision gate remains preserved"],
        ["Not verified","Production UI compatibility"],
        ["Not verified","Production business-rule correctness"],
        ["Not verified","Production timing/savings"],
    ],columns=["status","statement"])
    excel_write(OUTPUT,{"Executive_Assessment":summary,"Assessment":assessment,"Input_Validation":r})
    print(OUTPUT)

if __name__=="__main__": main()
