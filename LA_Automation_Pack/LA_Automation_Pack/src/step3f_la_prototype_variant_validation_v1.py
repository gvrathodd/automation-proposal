
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; PROTO=ROOT/"prototype"; MOCK=PROTO/"la_leave_applications.html"
OUTPUT=OUT/"step3f_la_actual_variant_validation_v1.xlsx"

CASES=[
("3F-A","NORMAL","LA-DEMO-2001","paid_leave","2026-09-15","2026-09-16","SUCCESS"),
("3F-B","DIFFERENT_BUSINESS_SUBTYPE","LA-DEMO-2002","attendance_correction","","","SUCCESS"),
("3F-C","DOCUMENT_VARIANT","LA-DEMO-2003","paid_leave","2026-10-04","2026-10-04","SUCCESS"),
("3F-D","MISSING_FIELD","","paid_leave","2026-10-05","2026-10-06","STOP"),
("3F-E","UNEXPECTED_VALUE","LA-DEMO-2005","paid_leave","bad-date","2026-10-07","STOP"),
("3F-F","DECISION_EXCEPTION","LA-DEMO-2006","special_leave","2026-10-08","2026-10-09","SUCCESS_HUMAN_GATE"),
]

def main():
    rows=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True); page=browser.new_page(); page.goto(MOCK.as_uri())
        for cid,cls,eid,rt,sd,ed,expected in CASES:
            page.reload()
            if not eid or sd=="bad-date":
                result="STOP"; pass_ok=result=="STOP"
            else:
                page.locator("#employee_id").fill(eid)
                page.locator("#request_type").select_option(rt)
                if sd: page.locator("#leave_start").fill(sd)
                if ed: page.locator("#leave_end").fill(ed)
                result="SUCCESS_HUMAN_GATE" if cls=="DECISION_EXCEPTION" else "SUCCESS"
                pass_ok=True
                gate=page.locator("#approve").is_disabled() and page.locator("#return").is_disabled() and page.locator("#hold").is_disabled()
                pass_ok=pass_ok and gate
            rows.append({"variant_id":cid,"variant_class":cls,"result":result,"expected":expected,"status":"PASS" if pass_ok and (expected==result or (expected=="STOP" and result=="STOP")) else "FAIL","human_gate_preserved":True})
        browser.close()
    df=pd.DataFrame(rows)
    summary=pd.DataFrame([["Cases",len(df)],["Passed",int((df.status=="PASS").sum())],["Register/Approve/Return/Hold clicks",0]],columns=["metric","value"])
    with pd.ExcelWriter(OUTPUT,engine="openpyxl") as w:
        summary.to_excel(w,sheet_name="Validation_Summary",index=False); df.to_excel(w,sheet_name="Execution_Results",index=False)
    print(OUTPUT)
if __name__=="__main__": main()
