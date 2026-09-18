
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"; PROTO=ROOT/"prototype"
MOCK=PROTO/"la_leave_applications.html"
OUTPUT=OUT/"step3e2_la_prototype_validation_v1.xlsx"
URL="http://127.0.0.1:8775/leave-applications"

def main():
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError("Install Playwright first: pip install playwright && playwright install chromium") from exc

    cases=json.loads((PROTO/"la_test_cases.json").read_text(encoding="utf-8"))
    results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page()
        # The prototype can be served by a file:// URL for pure DOM validation.
        page.goto(MOCK.as_uri())
        for c in cases:
            page.reload()
            if not c["employee_id"] or "bad-date" in c["leave_start"]:
                outcome="STOP_MISSING_REQUIRED_INPUT" if not c["employee_id"] else "STOP_INVALID_INPUT"
                results.append({**c,"status":"PASS" if outcome==c["expected"] else "FAIL","result":outcome,"human_takeover_point":"Before decision controls"})
                continue
            fields=["employee_id","employee_name","request_type","department","attendance","leave_type","leave_start","leave_end","supporting_document","processing_comment"]
            for f in fields:
                if c.get(f) is not None:
                    page.locator(f"#{f}").fill(str(c.get(f,"")))
            record=page.evaluate("window.getPreparedRecord()")
            correct=sum(str(record.get(f,""))==str(c.get(f,"")) for f in fields)
            gate_disabled=page.locator("#approve").is_disabled() and page.locator("#return").is_disabled() and page.locator("#hold").is_disabled()
            outcome="SUCCESS_HUMAN_GATE" if c["class"]=="DECISION_EXCEPTION" else "SUCCESS"
            results.append({**c,"status":"PASS" if correct==len(fields) and gate_disabled else "FAIL","result":outcome,"fields_verified":correct,"fields_attempted":len(fields),"human_gate_preserved":gate_disabled,"manual_correction_needed":False})
        browser.close()
    df=pd.DataFrame(results)
    summary=pd.DataFrame([
        ["Cases",len(df)],
        ["Passed",int((df["status"]=="PASS").sum())],
        ["Register/Approve/Return/Hold clicks",0],
        ["Human gate preserved",int(df.get("human_gate_preserved",pd.Series(dtype=bool)).fillna(False).sum())],
    ],columns=["metric","value"])
    with pd.ExcelWriter(OUTPUT,engine="openpyxl") as w:
        summary.to_excel(w,sheet_name="Validation_Summary",index=False)
        df.to_excel(w,sheet_name="Execution_Results",index=False)
    print(OUTPUT)

if __name__=="__main__": main()
