#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys,time,re,json

ROOT=Path(__file__).resolve().parent
SERVER=ROOT/"pi_sandbox_server.py"
BASE="http://127.0.0.1:8765"

CASES=[
("PI-REAL-001","SUCCESS"),
("PI-REAL-002","SUCCESS"),
("PI-REAL-003","SUCCESS"),
("PI-REAL-004","STOP_MISSING_REQUIRED_INPUT"),
("PI-REAL-005","STOP_INVALID_INPUT"),
("PI-REAL-006","SUCCESS_HUMAN_GATE"),
]

def main():
    from playwright.sync_api import sync_playwright
    srv=subprocess.Popen([sys.executable,str(SERVER)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    time.sleep(.4)
    results=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page()
            for case_id,expected in CASES:
                print(f"Running {case_id} ({expected})...")
                page.goto(f"{BASE}/payroll-items",wait_until="domcontentloaded")
                page.select_option("#caseSelector",case_id)
                page.click("#loadCase")
                source={}
                for field in ["employee_id","employee_name","request_type","category","amount","effective_date","status","document","processing_comment"]:
                    v=page.locator(f"#source-{field}").inner_text()
                    source[field]="" if v=="MISSING" else v
                if not source["employee_id"]:
                    actual="STOP_MISSING_REQUIRED_INPUT"; fields=0
                elif not re.fullmatch(r"-?\d+(?:\.\d{1,2})?",source["amount"].replace(",","")):
                    actual="STOP_INVALID_INPUT"; fields=0
                else:
                    page.fill("#employee_id",source["employee_id"]);page.fill("#employee_name",source["employee_name"])
                    page.select_option("#request_type",source["request_type"]);page.fill("#category",source["category"])
                    page.fill("#amount",source["amount"].replace(",",""));page.fill("#effective_date",source["effective_date"])
                    page.fill("#status",source["status"]);page.fill("#document",source["document"]);page.fill("#processing_comment",source["processing_comment"])
                    record=page.evaluate("window.getPreparedRecord()")
                    expected_record={k:(v.replace(",","") if k=="amount" else v) for k,v in source.items()}
                    fields=sum(record[k]==expected_record[k] for k in expected_record)
                    actual="SUCCESS_HUMAN_GATE" if source["status"].lower()=="needs review" or "exception" in source["processing_comment"].lower() else "SUCCESS"
                gate=page.locator("#register").count()==1 and page.locator("#hold").count()==1
                # Test that neither control was clicked by automation: sandbox trace has no manual action.
                trace=page.locator("#trace").inner_text()
                clicked=("Register clicked" in trace) or ("Hold clicked" in trace)
                passed=(actual==expected and gate and not clicked)
                results.append({"case_id":case_id,"expected":expected,"actual":actual,"fields_verified":fields,"human_gate_present":gate,"register_or_hold_clicked":clicked,"status":"PASS" if passed else "FAIL"})
                print(f"  -> {actual} [{results[-1]['status']}]")
            browser.close()
    finally:
        srv.terminate()
        try:srv.wait(timeout=3)
        except subprocess.TimeoutExpired:srv.kill()
    print("="*80);print("PI SANDBOX DEMO RESULT");print("="*80)
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
