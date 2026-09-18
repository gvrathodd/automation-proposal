
from __future__ import annotations
import json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
PROTO=ROOT/"prototype"
HTML=PROTO/"la_leave_applications.html"
CASES=PROTO/"la_test_cases.json"
OUTPUT=OUT/"step3e1_la_prototype_evidence_v1.xlsx"
HOST="127.0.0.1"; PORT=8775
URL=f"http://{HOST}:{PORT}/leave-applications"

CASES_DATA=[
 {"case_id":"LA-E1-001","class":"NORMAL","request_type":"paid_leave","employee_id":"LA-DEMO-1001","employee_name":"Demo Employee 1","department":"Operations","attendance":"Regular","leave_type":"Paid Leave","leave_start":"2026-09-21","leave_end":"2026-09-22","processing_comment":"Standard leave preparation.","supporting_document":"","expected":"SUCCESS"},
 {"case_id":"LA-E1-002","class":"PARAMETER_VARIATION","request_type":"attendance_correction","employee_id":"LA-DEMO-1002","employee_name":"Demo Employee 2","department":"HR","attendance":"Correction requested","leave_type":"","leave_start":"","leave_end":"","processing_comment":"Attendance correction preparation.","supporting_document":"","expected":"SUCCESS"},
 {"case_id":"LA-E1-003","class":"DOCUMENT_VARIANT","request_type":"paid_leave","employee_id":"LA-DEMO-1003","employee_name":"Demo Employee 3","department":"Finance","attendance":"Regular","leave_type":"Paid Leave","leave_start":"2026-10-03","leave_end":"2026-10-03","processing_comment":"Document-assisted leave preparation.","supporting_document":"leave_note_003.pdf","expected":"SUCCESS"},
 {"case_id":"LA-E1-004","class":"MISSING_INPUT","request_type":"paid_leave","employee_id":"","employee_name":"Demo Employee 4","department":"Operations","attendance":"Regular","leave_type":"Paid Leave","leave_start":"2026-09-25","leave_end":"2026-09-26","processing_comment":"Missing employee ID.","supporting_document":"","expected":"STOP_MISSING_REQUIRED_INPUT"},
 {"case_id":"LA-E1-005","class":"INVALID_INPUT","request_type":"paid_leave","employee_id":"LA-DEMO-1005","employee_name":"Demo Employee 5","department":"Operations","attendance":"Regular","leave_type":"Paid Leave","leave_start":"bad-date","leave_end":"2026-09-30","processing_comment":"Invalid date.","supporting_document":"","expected":"STOP_INVALID_INPUT"},
 {"case_id":"LA-E1-006","class":"DECISION_EXCEPTION","request_type":"paid_leave","employee_id":"LA-DEMO-1006","employee_name":"Demo Employee 6","department":"HR","attendance":"Needs review","leave_type":"Special Leave","leave_start":"2026-10-05","leave_end":"2026-10-06","processing_comment":"Exception requires human review.","supporting_document":"supporting_006.pdf","expected":"SUCCESS_HUMAN_GATE"},
]

HTML_TEXT="""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Leave Applications - Local Prototype</title>
<style>body{font-family:Arial;margin:30px;background:#f6f7f9}.wrap{max-width:1100px;margin:auto}.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:18px;margin-bottom:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{font-weight:600}input,select,textarea{font:inherit;padding:8px}textarea{min-height:70px}.full{grid-column:1/-1}.gate{border:2px solid #444}.buttons{display:flex;gap:10px}button{padding:10px 16px}.result{background:#eef2f6;padding:12px;white-space:pre-wrap}</style></head>
<body><div class="wrap">
<div class="card"><h1>Leave Applications - Local Prototype</h1><div>Route: <b>/leave-applications</b></div></div>
<div class="card"><h2>Prepared Attendance / Leave Record</h2><div class="grid">
<div><label>Employee ID<input id="employee_id"></label></div><div><label>Employee Name<input id="employee_name"></label></div>
<div><label>Request Type<select id="request_type"><option value="">Select...</option><option value="paid_leave">Paid leave</option><option value="attendance_correction">Attendance correction</option><option value="special_leave">Special leave</option></select></label></div>
<div><label>Department<input id="department"></label></div>
<div><label>Attendance<input id="attendance"></label></div><div><label>Leave Type<input id="leave_type"></label></div>
<div><label>Leave Start<input id="leave_start" type="date"></label></div><div><label>Leave End<input id="leave_end" type="date"></label></div>
<div><label>Status<input id="status" readonly value="Prepared - Human Review Required"></label></div>
<div><label>Supporting Document<input id="supporting_document"></label></div>
<div class="full"><label>Processing Comment<textarea id="processing_comment"></textarea></label></div>
</div></div>
<div class="card gate"><h2>Human Review Gate</h2><p>Automation stops before business decision.</p><div class="buttons"><button id="approve" disabled>Approve</button><button id="return" disabled>Return for correction</button><button id="hold" disabled>Hold</button></div></div>
<div class="card"><h2>Verification</h2><div id="result" class="result">Waiting...</div></div>
<script>
window.LA_CASES=%CASES%;
window.getPreparedRecord=function(){let ids=["employee_id","employee_name","request_type","department","attendance","leave_type","leave_start","leave_end","status","supporting_document","processing_comment"];let o={};ids.forEach(id=>o[id]=document.querySelector("#"+id).value);return o;};
window.markVerification=function(ok,msg){let e=document.querySelector("#result");e.dataset.verification=ok?"PASS":"FAIL";e.textContent=msg;};
</script></div></body></html>"""

def write_mock():
    PROTO.mkdir(exist_ok=True)
    HTML.write_text(HTML_TEXT.replace("%CASES%", json.dumps(CASES_DATA, ensure_ascii=False)), encoding="utf-8")
    CASES.write_text(json.dumps(CASES_DATA,indent=2,ensure_ascii=False),encoding="utf-8")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] in ("/","/leave-applications"):
            data=HTML.read_bytes(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers(); self.wfile.write(data)
        else: self.send_error(404)
    def log_message(self,*args): pass

def server():
    ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()

def main():
    write_mock()
    print("Prototype written:",HTML)
    print("Install: pip install playwright pandas openpyxl && playwright install chromium")
    print("Run via the live/headless runner in prototype/")
if __name__=="__main__": main()
