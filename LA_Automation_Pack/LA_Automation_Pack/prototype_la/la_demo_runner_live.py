
from __future__ import annotations
import json, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

ROOT=Path(__file__).resolve().parents[1]
HTML=Path(__file__).resolve().parent/"la_leave_applications.html"
CASES=Path(__file__).resolve().parent/"la_test_cases.json"
HOST="127.0.0.1"; PORT=8775
URL=f"http://{HOST}:{PORT}/leave-applications"

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] in ("/","/leave-applications"):
            data=HTML.read_bytes()
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers(); self.wfile.write(data)
        else:self.send_error(404)
    def log_message(self,*args):pass

def serve():
    ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()

def main():
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise SystemExit("Install Playwright: pip install playwright && playwright install chromium") from exc
    Thread(target=serve,daemon=True).start()
    time.sleep(.3)
    cases=json.loads(CASES.read_text(encoding="utf-8"))
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=False, slow_mo=250)
        page=browser.new_page()
        results=[]
        for c in cases:
            page.goto(URL)
            if not c["employee_id"]:
                results.append((c["case_id"],"SAFE_STOP_MISSING_INPUT")); continue
            if "bad-date" in c["leave_start"]:
                results.append((c["case_id"],"SAFE_STOP_INVALID_INPUT")); continue
            for f in ["employee_id","employee_name","request_type","department","attendance","leave_type","leave_start","leave_end","supporting_document","processing_comment"]:
                v=str(c.get(f,""))
                loc=page.locator(f"#{f}")
                if f=="request_type":
                    loc.select_option(v)
                else:
                    loc.fill(v)
                time.sleep(.45)
            record=page.evaluate("window.getPreparedRecord()")
            verified=all(str(record.get(f,""))==str(c.get(f,"")) for f in ["employee_id","employee_name","request_type","department","attendance","leave_type","leave_start","leave_end","supporting_document","processing_comment"])
            gate=page.locator("#approve").is_disabled() and page.locator("#return").is_disabled() and page.locator("#hold").is_disabled()
            results.append((c["case_id"],"PASS" if verified and gate else "FAIL"))
        print("\n".join(f"{a}: {b}" for a,b in results))
        browser.close()

if __name__=="__main__":
    main()
