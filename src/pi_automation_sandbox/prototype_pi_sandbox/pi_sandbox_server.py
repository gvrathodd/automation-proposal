##!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse, webbrowser

ROOT=Path(__file__).resolve().parent
HTML=ROOT/"pi_sandbox.html"

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass
    def do_GET(self):
        route=self.path.split("?",1)[0]
        if route in {"/","/source-case","/payroll-items","/review"}:
            data=HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--port",type=int,default=8765)
    ap.add_argument("--open",action="store_true")
    args=ap.parse_args()
    srv=ThreadingHTTPServer((args.host,args.port),Handler)
    url=f"http://{args.host}:{args.port}/payroll-items"
    print(f"PI sandbox running at {url}")
    if args.open: webbrowser.open(url)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: srv.server_close()
if __name__=="__main__": main()
