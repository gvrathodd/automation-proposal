
from __future__ import annotations
from pathlib import Path
import re, pandas as pd
from la_common import ROOT, OUT, excel_write

OUTPUT = OUT / "step3e0_la_application_target_discovery_v1.xlsx"
EXCLUDED = {".git",".venv","venv",".venv_vision","outputs","src","__pycache__",".pytest_cache","node_modules","dist","build"}
TEXT_EXT = {".py",".js",".jsx",".ts",".tsx",".html",".htm",".json",".yaml",".yml",".toml",".md",".txt",".css",".scss"}

def main():
    hits=[]
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        if any(x.lower() in EXCLUDED for x in p.parts):
            continue
        try:
            txt=p.read_text(encoding="utf-8",errors="ignore")
        except Exception:
            continue
        if re.search(r"/leave-applications",txt,re.I):
            hits.append({"file":str(p.relative_to(ROOT)),"route_occurrence":True,
                         "looks_executable":bool(re.search(r"(FastAPI|Flask|createRoot|BrowserRouter|<Route|app\.get|app\.post)",txt,re.I)),
                         "mock_terms":bool(re.search(r"\b(mock|fixture|stub|dummy|fake)\b",txt,re.I))})
    df=pd.DataFrame(hits)
    if df.empty:
        cls="NO EXECUTABLE TARGET EVIDENCE"
    elif df["looks_executable"].any() and not df["mock_terms"].all():
        cls="POSSIBLE LOCAL TARGET — MANUAL VERIFY"
    else:
        cls="MOCK/REFERENCE ONLY — MANUAL VERIFY"
    summary=pd.DataFrame([
        ["route","/leave-applications"],
        ["classification",cls],
        ["principle","A route string alone is not proof of a runnable production target."],
    ],columns=["item","value"])
    excel_write(OUTPUT,{"README":summary,"Evidence_Files":df})
    print(OUTPUT)

if __name__=="__main__":
    main()
