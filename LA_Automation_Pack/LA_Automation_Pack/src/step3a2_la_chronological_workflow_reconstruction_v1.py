
from __future__ import annotations
from pathlib import Path
from collections import Counter
import pandas as pd
from la_common import OUT, excel_write, norm, phase_for_event, classify_app

A1_EVT = OUT / "step3a1_la_authoritative_event_detail_v1.csv"
A1_SEG = OUT / "step3a1_la_authoritative_segment_evidence_v1.csv"
OUTPUT = OUT / "step3a2_la_chronological_workflow_evidence_v1.xlsx"

def parse_key(seq):
    out=[]
    for x in seq:
        if x and (not out or out[-1] != x):
            out.append(x)
    return out

def main():
    ev = pd.read_csv(A1_EVT)
    seg = pd.read_csv(A1_SEG)
    if ev.empty:
        raise RuntimeError("No LA events. Run Step 3A-1 first.")
    ev = ev.sort_values(["session_id","segment_index","timestamp_ms"])
    ev["phase"] = ev.apply(lambda r: r["event_type"] if False else "", axis=1)

    # Reconstruct phase from event type using same rules as PI architecture.
    raw = []
    for (sid, idx), g in ev.groupby(["session_id","segment_index"], sort=False):
        phases = []
        apps = []
        routes = []
        for _, r in g.iterrows():
            fake = {"event_type": r["event_type"], "context": {"active_app": {"name": r["active_app"]}}, "payload": {}}
            p = phase_for_event(fake)
            if p != "DESKTOP_INPUT" or not phases:
                phases.append(p)
            app = classify_app(norm(r["active_app"]))
            if app and (not apps or apps[-1] != app):
                apps.append(app)
            if norm(r["routes"]):
                for x in norm(r["routes"]).split(" | "):
                    if x and x not in routes:
                        routes.append(x)
        raw.append({
            "session_id": sid,
            "segment_index": int(idx),
            "clean_phase_sequence": " → ".join(parse_key(phases)),
            "clean_application_sequence": " → ".join(parse_key(apps)),
            "routes": " | ".join(routes),
            "event_count": int(len(g)),
            "has_clipboard": int((g["event_type"]=="clipboard_change").any()),
            "has_form_input": int((g["event_type"]=="browser_form_input").any()),
            "has_navigation": int((g["event_type"]=="browser_navigation").any()),
            "has_document": int(g["event_type"].astype(str).str.startswith("document").any()),
        })
    seq = pd.DataFrame(raw)

    pattern = (seq.groupby(["clean_phase_sequence","clean_application_sequence"], dropna=False)
                 .agg(segment_count=("segment_index","size"),
                      session_count=("session_id","nunique"))
                 .reset_index()
                 .sort_values(["segment_count","session_count"], ascending=False))

    validation = pd.DataFrame([
        ["segments_from_3A1", len(seg), len(seq), "PASS" if len(seg)==len(seq) else "CHECK"],
        ["cross_session_patterns", int((pattern["session_count"]>=2).sum()) if not pattern.empty else 0, "descriptive", "PASS"],
    ], columns=["metric","value","expectation","status"])

    excel_write(OUTPUT, {
        "README": pd.DataFrame([["Purpose","Chronological LA workflow reconstruction"],["Population",str(len(seg))],["Note","Structural evidence only; not subprocess truth."]], columns=["item","value"]),
        "Segment_Workflow_Sequences": seq,
        "Workflow_Patterns": pattern,
        "Validation": validation
    })
    print(OUTPUT)

if __name__ == "__main__":
    main()
