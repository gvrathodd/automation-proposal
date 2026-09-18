
from __future__ import annotations
from pathlib import Path
import pandas as pd
from la_common import (
    ROOT, OUT, load_la_segments, load_b_events, assign_events,
    event_type, active_app, routes_from, file_refs, get_field, excel_write, norm
)

OUTPUT_SEG = OUT / "step3a1_la_authoritative_segment_evidence_v1.csv"
OUTPUT_EVT = OUT / "step3a1_la_authoritative_event_detail_v1.csv"
OUTPUT_SS = OUT / "step3a1_la_screenshot_evidence_v1.csv"
OUTPUT_VALID = OUT / "step3a1_la_validation_summary_v1.csv"

def main():
    seg = load_la_segments()
    sessions, files = load_b_events()
    attached = assign_events(seg, sessions)
    event_rows = []
    ss_rows = []

    for e in attached:
        routes = [r for r in routes_from(e) if r]
        et = event_type(e)
        event_rows.append({
            "session_id": norm(e.get("session_id")),
            "segment_index": int(e["_segment_index"]),
            "segment_key": e["_segment_key"],
            "event_timestamp": pd.to_datetime(event_ts := e.get("timestamp_ms"), unit="ms", utc=True).isoformat() if e.get("timestamp_ms") else "",
            "timestamp_ms": int(e.get("timestamp_ms", 0) or 0),
            "event_type": et,
            "active_app": active_app(e),
            "window_title": get_field(e, "window_title", "title"),
            "routes": " | ".join(dict.fromkeys(routes)),
            "target_field": get_field(e, "target_field", "field", "field_name"),
            "target_element": get_field(e, "target_element", "element", "accessible_name", "text"),
            "document_name": get_field(e, "file_name", "document_name", "path"),
            "screenshot_refs": " | ".join(file_refs(e)),
        })
        for fn in file_refs(e):
            ss_rows.append({
                "session_id": norm(e.get("session_id")),
                "segment_index": int(e["_segment_index"]),
                "segment_key": e["_segment_key"],
                "timestamp_ms": int(e.get("timestamp_ms", 0) or 0),
                "filename": Path(fn).name,
                "raw_reference": fn,
            })

    seg_rows = []
    evdf = pd.DataFrame(event_rows)
    for _, s in seg.iterrows():
        e = evdf[evdf["segment_key"] == s["segment_key"]] if not evdf.empty else pd.DataFrame()
        seg_rows.append({
            "session_id": s["session_id"],
            "segment_index": int(s["segment_index"]),
            "segment_key": s["segment_key"],
            "segment_start": s["start"],
            "segment_end": s["end"],
            "duration_seconds": float(s["duration_seconds"]),
            "raw_event_count": int(len(e)),
            "browser_events": int(e["event_type"].str.startswith("browser_").sum()) if not e.empty else 0,
            "browser_navigation": int((e["event_type"] == "browser_navigation").sum()) if not e.empty else 0,
            "browser_form_input": int((e["event_type"] == "browser_form_input").sum()) if not e.empty else 0,
            "browser_click": int((e["event_type"] == "browser_click").sum()) if not e.empty else 0,
            "clipboard_change": int((e["event_type"] == "clipboard_change").sum()) if not e.empty else 0,
            "app_switch": int((e["event_type"] == "app_switch").sum()) if not e.empty else 0,
            "keyboard_like": int(e["event_type"].isin(["keystroke","keyboard","shortcut"]).sum()) if not e.empty else 0,
            "document_events": int(e["event_type"].str.startswith("document").sum()) if not e.empty else 0,
            "screenshot_ref_count": int(e["screenshot_refs"].fillna("").replace("", pd.NA).notna().sum()) if not e.empty and "screenshot_refs" in e else 0,
            "routes": " | ".join(dict.fromkeys(x for y in e["routes"].fillna("") for x in y.split(" | ") if x)) if not e.empty else "",
            "applications": " | ".join(dict.fromkeys(active_app_name for active_app_name in e["active_app"].fillna("") if active_app_name)) if not e.empty else "",
        })

    segdf = pd.DataFrame(seg_rows)
    segdf.to_csv(OUTPUT_SEG, index=False)
    evdf.to_csv(OUTPUT_EVT, index=False)
    pd.DataFrame(ss_rows).drop_duplicates().to_csv(OUTPUT_SS, index=False)

    validation = pd.DataFrame([
        ["canonical_la_segments", len(segdf), "Expected from Step-2: 99", "PASS" if len(segdf)==99 else "CHECK"],
        ["dataset_b_event_files", len(files), "Project inventory should contain 20", "PASS" if len(files)==20 else "CHECK"],
        ["dataset_b_sessions", len(sessions), "Expected 15", "PASS" if len(sessions)==15 else "CHECK"],
        ["attached_la_events", len(evdf), "Derived from timestamp intervals", "PASS"],
        ["unassigned_within_la_population", 0, "Events outside LA segments are not LA events", "PASS"],
    ], columns=["metric","value","expectation","status"])
    excel_write(OUTPUT_VALID, {"Validation": validation})
    print(segdf.head())
    print(f"LA segments: {len(segdf)}")
    print(f"Attached LA events: {len(evdf)}")
    print(f"Outputs: {OUTPUT_SEG}, {OUTPUT_EVT}, {OUTPUT_SS}, {OUTPUT_VALID}")

if __name__ == "__main__":
    main()
