
from __future__ import annotations
from pathlib import Path
import pandas as pd
from la_common import OUT, load_b_events, load_la_segments, assign_events, event_type, routes_from, phase_for_event, excel_write

SEG=OUT/"step3a1_la_authoritative_segment_evidence_v1.csv"
OUTPUT=OUT/"step3h_la_real_dataset_replay_v1.xlsx"

def main():
    seg=load_la_segments()
    sessions, files=load_b_events()
    attached=assign_events(seg,sessions)
    ev=pd.DataFrame(attached)
    if not ev.empty:
        ev["event_type"]=ev["event_type"].astype(str)
        ev["phase"]=ev.apply(phase_for_event,axis=1)
        ev["has_route"]=ev.apply(lambda x:any("/leave-applications" in r.lower() for r in routes_from(x.to_dict())),axis=1)
    rows=[]
    for _,s in seg.iterrows():
        key=s["segment_key"]; g=ev[ev["_segment_key"]==key] if not ev.empty else pd.DataFrame()
        has_target=bool(g["has_route"].any()) if not g.empty else False
        has_form=bool((g["event_type"]=="browser_form_input").any()) if not g.empty else False
        has_transfer=bool((g["event_type"]=="clipboard_change").any()) if not g.empty else False
        has_decision=bool(g["event_type"].astype(str).str.contains("click").any()) if not g.empty else False
        if not has_target and not has_form:
            cls="INSUFFICIENT_TARGET_EVIDENCE"
        elif has_decision and has_transfer:
            cls="HUMAN_GATE"
        elif has_target and (has_form or has_transfer):
            cls="REPLAYABLE_WITH_VARIANT"
        else:
            cls="STRUCTURAL_ONLY"
        rows.append({"session_id":s["session_id"],"segment_index":int(s["segment_index"]),"segment_key":key,
                     "real_event_count":int(len(g)),"target_route":has_target,"form_input":has_form,
                     "clipboard":has_transfer,"decision_evidence":has_decision,"replay_class":cls})
    replay=pd.DataFrame(rows)
    summary=pd.DataFrame([
        ["canonical_la_segments",len(seg)],
        ["real_la_event_rows",len(ev)],
        ["raw_dataset_b_sessions",len(sessions)],
        ["raw_dataset_b_event_files",len(files)],
        ["production_access",False],
        ["purpose","Offline evidence replay using real Dataset-B events attached by timestamp interval"],
    ],columns=["metric","value"])
    counts=replay["replay_class"].value_counts().rename_axis("replay_class").reset_index(name="segments")
    excel_write(OUTPUT,{"README":summary,"Segment_Replay":replay,"Replay_Summary":counts})
    print(OUTPUT)
if __name__=="__main__": main()
