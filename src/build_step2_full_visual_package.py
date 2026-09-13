from pathlib import Path
from collections import defaultdict
import shutil
import pandas as pd
from loader import load_all_events, sort_session_events, get_timestamp_ms

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
SEGMENTS = ROOT / "outputs" / "dataset_b_segments.csv"
OCC = ROOT / "outputs" / "step2_dataset_b_family_occurrences.csv"

OUT = ROOT / "outputs" / "step2_visual_validation_package"
IMG = OUT / "screenshots"
MANIFEST = OUT / "screenshot_manifest.csv"
EVENTS = OUT / "event_timelines.csv"
SUMMARY = OUT / "segment_summary.csv"

TARGETS = [
    ("PF004","ses_20260701-190250-NEELA9BAF",26),
    ("PF004","ses_20260701-175747-SIDDHIGUPTAB00B",7),
    ("PF004","ses_20260701-191537-LAPTOP-76QMG9DE",19),
    ("PF005","ses_20260701-191537-LAPTOP-76QMG9DE",32),
    ("PF005","ses_20260701-173246-SIDDHIGUPTAB00B",16),
    ("PF005","ses_20260701-181413-LAPTOP-76QMG9DE",30),
    ("PF008","ses_20260701-175258-LAPTOP-76QMG9DE",37),
    ("PF008","ses_20260701-183232-LAPTOP-76QMG9DE",5),
    ("PF008","ses_20260701-171614-CHAITANYA0BCF",12),
    ("PF009","ses_20260701-183232-LAPTOP-76QMG9DE",14),
    ("PF009","ses_20260701-191537-LAPTOP-76QMG9DE",13),
    ("PF009","ses_20260701-190250-NEELA9BAF",31),
]

def etype(e):
    v=e.get("event_type"); return v if isinstance(v,str) else ""

def app(e):
    c=e.get("context") or {}; a=c.get("active_app") or {}; v=a.get("app_name")
    return v.strip() if isinstance(v,str) and v.strip() else "UNKNOWN"

def url(e):
    p=e.get("payload") or {}; c=e.get("context") or {}
    for v in (p.get("url"),p.get("current_url"),p.get("target_url"),
              c.get("url"),c.get("current_url"),c.get("browser_url")):
        if isinstance(v,str) and v.strip(): return v.strip()
    return ""

def shots(e):
    r=(e.get("payload") or {}).get("file_reference")
    if not r: return []
    if isinstance(r,dict):
        v=r.get("filename"); return [v] if isinstance(v,str) and v.strip() else []
    if isinstance(r,list):
        return [x.get("filename") for x in r
                if isinstance(x,dict) and isinstance(x.get("filename"),str) and x.get("filename").strip()]
    return []

def prep(events):
    out=[]
    for e in sort_session_events(events):
        t=get_timestamp_ms(e)
        if t is None: continue
        out.append({"timestamp_ms":int(t),
                    "timestamp":pd.to_datetime(int(t),unit="ms",utc=True),
                    "event_type":etype(e),"app":app(e),"url":url(e),
                    "screenshots":shots(e)})
    return out

def main():
    if not SEGMENTS.exists(): raise FileNotFoundError(SEGMENTS)
    seg=pd.read_csv(SEGMENTS,keep_default_na=False)
    seg["segment_index"]=pd.to_numeric(seg["segment_index"],errors="raise").astype(int)
    seg["start"]=pd.to_datetime(seg["start"],utc=True)
    seg["end"]=pd.to_datetime(seg["end"],utc=True)

    raw=load_all_events(DATA_ROOT)
    dsb={sid:ev for sid,ev in raw.items()
         if any(e.get("_dataset")=="B" for e in ev)}

    idx=defaultdict(list)
    for p in DATA_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}:
            idx[p.name].append(p)

    if OUT.exists(): shutil.rmtree(OUT)
    IMG.mkdir(parents=True)

    manifests=[]; event_rows=[]; summaries=[]

    for n,(family,sid,sidx) in enumerate(TARGETS,1):
        m=seg[(seg.session_id==sid)&(seg.segment_index==sidx)]
        if m.empty: raise RuntimeError(f"Missing target: {family} {sid} {sidx}")
        row=m.iloc[0]; start=row.start; end=row.end
        events=prep(dsb[sid])
        inside=[e for e in events if start <= e["timestamp"] <= end]

        # Complete event timeline for the segment
        for i,e in enumerate(inside,1):
            event_rows.append({
                "family_id":family,"target_number":n,"session_id":sid,
                "segment_index":sidx,"event_number":i,
                "timestamp":e["timestamp"],
                "seconds_from_segment_start":(e["timestamp"]-start).total_seconds(),
                "event_type":e["event_type"],"app":e["app"],"url":e["url"],
                "screenshot_refs":len(e["screenshots"])
            })

        # ALL screenshots referenced by events inside the segment
        by_name={}
        for e in inside:
            for fn in e["screenshots"]:
                by_name.setdefault(fn,e)
        ordered=sorted(by_name, key=lambda fn: by_name[fn]["timestamp"])

        folder=IMG / family / sid / f"segment_{sidx:03d}"
        folder.mkdir(parents=True,exist_ok=True)
        copied=0; missing=0

        for order,fn in enumerate(ordered,1):
            e=by_name[fn]; srcs=idx.get(fn,[])
            copies=[]
            if not srcs: missing += 1
            for j,src in enumerate(srcs,1):
                name=f"{order:03d}" + (f"_{j:02d}" if len(srcs)>1 else "") + "_" + fn
                dst=folder/name
                try:
                    shutil.copy2(src,dst); copies.append(str(dst)); copied += 1
                except OSError as ex:
                    print("WARNING:",ex)
            manifests.append({
                "family_id":family,"target_number":n,"session_id":sid,
                "segment_index":sidx,"screenshot_order":order,
                "timestamp":e["timestamp"],
                "seconds_from_segment_start":(e["timestamp"]-start).total_seconds(),
                "filename":fn,"source_event_type":e["event_type"],"source_app":e["app"],
                "source_url":e["url"],"source_paths":" | ".join(map(str,srcs)),
                "copied_paths":" | ".join(copies),"ambiguous_filename":len(srcs)>1,
                "missing_file":not bool(srcs)
            })

        summaries.append({
            "family_id":family,"target_number":n,"session_id":sid,"segment_index":sidx,
            "start":start,"end":end,"duration_seconds":(end-start).total_seconds(),
            "event_count":len(inside),"screenshot_count":len(ordered),
            "copied_screenshot_files":copied,"missing_screenshots":missing,
            "app_sequence":" > ".join(dict.fromkeys(e["app"] for e in inside if e["app"])),
            "browser_urls":" | ".join(dict.fromkeys(e["url"] for e in inside if e["url"]))
        })
        print(f"[{n}/12] {family} | {sid} | segment {sidx}: {len(ordered)} screenshots, {len(inside)} events")

    pd.DataFrame(manifests).to_csv(MANIFEST,index=False,encoding="utf-8-sig")
    pd.DataFrame(event_rows).to_csv(EVENTS,index=False,encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(SUMMARY,index=False,encoding="utf-8-sig")

    print("\n" + "="*70)
    print("FULL VISUAL VALIDATION PACKAGE COMPLETE")
    print("="*70)
    print("Target segments:",len(TARGETS))
    print("Unique screenshot references:",len(set(x["filename"] for x in manifests)))
    print("Manifest rows:",len(manifests))
    print("Event rows:",len(event_rows))
    print("Package:",OUT)
    print("Manifest:",MANIFEST)
    print("Event timelines:",EVENTS)
    print("Summary:",SUMMARY)

if __name__=="__main__":
    main()
