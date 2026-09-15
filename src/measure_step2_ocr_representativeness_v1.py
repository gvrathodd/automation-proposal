
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
SEGMENTS_FILE = PROJECT_ROOT / "segments_v2.jsonl"
OCR_FILE = OUT / "dataset_b_ocr_translated_20pct.jsonl"

FAMILY_OUT = OUT / "step2_layer6_ocr_representativeness_by_family.csv"
SESSION_OUT = OUT / "step2_layer6_ocr_representativeness_by_session.csv"
SEGMENT_OUT = OUT / "step2_layer6_ocr_coverage_by_segment.csv"
SUMMARY_OUT = OUT / "step2_layer6_ocr_representativeness_summary.csv"

def load_segments():
    rows=[]
    with SEGMENTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj=json.loads(line)
                rows.append({
                    "session_id": str(obj["session_id"]),
                    "start": pd.to_datetime(obj["start"], utc=True),
                    "end": pd.to_datetime(obj["end"], utc=True),
                    "label": str(obj["label"]),
                })
    df=pd.DataFrame(rows)
    if len(df)!=540:
        raise RuntimeError(f"Expected 540 segments; found {len(df)}")
    df["segment_index"]=df.groupby("session_id").cumcount()+1
    df["segment_key"]=df["session_id"].astype(str)+"::"+df["segment_index"].astype(str)
    return df

def load_ocr():
    rows=[]
    with OCR_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj=json.loads(line)
            sid=str(obj.get("session_id",""))
            idx=obj.get("segment_index")
            if not sid or idx is None: continue
            rows.append({
                "segment_key": f"{sid}::{int(idx)}",
                "session_id": sid,
                "segment_index": int(idx),
                "ocr_mean_score": float(obj["ocr_mean_score"]) if obj.get("ocr_mean_score") not in (None,"") else None,
                "translation_status": str(obj.get("translation_status","")),
            })
    df=pd.DataFrame(rows)
    if len(df)!=1113:
        raise RuntimeError(f"Expected 1,113 translated OCR records; found {len(df)}")
    return df

def main():
    print("="*78)
    print("STEP 2 — OCR REPRESENTATIVENESS CHECK")
    print("="*78)
    if not SEGMENTS_FILE.exists(): raise FileNotFoundError(SEGMENTS_FILE)
    if not OCR_FILE.exists(): raise FileNotFoundError(OCR_FILE)

    segments=load_segments()
    ocr=load_ocr()
    print(f"All segments: {len(segments):,}")
    print(f"OCR screenshot records: {len(ocr):,}")

    ocr_by_segment=(ocr.groupby("segment_key").agg(
        ocr_screenshot_count=("segment_key","size"),
        ocr_mean_confidence=("ocr_mean_score","mean"),
        complete_translation_records=("translation_status",lambda s:int((s=="complete").sum())),
    ).reset_index())

    covered=segments.merge(ocr_by_segment,on="segment_key",how="left",validate="one_to_one")
    covered["ocr_screenshot_count"]=covered["ocr_screenshot_count"].fillna(0).astype(int)
    covered["ocr_mean_confidence"]=covered["ocr_mean_confidence"].fillna(0)
    covered["ocr_present"]=(covered["ocr_screenshot_count"]>0).astype(int)

    family=(covered.groupby("label").agg(
        total_segments=("segment_key","size"),
        segments_with_ocr=("ocr_present","sum"),
        total_ocr_screenshots=("ocr_screenshot_count","sum"),
        mean_ocr_screenshots_per_segment=("ocr_screenshot_count","mean"),
        mean_ocr_screenshots_per_covered_segment=("ocr_screenshot_count",lambda s:s[s>0].mean() if (s>0).any() else 0),
        median_ocr_screenshots_per_covered_segment=("ocr_screenshot_count",lambda s:s[s>0].median() if (s>0).any() else 0),
        max_ocr_screenshots_per_segment=("ocr_screenshot_count","max"),
        mean_ocr_confidence=("ocr_mean_confidence",lambda s:s[s>0].mean() if (s>0).any() else 0),
    ).reset_index().rename(columns={"label":"family"}))
    family["ocr_coverage_pct"]=family["segments_with_ocr"]/family["total_segments"]*100

    session_family=(covered.groupby(["label","session_id"]).agg(
        total_segments=("segment_key","size"),
        segments_with_ocr=("ocr_present","sum"),
        total_ocr_screenshots=("ocr_screenshot_count","sum"),
        mean_ocr_screenshots_per_covered_segment=("ocr_screenshot_count",lambda s:s[s>0].mean() if (s>0).any() else 0),
    ).reset_index().rename(columns={"label":"family"}))
    session_family["ocr_coverage_pct"]=session_family["segments_with_ocr"]/session_family["total_segments"]*100

    session=(covered.groupby("session_id").agg(
        total_segments=("segment_key","size"),
        segments_with_ocr=("ocr_present","sum"),
        total_ocr_screenshots=("ocr_screenshot_count","sum"),
        mean_ocr_screenshots_per_covered_segment=("ocr_screenshot_count",lambda s:s[s>0].mean() if (s>0).any() else 0),
        median_ocr_screenshots_per_covered_segment=("ocr_screenshot_count",lambda s:s[s>0].median() if (s>0).any() else 0),
    ).reset_index())
    session["ocr_coverage_pct"]=session["segments_with_ocr"]/session["total_segments"]*100

    segment_output=covered[[
        "segment_key","session_id","segment_index","label","start","end",
        "ocr_present","ocr_screenshot_count","ocr_mean_confidence"
    ]].copy()

    total_segments=len(segments)
    covered_segments=int(covered["ocr_present"].sum())
    uncovered=total_segments-covered_segments
    summary=pd.DataFrame([
        ["all_segments",total_segments],
        ["ocr_screenshot_records",len(ocr)],
        ["segments_with_ocr",covered_segments],
        ["segments_without_ocr",uncovered],
        ["overall_segment_coverage_pct",covered_segments/total_segments*100],
        ["family_min_coverage_pct",float(family["ocr_coverage_pct"].min())],
        ["family_max_coverage_pct",float(family["ocr_coverage_pct"].max())],
        ["family_coverage_range_pct_points",float(family["ocr_coverage_pct"].max()-family["ocr_coverage_pct"].min())],
        ["sessions",session["session_id"].nunique()],
    ],columns=["metric","value"])

    family.to_csv(FAMILY_OUT,index=False,encoding="utf-8-sig")
    session_family.to_csv(SESSION_OUT,index=False,encoding="utf-8-sig")
    segment_output.to_csv(SEGMENT_OUT,index=False,encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUT,index=False,encoding="utf-8-sig")

    print("\n"+ "="*78)
    print("FAMILY-LEVEL OCR COVERAGE")
    print("="*78)
    print(family.to_string(index=False))

    print("\n"+ "="*78)
    print("SESSION-LEVEL OCR COVERAGE")
    print("="*78)
    print(session[["session_id","total_segments","segments_with_ocr","ocr_coverage_pct","total_ocr_screenshots","mean_ocr_screenshots_per_covered_segment"]].sort_values("ocr_coverage_pct").to_string(index=False))

    print("\n"+ "="*78)
    print("OVERALL")
    print("="*78)
    print(summary.to_string(index=False))

    print(f"\nSegments WITHOUT OCR: {uncovered:,}")
    if uncovered:
        print("First 30:")
        print(segment_output[segment_output["ocr_present"]==0][["session_id","segment_index","label"]].head(30).to_string(index=False))

    print("\nOutputs:")
    print(FAMILY_OUT)
    print(SESSION_OUT)
    print(SEGMENT_OUT)
    print(SUMMARY_OUT)

if __name__=="__main__":
    main()
