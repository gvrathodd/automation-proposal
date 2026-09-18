
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from la_common import OUT, excel_write

A1=OUT/"step3a1_la_authoritative_event_detail_v1.csv"
A3=OUT/"step3a3_la_ocr_semantic_evidence_v1.xlsx"
H3=OUT/"step3h_la_real_dataset_replay_v1.xlsx"
OUTPUT=OUT/"step3i_la_real_data_extraction_similarity_v1.xlsx"

def tokens(s):
    return set(re.findall(r"[a-z0-9_]+|[ぁ-んァ-ン一-龥]{2,}",str(s).lower()))

def jac(a,b):
    a=tokens(a); b=tokens(b)
    if not a and not b:return 1.0
    if not a or not b:return 0.0
    return len(a&b)/len(a|b)

def main():
    ev=pd.read_csv(A1)
    sem=pd.read_excel(A3,sheet_name="LA_Semantic_Profiles")
    replay=pd.read_excel(H3,sheet_name="Segment_Replay")
    rows=[]
    for _,s in sem.iterrows():
        sid=s["session_id"]; idx=int(s["segment_index"])
        g=ev[(ev.session_id==sid)&(ev.segment_index==idx)]
        raw_text=" | ".join(g.get("target_field",pd.Series(dtype=str)).fillna("").astype(str).tolist())
        ocr_text=str(s.get("ocr_japanese","") or "")
        rows.append({"session_id":sid,"segment_index":idx,
                     "ocr_supported":bool(s.get("direct_ocr_present",False)),
                     "raw_target_evidence":raw_text,
                     "ocr_semantic_text":ocr_text[:4000],
                     "token_jaccard":round(jac(raw_text,ocr_text),4),
                     "interpretation":"Raw operation logs expose interaction/target metadata; screenshot OCR is needed for business-content details when fields are not present in payloads."})
    df=pd.DataFrame(rows)
    summary=pd.DataFrame([
        ["Population","LA direct-OCR semantic profiles compared with raw event metadata"],
        ["No OCR injection","OCR text is used only after raw extraction for comparison."],
        ["No production claim","This measures evidence overlap, not production automation accuracy."],
    ],columns=["item","value"])
    excel_write(OUTPUT,{"README":summary,"Segment_Comparison":df,"Replay_Coverage":replay})
    print(OUTPUT)
if __name__=="__main__": main()
