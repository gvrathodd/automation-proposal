
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from la_common import ROOT, DATA_ROOT, OUT, discover_event_files, identify_dataset, read_jsonl, file_refs, norm, excel_write

A1_SEG = OUT / "step3a1_la_authoritative_segment_evidence_v1.csv"
A1_EVT = OUT / "step3a1_la_authoritative_event_detail_v1.csv"
OUTPUT = OUT / "step3a3_la_ocr_semantic_evidence_v1.xlsx"

def locate_screenshot(filename: str) -> Path | None:
    name = Path(filename).name
    direct = list(DATA_ROOT.rglob(name))
    if direct:
        return sorted(direct)[0]
    return None

def ocr_image(path: Path):
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return [], "PaddleOCR not installed"
    ocr = PaddleOCR(use_angle_cls=True, lang="japan")
    try:
        result = ocr.predict(str(path))
    except Exception as exc:
        return [], str(exc)
    texts=[]
    scores=[]
    # PaddleOCR versions expose different result structures; handle common variants.
    try:
        for item in result:
            if isinstance(item, dict):
                data = item
                for key in ("rec_texts","rec_scores"):
                    if key not in data:
                        continue
                rt = data.get("rec_texts", [])
                rs = data.get("rec_scores", [])
                texts.extend(str(x) for x in rt)
                scores.extend(float(x) for x in rs[:len(rt)])
            else:
                # Older result object
                txts = getattr(item, "rec_texts", []) or []
                scrs = getattr(item, "rec_scores", []) or []
                texts.extend(str(x) for x in txts)
                scores.extend(float(x) for x in scrs[:len(txts)])
    except Exception:
        pass
    return list(zip(texts, scores)), ""

def main():
    evt = pd.read_csv(A1_EVT)
    rows=[]
    for _, r in evt.iterrows():
        refs = norm(r.get("screenshot_refs"))
        if not refs:
            continue
        for ref in refs.split(" | "):
            p = locate_screenshot(ref)
            if p is None:
                rows.append({"session_id":r["session_id"],"segment_index":int(r["segment_index"]),"filename":Path(ref).name,"resolved":False,"ocr_text":"","mean_confidence":None})
                continue
            pairs, err = ocr_image(p)
            txt = " | ".join(x[0] for x in pairs)
            conf = sum(x[1] for x in pairs)/len(pairs) if pairs else None
            rows.append({
                "session_id":r["session_id"],"segment_index":int(r["segment_index"]),
                "filename":p.name,"resolved":True,"ocr_text":txt,"mean_confidence":conf,
                "ocr_error":err
            })
    df = pd.DataFrame(rows).drop_duplicates(["session_id","segment_index","filename"])
    semantic_rows=[]
    term_map = {
        "勤怠":"attendance", "休暇":"leave", "有給":"paid_leave", "社員id":"employee_id",
        "社員 ID":"employee_id", "氏名":"employee_name", "部署":"department",
        "ステータス":"status", "承認":"approve", "差戻し":"return_for_correction",
        "登録確定":"register", "保留":"hold", "処理コメント":"processing_comment",
        "フレックス":"flexible_schedule"
    }
    if not df.empty:
        for (sid, idx), g in df.groupby(["session_id","segment_index"]):
            text = " | ".join(g["ocr_text"].fillna("").tolist())
            lt = text.lower()
            terms=[]
            for jp, en in term_map.items():
                if jp.lower() in lt:
                    terms.append(en)
            semantic_rows.append({
                "session_id":sid,"segment_index":int(idx),
                "direct_ocr_present": bool(text.strip()),
                "ocr_japanese": text[:20000],
                "business_terms": " | ".join(dict.fromkeys(terms)),
                "request_type": "attendance_leave" if ("attendance" in terms or "leave" in terms or "paid_leave" in terms) else "",
                "decision_options": "approve | return_for_correction | hold" if any(x in terms for x in ("approve","return_for_correction","hold")) else "",
                "field_names": " | ".join(dict.fromkeys(x for x in ("employee_id","employee_name","department","attendance","leave","paid_leave","status","processing_comment") if x in terms)),
                "screenshot_count": int(len(g))
            })
    sem = pd.DataFrame(semantic_rows)
    coverage = pd.DataFrame([
        ["LA canonical segments", len(pd.read_csv(A1_SEG)), ""],
        ["Segments with direct OCR text", int(sem["direct_ocr_present"].sum()) if not sem.empty else 0, ""],
        ["Screenshot records", len(df), ""],
        ["Resolved screenshots", int(df["resolved"].sum()) if not df.empty else 0, ""],
    ], columns=["metric","value","note"])
    excel_write(OUTPUT, {"README":coverage, "OCR_Records":df, "LA_Semantic_Profiles":sem})
    print(OUTPUT)

if __name__ == "__main__":
    main()
