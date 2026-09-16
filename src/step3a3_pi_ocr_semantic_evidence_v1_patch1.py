
"""
STEP 3A-3 — PI OCR SEMANTIC EVIDENCE V1

Purpose
-------
Join the direct translated OCR evidence to the 209 pi segments and 19 recurring
structural workflow patterns discovered in Step 3A-2.

This layer is semantic evidence extraction, NOT subprocess classification.

Inputs
------
Required:
    outputs/step3a1_pi_authoritative_segment_evidence_v2.csv
    outputs/step3a1_pi_ocr_semantic_evidence_v2.csv
    outputs/step3a2_pi_chronological_workflow_evidence_v3.xlsx
    outputs/dataset_b_ocr_raw_20pct.jsonl
    outputs/dataset_b_ocr_translated_20pct.jsonl

Optional:
    outputs/dataset_b_ocr_translation_cache.json
    outputs/dataset_b_ocr_translation_summary.json

Fallback:
    If the Step-3A-2 workbook is unavailable, the script will use the
    latest matching Step-3A-2 workbook found under outputs/.

Outputs
-------
ONE authoritative workbook:
    outputs/step3a3_pi_ocr_semantic_evidence_v1.xlsx

Sheets
------
README
PI_OCR_Joined
PI_Semantic_Profiles
Pattern_Semantic_Summary
OCR_Term_Frequency
Coverage_Check
Translation_Metadata

Method
------
For every OCR-covered pi execution:
    Japanese OCR -> English analytical translation -> structured semantic fields

Structured semantic fields:
    business_object
    request_type
    field_names
    status
    action
    document
    visible_instruction
    decision_option

The extraction is evidence-preserving:
- Japanese source is retained.
- English translation is retained where available.
- A semantic field is only filled when there is textual evidence for it.
- No OCR semantic result is extrapolated to uncovered pi segments.
- The 58 OCR-covered pi segments are not treated as a representative sample
  for prevalence claims.
- Existing Step-3A-2 structural patterns are used for grouping only.
- Step-2 outputs are optional supporting context.

Safety/privacy
--------------
Potentially sensitive values such as employee names/IDs and arbitrary OCR text
are retained only in the detailed evidence sheet when directly supplied by
the OCR file. This script does not attempt to invent or redact source content;
the main semantic profile deliberately emphasizes labels/terms rather than
copying full screenshot OCR dumps.

No segment labels are modified.
segments.jsonl is never modified.
"""

from __future__ import annotations
from collections import Counter, defaultdict

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"

A1_SEG = OUT / "step3a1_pi_authoritative_segment_evidence_v2.csv"
A1_OCR = OUT / "step3a1_pi_ocr_semantic_evidence_v2.csv"
A2_WB = OUT / "step3a2_pi_chronological_workflow_evidence_v3.xlsx"
RAW_OCR = OUT / "dataset_b_ocr_raw_20pct.jsonl"
TRANS_OCR = OUT / "dataset_b_ocr_translated_20pct.jsonl"
CACHE = OUT / "dataset_b_ocr_translation_cache.json"
SUMMARY = OUT / "dataset_b_ocr_translation_summary.json"


def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def uniq(values, limit=80):
    result = []
    seen = set()
    for v in values:
        s = norm(v)
        if not s or s in seen:
            continue
        seen.add(s)
        result.append(s)
        if len(result) >= limit:
            break
    return result


def join(values, limit=80):
    return " | ".join(uniq(values, limit))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj["_line_no"] = line_no
            rows.append(obj)
    return rows


def find_latest_step3a2() -> Path | None:
    candidates = sorted(
        OUT.glob("step3a2_pi_chronological_workflow_evidence*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_a2_patterns() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = A2_WB if A2_WB.exists() else find_latest_step3a2()
    if path is None:
        return pd.DataFrame(), pd.DataFrame()

    patterns = pd.read_excel(path, sheet_name="Workflow_Patterns")
    members = pd.read_excel(path, sheet_name="PI_Segments")
    return patterns, members


# ---------------------------------------------------------------------------
# Semantic term rules
# ---------------------------------------------------------------------------

FIELD_PATTERNS = [
    r"employee id", r"社員\s*id", r"社員番号",
    r"氏名", r"name", r"full name",
    r"区分", r"category", r"type", r"種別",
    r"金額", r"amount", r"金額",
    r"ステータス", r"status",
    r"処理コメント", r"processing comments?",
    r"変更種別", r"change type",
    r"遡及", r"retroactive",
]

STATUS_PATTERNS = [
    (r"未処理|unprocessed|pending processing|処理待ち", "UNPROCESSED_OR_PENDING"),
    (r"処理済|processed|完了|completed", "PROCESSED_OR_COMPLETED"),
    (r"保留|hold", "HOLD"),
    (r"承認|approval|approved|承認済", "APPROVAL"),
    (r"却下|reject|rejected", "REJECTED"),
]

ACTION_PATTERNS = [
    (r"登録確定|confirm|finalize|register", "CONFIRM_OR_FINALIZE"),
    (r"保留|hold", "HOLD"),
    (r"承認|approve|approval", "APPROVE"),
    (r"却下|reject|rejection", "REJECT"),
    (r"変更|change|modify|update", "CHANGE_OR_UPDATE"),
    (r"申請|apply|application|request", "SUBMIT_OR_REQUEST"),
    (r"精算|reimbursement|settle", "REIMBURSEMENT_PROCESSING"),
]

REQUEST_PATTERNS = [
    (r"給与変更|payroll change|payroll", "PAYROLL_CHANGE"),
    (r"経費精算|expense reimbursement|expense", "EXPENSE_REIMBURSEMENT"),
    (r"扶養控除変更|dependent.?tax|dependent deduction", "DEPENDENT_DEDUCTION_CHANGE"),
    (r"役職手当新設|allowance|role allowance", "ALLOWANCE_CHANGE"),
    (r"交通費|transportation expense", "TRANSPORTATION_EXPENSE"),
    (r"勤怠|attendance", "ATTENDANCE"),
    (r"休暇申請|leave application|leave", "LEAVE_APPLICATION"),
]

DOCUMENT_PATTERNS = [
    (r"word|document|文書|書類", "WORD_OR_DOCUMENT"),
    (r"excel|spreadsheet|表計算", "EXCEL_OR_SPREADSHEET"),
    (r"notepad|メモ", "NOTEPAD_OR_MEMO"),
]

INSTRUCTION_PATTERNS = [
    (r"入力|enter|input|記入|fill", "ENTER_OR_FILL"),
    (r"確認|check|review|verify", "REVIEW_OR_VERIFY"),
    (r"選択|select|choose", "SELECT"),
    (r"保存|save", "SAVE"),
    (r"提出|submit", "SUBMIT"),
    (r"登録|register", "REGISTER"),
]

DECISION_PATTERNS = [
    (r"登録確定|confirm|finalize", "CONFIRM_OR_FINALIZE"),
    (r"保留|hold", "HOLD"),
    (r"承認|approve|approval", "APPROVE"),
    (r"却下|reject|rejection", "REJECT"),
]


def canonical_match(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    found = []
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.I):
            found.append(label)
    return uniq(found, 20)


def infer_business_object(text: str) -> str:
    t = text.lower()

    if re.search(r"employee|社員|従業員|給与|payroll|経費|expense", t):
        return "EMPLOYEE_RECORD"
    if re.search(r"申請|application|request", t):
        return "EMPLOYEE_APPLICATION_OR_REQUEST"
    return ""


def infer_semantics(japanese: str, english: str, route: str, step2_purpose: str = "") -> dict[str, str]:
    # Japanese is intentionally retained as primary evidence. English is used
    # as an analytical aid only.
    text = " ".join(x for x in (japanese, english, route, step2_purpose) if x)

    request = canonical_match(text, REQUEST_PATTERNS)
    statuses = canonical_match(text, STATUS_PATTERNS)
    actions = canonical_match(text, ACTION_PATTERNS)
    docs = canonical_match(text, DOCUMENT_PATTERNS)
    instructions = canonical_match(text, INSTRUCTION_PATTERNS)
    decisions = canonical_match(text, DECISION_PATTERNS)

    field_hits = []
    for p in FIELD_PATTERNS:
        if re.search(p, text, flags=re.I):
            field_hits.append(p)

    # Human-readable field names are more useful than regex strings.
    field_labels = []
    if re.search(r"employee\s*id|社員\s*id|社員番号", text, re.I):
        field_labels.append("EMPLOYEE_ID")
    if re.search(r"氏名|employee name|name|full name", text, re.I):
        field_labels.append("EMPLOYEE_NAME")
    if re.search(r"区分|category", text, re.I):
        field_labels.append("CATEGORY")
    if re.search(r"金額|amount", text, re.I):
        field_labels.append("AMOUNT")
    if re.search(r"種別|type", text, re.I):
        field_labels.append("TYPE")
    if re.search(r"ステータス|status", text, re.I):
        field_labels.append("STATUS")
    if re.search(r"処理コメント|processing comments?", text, re.I):
        field_labels.append("PROCESSING_COMMENT")
    if re.search(r"変更種別|change type", text, re.I):
        field_labels.append("CHANGE_TYPE")
    if re.search(r"遡及|retroactive", text, re.I):
        field_labels.append("RETROACTIVE_FLAG")

    return {
        "business_object": infer_business_object(text),
        "request_type": "; ".join(request),
        "field_names": "; ".join(uniq(field_labels, 30)),
        "status": "; ".join(statuses),
        "action": "; ".join(actions),
        "document": "; ".join(docs),
        "visible_instruction": "; ".join(instructions),
        "decision_options": "; ".join(decisions),
    }


# ---------------------------------------------------------------------------
# OCR source parsing
# ---------------------------------------------------------------------------

def make_term_rows(record: dict[str, Any], source_kind: str) -> list[dict[str, Any]]:
    rows = []

    terms = record.get("japanese_ocr_lines") or []
    if terms:
        for i, t in enumerate(terms):
            if not isinstance(t, dict):
                continue
            rows.append({
                "term_order": i + 1,
                "japanese": norm(t.get("japanese")),
                "english": norm(t.get("english")),
                "score": t.get("score", ""),
                "source_kind": source_kind,
            })

    if not rows:
        jp = norm(record.get("ocr_text"))
        en = norm(record.get("english_translation"))
        if jp or en:
            rows.append({
                "term_order": 1,
                "japanese": jp,
                "english": en,
                "score": record.get("ocr_mean_score", ""),
                "source_kind": source_kind,
            })

    return rows


def build_ocr_maps(
    raw_records: list[dict[str, Any]],
    translated_records: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:

    result: dict[tuple[str, int], list[dict[str, Any]]] = {}

    # Translated records are preferred because they have both languages.
    for record in translated_records:
        sid = record.get("session_id")
        idx = record.get("segment_index")
        if sid is None or idx is None:
            continue

        try:
            key = (str(sid), int(idx))
        except Exception:
            continue

        rows = make_term_rows(record, "translated_20pct")
        if not rows:
            continue

        result.setdefault(key, []).extend(rows)

    # Add raw OCR records if a translated record is missing.
    for record in raw_records:
        sid = record.get("session_id")
        idx = record.get("segment_index")
        if sid is None or idx is None:
            continue

        try:
            key = (str(sid), int(idx))
        except Exception:
            continue

        if key in result:
            continue

        rows = make_term_rows(record, "raw_20pct")
        if rows:
            result[key] = rows

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("STEP 3A-3 — PI OCR SEMANTIC EVIDENCE V1")
    print("=" * 80)

    for path in (A1_SEG, A1_OCR, RAW_OCR, TRANS_OCR):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input missing: {path}"
            )

    seg = pd.read_csv(A1_SEG)
    a1_ocr = pd.read_csv(A1_OCR)
    patterns, a2_members = load_a2_patterns()

    raw_records = load_jsonl(RAW_OCR)
    translated_records = load_jsonl(TRANS_OCR)

    print(f"pi segments from 3A-1: {len(seg)}")
    print(f"3A-1 OCR rows: {len(a1_ocr)}")
    print(f"translated OCR records: {len(translated_records)}")
    print(f"raw OCR records: {len(raw_records)}")
    print(f"Step-3A-2 patterns loaded: {len(patterns)}")

    # Restrict to canonical pi keys.
    pi_keys = set(
        (str(s), int(i))
        for s, i in zip(seg["session_id"], seg["segment_index"])
    )

    ocr_map = build_ocr_maps(raw_records, translated_records)

    # Map A2 structural pattern to each segment.
    pattern_map = {}
    if not a2_members.empty and "workflow_signature" in a2_members.columns:
        for _, r in a2_members.iterrows():
            try:
                pattern_map[(str(r["session_id"]), int(r["segment_index"]))] = {
                    "workflow_signature": norm(r.get("workflow_signature")),
                    "pattern_segment_count": r.get("pattern_segment_count", ""),
                    "pattern_status": norm(r.get("pattern_status")),
                }
            except Exception:
                pass

    # Step-2 business-purpose context is optional.
    step2_purpose_map = {}
    for _, r in seg.iterrows():
        key = (str(r["session_id"]), int(r["segment_index"]))
        step2_purpose_map[key] = norm(
            r.get("step2_business_purpose")
            or r.get("step2_business_process")
        )

    joined_rows = []
    semantic_rows = []
    term_rows = []

    for _, s in seg.iterrows():
        key = (str(s["session_id"]), int(s["segment_index"]))
        recs = ocr_map.get(key, [])

        p = pattern_map.get(key, {})
        route = norm(s.get("route_evidence_raw")) or norm(
            s.get("browser_route_or_url_evidence")
        )

        jp = join([r["japanese"] for r in recs], 100)
        en = join([r["english"] for r in recs], 100)
        semantic = infer_semantics(
            jp,
            en,
            route,
            step2_purpose_map.get(key, ""),
        )

        base = {
            "session_id": key[0],
            "segment_index": key[1],
            "duration_seconds": s.get("duration_seconds"),
            "workflow_signature": p.get("workflow_signature", ""),
            "pattern_segment_count": p.get("pattern_segment_count", ""),
            "pattern_status": p.get("pattern_status", ""),
            "route_evidence": route,
            "application_sequence": norm(
                s.get("application_sequence_raw_events")
                or s.get("application_sequence")
            ),
            "raw_event_count": s.get("raw_event_count", ""),
            "screenshot_reference_event_count": s.get(
                "screenshot_reference_event_count", ""
            ),
            "step2_business_purpose": step2_purpose_map.get(key, ""),
            "ocr_direct": bool(recs),
            "ocr_record_count": len(recs),
            "ocr_japanese": jp,
            "ocr_english": en,
            "ocr_mean_score": (
                pd.to_numeric(
                    [r["score"] for r in recs],
                    errors="coerce"
                ).mean()
                if recs else ""
            ),
            **semantic,
        }

        joined_rows.append(base)

        if recs:
            semantic_rows.append({
                k: base[k]
                for k in (
                    "session_id",
                    "segment_index",
                    "workflow_signature",
                    "pattern_segment_count",
                    "pattern_status",
                    "route_evidence",
                    "application_sequence",
                    "step2_business_purpose",
                    "ocr_record_count",
                    "ocr_mean_score",
                    "business_object",
                    "request_type",
                    "field_names",
                    "status",
                    "action",
                    "document",
                    "visible_instruction",
                    "decision_options",
                )
            })

            for r in recs:
                term_rows.append({
                    "session_id": key[0],
                    "segment_index": key[1],
                    "workflow_signature": p.get("workflow_signature", ""),
                    "source_kind": r["source_kind"],
                    "term_order": r["term_order"],
                    "japanese": r["japanese"],
                    "english": r["english"],
                    "score": r["score"],
                })

    joined = pd.DataFrame(joined_rows)
    semantic = pd.DataFrame(semantic_rows)
    terms = pd.DataFrame(term_rows)

    # Pattern semantic summary: semantic evidence observed within each pattern.
    if not patterns.empty and "workflow_signature" in joined.columns:
        obs = joined[joined["ocr_direct"]].copy()

        def aggregate_pattern(g: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "ocr_covered_segments": len(g),
                "business_objects_observed": join(g["business_object"]),
                "request_types_observed": join(g["request_type"]),
                "field_names_observed": join(g["field_names"]),
                "statuses_observed": join(g["status"]),
                "actions_observed": join(g["action"]),
                "documents_observed": join(g["document"]),
                "instructions_observed": join(g["visible_instruction"]),
                "decision_options_observed": join(g["decision_options"]),
                "japanese_evidence": join(g["ocr_japanese"], 160),
                "english_evidence": join(g["ocr_english"], 160),
            })

        if not obs.empty:
            pattern_sem = (
                obs.groupby("workflow_signature", dropna=False)
                .apply(aggregate_pattern, include_groups=False)
                .reset_index()
            )
        else:
            pattern_sem = pd.DataFrame(
                columns=[
                    "workflow_signature",
                    "ocr_covered_segments",
                    "business_objects_observed",
                    "request_types_observed",
                    "field_names_observed",
                    "statuses_observed",
                    "actions_observed",
                    "documents_observed",
                    "instructions_observed",
                    "decision_options_observed",
                    "japanese_evidence",
                    "english_evidence",
                ]
            )

        # Step-3A-2 may itself contain an `ocr_covered_segments` column.
        # Rename the semantic aggregation count before merging so pandas does
        # not create opaque _x/_y columns.
        if "ocr_covered_segments" in patterns.columns:
            pattern_sem = pattern_sem.rename(
                columns={"ocr_covered_segments": "ocr_covered_segments_semantic"}
            )

        pattern_sem = patterns.merge(
            pattern_sem,
            on="workflow_signature",
            how="left",
        )
    else:
        pattern_sem = pd.DataFrame()

    # OCR term frequencies, based only on directly OCR-covered pi segments.
    freq_counter = Counter()
    jp_to_en = defaultdict(Counter)

    for _, r in terms.iterrows():
        jp = norm(r.get("japanese"))
        en = norm(r.get("english"))
        if not jp:
            continue
        freq_counter[jp] += 1
        if en:
            jp_to_en[jp][en] += 1

    freq_rows = []
    for jp, count in freq_counter.most_common():
        english = jp_to_en[jp].most_common(1)[0][0] if jp_to_en[jp] else ""
        freq_rows.append({
            "japanese_term_or_text": jp,
            "direct_ocr_occurrences": count,
            "most_common_english_analytical_translation": english,
            "translation_variants_observed": join(
                jp_to_en[jp].keys(), 20
            ),
        })
    freq = pd.DataFrame(freq_rows)

    # Coverage and provenance checks.
    coverage = pd.DataFrame([
        {"metric": "canonical_pi_segments", "value": len(seg)},
        {"metric": "pi_segments_with_direct_ocr", "value": int(joined["ocr_direct"].sum())},
        {"metric": "direct_ocr_coverage_pct_of_pi", "value": round(100 * joined["ocr_direct"].mean(), 2)},
        {"metric": "pi_segments_without_direct_ocr", "value": int((~joined["ocr_direct"]).sum())},
        {"metric": "semantic_records", "value": len(semantic)},
        {"metric": "ocr_term_rows", "value": len(terms)},
        {"metric": "unique_japanese_terms_or_texts", "value": len(freq)},
        {"metric": "patterns_from_3a2", "value": len(patterns)},
        {"metric": "patterns_with_direct_ocr_evidence", "value": (
            pattern_sem["ocr_covered_segments_semantic"].notna().sum()
            if not pattern_sem.empty and "ocr_covered_segments_semantic" in pattern_sem.columns
            else 0
        )},
        {"metric": "interpretation_rule", "value":
            "Direct OCR explains observed semantics; no prevalence extrapolation to uncovered pi segments."},
    ])

    translation_meta = []

    for path, label in [
        (RAW_OCR, "raw_ocr_20pct"),
        (TRANS_OCR, "translated_ocr_20pct"),
        (CACHE, "translation_cache"),
        (SUMMARY, "translation_summary"),
    ]:
        if path.exists():
            if path.suffix.lower() == ".json":
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            translation_meta.append({
                                "source": label,
                                "metadata_key": k,
                                "metadata_value": json.dumps(v, ensure_ascii=False),
                            })
                    else:
                        translation_meta.append({
                            "source": label,
                            "metadata_key": "record_type",
                            "metadata_value": type(obj).__name__,
                        })
                except Exception as exc:
                    translation_meta.append({
                        "source": label,
                        "metadata_key": "read_error",
                        "metadata_value": str(exc),
                    })
            else:
                translation_meta.append({
                    "source": label,
                    "metadata_key": "file",
                    "metadata_value": path.name,
                })

    translation_meta_df = pd.DataFrame(translation_meta)

    # -----------------------------------------------------------------------
    # Excel output — one authoritative workbook.
    # -----------------------------------------------------------------------
    workbook = OUT / "step3a3_pi_ocr_semantic_evidence_v1.xlsx"

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        readme = pd.DataFrame([
            ["Purpose", "Direct OCR semantic interpretation for pi executions."],
            ["Scope", "Only directly OCR-covered pi executions are used for semantic understanding; no prevalence extrapolation."],
            ["Primary sources", "segments.jsonl + raw Dataset-B events + translated/raw OCR + Step-3A-2 structural patterns."],
            ["Semantic fields", "business object, request type, field names, status, action, document, visible instruction, decision options."],
            ["Interpretation status", "Observed text is evidence; semantic labels are analytical interpretations and remain tied to the observed OCR."],
            ["Important limitation", "OCR coverage is incomplete and uneven; uncovered pi segments remain semantically unresolved unless independently evidenced."],
            ["Canonical segmentation", "segments.jsonl is not modified."],
        ], columns=["item", "value"])
        readme.to_excel(writer, sheet_name="README", index=False)

        joined.to_excel(writer, sheet_name="PI_OCR_Joined", index=False)
        semantic.to_excel(writer, sheet_name="PI_Semantic_Profiles", index=False)
        pattern_sem.to_excel(writer, sheet_name="Pattern_Semantic_Summary", index=False)
        freq.to_excel(writer, sheet_name="OCR_Term_Frequency", index=False)
        coverage.to_excel(writer, sheet_name="Coverage_Check", index=False)
        translation_meta_df.to_excel(writer, sheet_name="Translation_Metadata", index=False)

        # Formatting
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            if ws.max_row > 1 and ws.max_column > 1:
                ws.auto_filter.ref = ws.dimensions

            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F4E78")
                c.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True,
                )

            for idx, col_cells in enumerate(
                ws.iter_cols(min_row=1, max_row=min(ws.max_row, 120)),
                start=1
            ):
                max_len = 12
                for cell in col_cells:
                    max_len = max(max_len, len(str(cell.value or "")))
                ws.column_dimensions[get_column_letter(idx)].width = min(
                    max_len + 2, 45
                )

    print("\n" + "=" * 80)
    print("3A-3 RESULT")
    print("=" * 80)
    print(f"Canonical pi segments: {len(seg)}")
    direct_from_a1 = (
        int(seg["ocr_record_count"].fillna(0).astype(float).gt(0).sum())
        if "ocr_record_count" in seg.columns else None
    )
    direct_from_join = int(joined["ocr_direct"].sum())

    print(
        f"Direct OCR-covered pi segments (3A-1): "
        f"{direct_from_a1 if direct_from_a1 is not None else 'unknown'}"
    )
    print(
        f"Direct OCR-covered pi segments (joined OCR files): "
        f"{direct_from_join} "
        f"({100 * joined['ocr_direct'].mean():.2f}%)"
    )
    if direct_from_a1 is not None and direct_from_a1 != direct_from_join:
        print(
            "WARNING: 3A-1 OCR coverage and independent OCR-file coverage differ. "
            "This is reported explicitly and is NOT silently reconciled."
        )
    print(f"Semantic OCR profiles: {len(semantic)}")
    print(f"OCR term rows: {len(terms)}")
    print(f"Recurring structural patterns loaded: {len(patterns)}")
    semantic_pattern_count = (
        int(pattern_sem["ocr_covered_segments_semantic"].notna().sum())
        if not pattern_sem.empty and "ocr_covered_segments_semantic" in pattern_sem.columns
        else 0
    )
    print(
        f"Patterns with direct OCR semantic evidence: "
        f"{semantic_pattern_count}"
    )

    print("\nIMPORTANT")
    print(
        "The semantic fields are evidence-supported interpretations of "
        "direct OCR-covered executions. They are NOT extrapolated to the "
        "uncovered pi population."
    )

    print("\nOUTPUT")
    print(workbook)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3A-3 COMPLETE.")


if __name__ == "__main__":
    main()
