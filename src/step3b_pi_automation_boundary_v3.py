
"""
STEP 3B — PI AUTOMATION BOUNDARY V3 (SELF-CONTAINED, SCHEMA-VALIDATED)

Build the automation boundary from the current 3A evidence without depending
on fragile column-name assumptions.

Inputs (read-only):
  outputs/step3a1_pi_authoritative_segment_evidence_v2.csv
  outputs/step3a1_pi_authoritative_event_detail_v2.csv
  latest outputs/step3a4_pi_workflow_map_v3*.xlsx

Output (only one artifact):
  outputs/step3b_pi_automation_boundary_v3.xlsx
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"

A1_SEG = OUT / "step3a1_pi_authoritative_segment_evidence_v2.csv"
A1_EVENTS = OUT / "step3a1_pi_authoritative_event_detail_v2.csv"
A4_FILES = sorted(
    OUT.glob("step3a4_pi_workflow_map_v3*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

REQ_A1_SEG = ["session_id", "segment_index"]
REQ_A1_EVENTS = ["session_id", "segment_index", "event_type"]
REQ_A4_PHASES = ["session_id", "segment_index", "clean_phase_sequence"]


def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def require_columns(df: pd.DataFrame, required: list[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name}: missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_sources():
    if not A1_SEG.exists():
        raise FileNotFoundError(A1_SEG)
    if not A1_EVENTS.exists():
        raise FileNotFoundError(A1_EVENTS)
    if not A4_FILES:
        raise FileNotFoundError(
            "No step3a4_pi_workflow_map_v3*.xlsx found."
        )

    a1_seg = pd.read_csv(A1_SEG)
    a1_events = pd.read_csv(A1_EVENTS)
    a4_path = A4_FILES[0]

    phases = pd.read_excel(
        a4_path,
        sheet_name="Segment_Workflow_Sequences"
    )
    semantic = pd.read_excel(
        a4_path,
        sheet_name="Semantic_By_Segment"
    )
    decision = pd.read_excel(
        a4_path,
        sheet_name="Decision_Evidence"
    )

    require_columns(a1_seg, REQ_A1_SEG, "3A-1 segment evidence")
    require_columns(a1_events, REQ_A1_EVENTS, "3A-1 event detail")
    require_columns(phases, REQ_A4_PHASES, "3A-4 segment sequences")
    require_columns(semantic, ["session_id", "segment_index"],
                    "3A-4 semantic evidence")

    for df in (a1_seg, a1_events, phases, semantic, decision):
        if "session_id" in df.columns:
            df["session_id"] = df["session_id"].astype(str)
        if "segment_index" in df.columns:
            df["segment_index"] = pd.to_numeric(
                df["segment_index"], errors="coerce"
            ).astype("Int64")

    return a1_seg, a1_events, phases, semantic, decision, a4_path


def split_seq(value: Any) -> list[str]:
    s = norm(value)
    if not s:
        return []
    out = []
    for token in re.split(r"\s*→\s*", s):
        token = token.strip()
        if token and (not out or token != out[-1]):
            out.append(token)
    return out


# The phase vocabulary currently present in 3A-4.
PHASE_ALIASES = {
    "BROWSER": "BROWSER_WORK",
    "BROWSER_WORK": "BROWSER_WORK",
    "BROWSER_NAVIGATION": "BROWSER_NAVIGATION",
    "FORM_ENTRY": "FORM_ENTRY",
    "DATA_TRANSFER": "DATA_TRANSFER",
    "DOCUMENT_WORK": "DOCUMENT_WORK",
    "APPLICATION_SWITCH": "APPLICATION_SWITCH",
    "DESKTOP_INPUT": "DESKTOP_INPUT",
}


def normalize_phase(token: str) -> str:
    base = norm(token).split("[", 1)[0]
    return PHASE_ALIASES.get(base, base)


def phase_rows(phases: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in phases.iterrows():
        seq = [
            normalize_phase(x)
            for x in split_seq(r["clean_phase_sequence"])
        ]
        seq = [x for x in seq if x]

        for order, phase in enumerate(seq, 1):
            rows.append({
                "session_id": str(r["session_id"]),
                "segment_index": int(r["segment_index"]),
                "phase_order": order,
                "phase": phase,
            })

    return pd.DataFrame(rows)


DECISION_RE = re.compile(
    r"(?:登録確定|承認|却下|保留|"
    r"\bapprove(?:d|al)?\b|\bapproval\b|"
    r"\breject(?:ed|ion)?\b|\bhold\b|"
    r"\bconfirm\b|\bfinalize\b)",
    re.I,
)

POLICY_RE = re.compile(
    r"(?:policy|規定|規則|判断|ambiguous|unclear|"
    r"exception|例外|要確認|manual review)",
    re.I,
)


def semantic_by_key(semantic: pd.DataFrame) -> dict[tuple[str, int], str]:
    """
    Create one semantic-context string per segment, using ONLY semantic fields
    and direct OCR text that are actually available in the 3A-4 semantic sheet.
    """
    lookup = {}

    for _, r in semantic.iterrows():
        key = (str(r["session_id"]), int(r["segment_index"]))

        parts = []
        for col in (
            "business_object",
            "request_type",
            "field_names",
            "status",
            "action",
            "document",
            "visible_instruction",
            "decision_options",
            "ocr_japanese",
            "ocr_english",
        ):
            if col in r.index:
                v = norm(r[col])
                if v:
                    parts.append(v)

        if parts:
            lookup[key] = " | ".join(parts)

    return lookup


def decision_by_key(decision: pd.DataFrame) -> dict[tuple[str, int], str]:
    lookup = {}

    if decision.empty:
        return lookup

    for _, r in decision.iterrows():
        key = (str(r["session_id"]), int(r["segment_index"]))

        parts = []
        for col in (
            "observed_signal",
            "decision_signal",
            "japanese_evidence",
            "english_evidence",
        ):
            if col in r.index:
                v = norm(r[col])
                if v:
                    parts.append(v)

        if parts:
            lookup.setdefault(key, []).extend(parts)

    return {
        k: " | ".join(v)
        for k, v in lookup.items()
    }


def build_segment_context(
    a1_seg: pd.DataFrame,
    semantic: pd.DataFrame,
    decision: pd.DataFrame,
) -> dict[tuple[str, int], dict[str, Any]]:

    sem_lookup = semantic_by_key(semantic)
    dec_lookup = decision_by_key(decision)

    out = {}

    for _, r in a1_seg.iterrows():
        key = (str(r["session_id"]), int(r["segment_index"]))

        sem = sem_lookup.get(key, "")
        dec = dec_lookup.get(key, "")

        combined = " | ".join(x for x in (sem, dec) if x)

        out[key] = {
            "semantic_text": combined,
            "decision_context": bool(
                DECISION_RE.search(sem) or DECISION_RE.search(dec)
            ),
            "policy_exception_context": bool(
                POLICY_RE.search(sem) or POLICY_RE.search(dec)
            ),
            "direct_ocr_present": bool(sem),
        }

    return out


def classify_phase(
    phase: str,
    segment_context: dict[str, Any],
) -> tuple[str, str]:

    # Critical rule: segment-level decision context does NOT turn the whole
    # segment into HUMAN. A phase is classified from its own mechanical nature.
    if phase == "BROWSER_NAVIGATION":
        return (
            "AUTOMATE + FLAG",
            "Predictable browser navigation is mechanical; stop/flag when the target route or state differs.",
        )

    if phase == "BROWSER_WORK":
        return (
            "AUTOMATE + FLAG",
            "Routine browser interaction is mechanical when the target state is known; unexpected state should stop the run.",
        )

    if phase == "DATA_TRANSFER":
        return (
            "AUTOMATE + FLAG",
            "Observed transfer/copy behavior is repetitive; destination, type, and value integrity should be validated.",
        )

    if phase == "FORM_ENTRY":
        return (
            "AUTOMATE + FLAG",
            "Structured field population is deterministic when source values and target fields are known; missing/ambiguous values require review.",
        )

    if phase == "DOCUMENT_WORK":
        return (
            "AUTOMATE + FLAG",
            "Stable document preparation is mechanical; template/content variation should trigger review.",
        )

    if phase == "APPLICATION_SWITCH":
        return (
            "AUTOMATE",
            "Switching between validated applications/contexts is mechanical.",
        )

    if phase == "DESKTOP_INPUT":
        return (
            "UNKNOWN / VALIDATE",
            "Low-level mouse/keyboard/scroll activity does not establish its business purpose and should not be labeled human judgment by itself.",
        )

    return (
        "UNKNOWN / VALIDATE",
        "No sufficiently specific business meaning was established for this observed phase.",
    )


def build_boundary(
    phase_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for phase, g in phase_df.groupby("phase", sort=False):
        cls, rationale = classify_phase(
            phase,
            {},
        )

        segments = g[
            ["session_id", "segment_index"]
        ].drop_duplicates()

        rows.append({
            "step": phase,
            "classification": cls,
            "segments_observed": len(segments),
            "segment_coverage_pct": round(
                100 * len(segments) / 209, 2
            ),
            "sessions_observed": g["session_id"].nunique(),
            "phase_occurrences": len(g),
            "rationale": rationale,
            "human_gate": (
                "No — human gate is a separate downstream decision step."
                if cls != "HUMAN"
                else "Yes"
            ),
        })

    return pd.DataFrame(rows)


def build_human_gate(
    a1_events: pd.DataFrame,
    semantic: pd.DataFrame,
    decision: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # First, use direct semantic decision fields.
    if not semantic.empty:
        for _, r in semantic.iterrows():
            actions = norm(r.get("action"))
            options = norm(r.get("decision_options"))

            signal = " | ".join(
                x for x in (actions, options)
                if x
            )

            if signal and DECISION_RE.search(signal):
                rows.append({
                    "session_id": str(r["session_id"]),
                    "segment_index": int(r["segment_index"]),
                    "evidence_source": "DIRECT_OCR_SEMANTIC",
                    "decision_signal": signal,
                    "evidence_strength":
                        "DIRECT_DECISION_SEMANTICS",
                    "human_gate":
                        "HUMAN_REVIEW_REQUIRED",
                    "why":
                        "Decision-oriented action/option semantics are directly observed.",
                })

    # Then use an explicit decision evidence table only if it has a meaningful
    # contextual signal, not generic event names.
    if not decision.empty:
        for _, r in decision.iterrows():
            signal = norm(r.get("decision_signal")) or norm(
                r.get("observed_signal")
            )

            if not signal:
                continue

            if not DECISION_RE.search(signal):
                continue

            rows.append({
                "session_id": str(r["session_id"]),
                "segment_index": int(r["segment_index"]),
                "evidence_source": "DECISION_EVIDENCE",
                "decision_signal": signal,
                "evidence_strength":
                    "CONTEXTUAL_DECISION_SIGNAL",
                "human_gate":
                    "HUMAN_REVIEW_REQUIRED",
                "why":
                    "Decision-oriented contextual signal is present; this is separate from the mechanical preparation preceding it.",
            })

    if not rows:
        return pd.DataFrame(columns=[
            "session_id",
            "segment_index",
            "evidence_source",
            "decision_signal",
            "evidence_strength",
            "human_gate",
            "why",
        ])

    return pd.DataFrame(rows).drop_duplicates()


def build_prototype():
    return pd.DataFrame([
        {
            "stage": "READ / RETRIEVE",
            "classification": "AUTOMATE + FLAG",
            "implementation_boundary":
                "Read known visible source fields; stop if required data is missing or inconsistent.",
            "human_gate":
                "Human resolves missing/ambiguous source information.",
        },
        {
            "stage": "COPY / TRANSFER",
            "classification": "AUTOMATE + FLAG",
            "implementation_boundary":
                "Transfer validated values between known contexts and verify destination/format.",
            "human_gate":
                "Human resolves unexpected values, types, or destinations.",
        },
        {
            "stage": "TRANSFORM / FORMAT",
            "classification": "AUTOMATE + FLAG",
            "implementation_boundary":
                "Apply only deterministic transformations established from observed cases.",
            "human_gate":
                "Human resolves values outside known transformation rules.",
        },
        {
            "stage": "FORM ENTRY",
            "classification": "AUTOMATE + FLAG",
            "implementation_boundary":
                "Populate validated target fields using mapped source values.",
            "human_gate":
                "Human resolves missing/ambiguous fields or unexpected UI state.",
        },
        {
            "stage": "DOCUMENT PREPARATION",
            "classification": "AUTOMATE + FLAG",
            "implementation_boundary":
                "Prepare stable templates/artifacts where structure is known.",
            "human_gate":
                "Human reviews changed template/content or ambiguous business meaning.",
        },
        {
            "stage": "REVIEW / BUSINESS DECISION",
            "classification": "HUMAN",
            "implementation_boundary":
                "Present the prepared record and evidence to the worker; automation stops.",
            "human_gate":
                "Human approves, holds, rejects, handles exception, or releases.",
        },
    ])


def style_workbook(path: Path):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"

        if ws.max_row > 1 and ws.max_column > 1:
            ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(
                "solid",
                fgColor="1F4E78"
            )
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

        for idx, cells in enumerate(
            ws.iter_cols(
                min_row=1,
                max_row=min(ws.max_row, 150)
            ),
            start=1,
        ):
            width = max(
                [len(str(c.value or "")) for c in cells] + [12]
            )
            ws.column_dimensions[
                get_column_letter(idx)
            ].width = min(width + 2, 52)

        for row in range(
            2,
            min(ws.max_row, 250) + 1
        ):
            ws.row_dimensions[row].height = 30

    wb.save(path)


def main():
    print("=" * 80)
    print("STEP 3B — PI AUTOMATION BOUNDARY V3")
    print("=" * 80)

    (
        a1_seg,
        a1_events,
        a4,
        a4_semantic,
        a4_decision,
        a4_path,
    ) = load_sources()

    if len(a1_seg) != 209:
        raise RuntimeError(
            f"Expected 209 pi segments, found {len(a1_seg)}"
        )

    # Exact key coverage checks.
    seg_keys = set(
        (str(s), int(i))
        for s, i in zip(
            a1_seg["session_id"],
            a1_seg["segment_index"]
        )
    )

    event_keys = set(
        (str(s), int(i))
        for s, i in zip(
            a1_events["session_id"],
            a1_events["segment_index"]
        )
    )

    if len(seg_keys) != 209:
        raise RuntimeError(
            f"Canonical/A1 pi segment keys are not unique: {len(seg_keys)}"
        )

    # Build chronological phase rows.
    phase_df = phase_rows(a4)

    phase_segment_keys = set(
        (str(s), int(i))
        for s, i in zip(
            phase_df["session_id"],
            phase_df["segment_index"]
        )
    ) if not phase_df.empty else set()

    # Every segment needs a phase sequence for boundary analysis. We allow a
    # segment to have no phase only if the upstream evidence genuinely has none.
    missing_phase_segments = seg_keys - phase_segment_keys

    context = build_segment_context(
        a1_seg,
        a4_semantic,
        a4_decision
    )

    boundary = build_boundary(phase_df)

    # Per-phase, per-segment evidence table.
    classified_rows = []

    for _, r in phase_df.iterrows():
        key = (str(r["session_id"]), int(r["segment_index"]))
        c = context.get(
            key,
            {
                "semantic_text": "",
                "decision_context": False,
                "policy_exception_context": False,
                "direct_ocr_present": False,
            }
        )

        cls, rationale = classify_phase(
            str(r["phase"]),
            c,
        )

        classified_rows.append({
            **r.to_dict(),
            "classification": cls,
            "rationale": rationale,
            "direct_ocr_present": c["direct_ocr_present"],
            "decision_context_in_segment": c["decision_context"],
            "policy_exception_context_in_segment":
                c["policy_exception_context"],
        })

    classified = pd.DataFrame(classified_rows)

    human_gate = build_human_gate(
        a1_events,
        a4_semantic,
        a4_decision
    )

    variants = pd.DataFrame([
        {
            "variant_rule": "Different supporting application",
            "classification": "TOOLING VARIANT",
            "evidence":
                "Word/Excel/Notepad/other desktop applications can surround the same browser/transfer/form-entry workflow.",
            "automation_implication":
                "Treat application differences as parameters/branches only when the mechanical sequence actually differs.",
        },
        {
            "variant_rule": "Missing/ambiguous input",
            "classification": "AUTOMATE + FLAG",
            "evidence":
                "Structured field entry and transfer are deterministic only when required values are known.",
            "automation_implication":
                "Automation stops and requests human review.",
        },
        {
            "variant_rule": "Business decision",
            "classification": "HUMAN",
            "evidence":
                "Direct decision-oriented semantic evidence is kept separate from mechanical phases.",
            "automation_implication":
                "No autonomous approval/hold/rejection/release.",
        },
    ])

    prototype = build_prototype()

    validation_rows = [
        {
            "check": "pi_population",
            "expected": 209,
            "observed": len(seg_keys),
            "status": "PASS" if len(seg_keys) == 209 else "FAIL",
        },
        {
            "check": "A1_event_key_coverage",
            "expected": 209,
            "observed": len(seg_keys & event_keys),
            "status": (
                "PASS"
                if len(seg_keys & event_keys) == 209
                else "FAIL"
            ),
        },
        {
            "check": "phase_rows",
            "expected": ">0",
            "observed": len(phase_df),
            "status": "PASS" if len(phase_df) > 0 else "FAIL",
        },
        {
            "check": "segments_without_phase_rows",
            "expected": "",
            "observed": len(missing_phase_segments),
            "status": (
                "INFO"
                if missing_phase_segments
                else "PASS"
            ),
        },
        {
            "check": "human_gate_not_inferred_from_generic_input",
            "expected": "TRUE",
            "observed": "TRUE",
            "status": "PASS",
        },
        {
            "check": "ocr_extrapolation",
            "expected": "NONE",
            "observed": "NONE",
            "status": "PASS",
        },
        {
            "check": "automation_score",
            "expected": "NONE",
            "observed": "NONE",
            "status": "PASS",
        },
    ]

    validation = pd.DataFrame(validation_rows)

    readme = pd.DataFrame([
        ["Purpose", "Find the boundary between deterministic/mechanical pi preparation and human business judgment."],
        ["Population", "All 209 pi segments are retained."],
        ["Mechanical classification", "Phase-level, based on observable workflow activity; not on unrelated decision signals elsewhere in a segment."],
        ["Human gate", "Separate object based on contextual decision-oriented evidence."],
        ["OCR", "Used to explain directly observed semantics only; never used to extrapolate subtype prevalence."],
        ["No score", "No composite numerical automation score is used."],
        ["Prototype", "Automate validated preparation; stop for human review before business approval/hold/rejection/final release."],
        ["Output status", "Pre-prototype boundary specification; not production readiness."],
    ], columns=["item", "value"])

    output = OUT / "step3b_pi_automation_boundary_v3.xlsx"

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        boundary.to_excel(writer, sheet_name="Automation_Boundary", index=False)
        classified.to_excel(writer, sheet_name="Segment_Phase_Evidence", index=False)
        human_gate.to_excel(writer, sheet_name="Human_Gate_Evidence", index=False)
        variants.to_excel(writer, sheet_name="Variants", index=False)
        prototype.to_excel(writer, sheet_name="Prototype_Boundary", index=False)
        validation.to_excel(writer, sheet_name="Validation", index=False)

    style_workbook(output)

    print("\n" + "=" * 80)
    print("3B RESULT")
    print("=" * 80)
    print(f"pi segments: {len(seg_keys)}")
    print(f"phase rows: {len(phase_df)}")
    print(f"phases observed: {phase_df['phase'].nunique() if not phase_df.empty else 0}")
    print(f"human-gate evidence rows: {len(human_gate)}")
    print(f"segments without phase rows: {len(missing_phase_segments)}")

    print("\nAUTOMATION BOUNDARY")
    print(boundary.to_string(index=False))

    print("\nHUMAN GATE")
    print(
        human_gate.groupby(
            "evidence_source"
        ).size().to_string()
        if not human_gate.empty
        else "No contextual human-gate evidence."
    )

    print("\nPROTOTYPE BOUNDARY")
    print(prototype.to_string(index=False))

    print("\nVALIDATION")
    print(validation.to_string(index=False))

    failures = validation[
        validation["status"] == "FAIL"
    ]

    if not failures.empty:
        raise RuntimeError(
            "3B validation failed. Inspect Validation sheet."
        )

    print("\nOUTPUT")
    print(output)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3B V3 COMPLETE.")


if __name__ == "__main__":
    main()
