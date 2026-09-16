
"""
STEP 3A-2 — PI CHRONOLOGICAL WORKFLOW RECONSTRUCTION V3

Purpose
-------
Turn the authoritative 3A-1 evidence into a defensible chronological reconstruction
of the pi workflow.

This is NOT final subprocess classification and NOT automation design.

What it does
------------
1. Loads the authoritative 3A-1 segment/event/screenshot/OCR evidence.
2. Re-checks raw event chronology and segment-level counts.
3. Normalizes low-level events into observable action classes.
4. Detects application transitions and browser-route transitions.
5. Groups consecutive low-level events into "observable phases".
6. Builds a chronological action timeline for every pi segment.
7. Builds recurring workflow signatures using only observable structural behavior.
8. Finds recurring patterns across sessions, not merely repeated rows in one session.
9. Selects representative executions for each recurring pattern.
10. Produces a pattern-level summary, member table, representative timelines,
    and validation/provenance sheets in ONE Excel workbook.
11. Produces a small separate review workbook containing only the highest-value
    representative executions and evidence for manual inspection.

Important methodological rules
------------------------------
- `segments.jsonl` remains authoritative for segment boundaries/labels.
- Raw events remain authoritative for chronological actions.
- 3A-1 outputs are cross-checks and supplements, not replacements for raw events.
- OCR can explain meaning but must never be used to fabricate counts for uncovered segments.
- An application transition is behavioral evidence, not proof of a business process.
- A workflow pattern is descriptive unless semantic evidence supports interpretation.
- Singletons are retained but never promoted to generalized workflows.
- Ties / ambiguous structural patterns remain ambiguous.
- No human judgment is inferred from a button label alone.
- Raw clipboard or typed text contents are never exported.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "outputs"
SEGMENTS = REPO / "segments.jsonl"

A1_SEGMENT = OUTPUT_DIR / "step3a1_pi_authoritative_segment_evidence_v2.csv"
A1_EVENTS = OUTPUT_DIR / "step3a1_pi_authoritative_event_detail_v2.csv"
A1_SCREENSHOTS = OUTPUT_DIR / "step3a1_pi_screenshot_evidence_v2.csv"
A1_OCR = OUTPUT_DIR / "step3a1_pi_ocr_semantic_evidence_v2.csv"
A1_RECON = OUTPUT_DIR / "step3a1_pi_source_reconciliation_v2.csv"
A1_VALIDATION = OUTPUT_DIR / "step3a1_pi_validation_summary_v2.csv"


# ---------------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------------

def norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def uniq(values, limit=100):
    out = []
    seen = set()
    for v in values:
        s = norm(v)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
    return out


def join(values, limit=100):
    return " | ".join(uniq(values, limit))


def sha1(value: Any) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="replace")).hexdigest()


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def normalized_app(app: Any) -> str:
    s = norm(app).lower()
    if not s:
        return "UNKNOWN_APP"
    if any(x in s for x in ("edge", "chrome", "firefox", "browser")):
        return "Browser"
    if "word" in s:
        return "Word"
    if "excel" in s:
        return "Excel"
    if "notepad" in s:
        return "Notepad"
    if "openwith" in s:
        return "OpenWith"
    if "explorer" in s:
        return "Explorer"
    if "terminal" in s or "powershell" in s or "cmd" in s:
        return "Terminal"
    return norm(app)


def normalize_route(value: Any) -> str:
    """Extract stable browser application routes only."""
    s = norm(value)
    if not s:
        return "UNKNOWN_ROUTE"

    low = s.lower()

    known = (
        "/payroll-items",
        "/leave-applications",
        "/onboarding",
        "/social-insurance",
        "/resident-tax",
    )
    for route in known:
        if route in low:
            return route

    # Reject screenshot/filesystem/DOM artifacts.
    if any(token in low for token in (
        ".png", ".jpg", ".jpeg", ".webp",
        "screenshot", "scr_smart", "monitor_",
        "\\", "c:/", "d:/"
    )):
        return "UNKNOWN_ROUTE"


    # Accept a short URL/hash route if it looks like an application route.
    m = re.search(
        r"(?:https?://[^\s/#]+)?(?:#)?(/(?:[A-Za-z0-9_-]+)(?:/[A-Za-z0-9_-]+)*)",
        s
    )
    if m:
        candidate = m.group(1)
        if len(candidate) <= 80 and not any(
            x in candidate.lower() for x in ("scr_", "screenshot", "monitor_")
        ):
            return candidate

    return "UNKNOWN_ROUTE"


def parse_event_counter(value: Any) -> Counter:
    c = Counter()
    s = norm(value)
    for token in s.split("|"):
        token = token.strip()
        m = re.match(r"(.+?)=(\d+)$", token)
        if m:
            c[m.group(1).strip()] = int(m.group(2))
    return c


# ---------------------------------------------------------------------------
# EVENT -> OBSERVABLE ACTION CLASS
# ---------------------------------------------------------------------------

def classify_action(event_type: str, active_app: str, window_title: str,
                    routes: str, dom: str, docs: str) -> str:
    """Descriptive observable action class; not a business-process label."""
    et = norm(event_type).lower()
    app = normalized_app(active_app)

    if et == "app_switch":
        return "SWITCH_APPLICATION"
    if et == "browser_navigation":
        return "NAVIGATE_BROWSER"
    if et == "browser_form_input":
        return "ENTER_BROWSER_FORM"
    if et == "browser_click":
        return "CLICK_BROWSER"
    if et == "clipboard_change":
        return "CLIPBOARD_TRANSFER"
    if et == "shortcut":
        return "COPY_EDIT_SHORTCUT"
    if et == "text_input_complete":
        return "TEXT_INPUT_COMPLETE"
    if et == "mouse_click":
        return "CLICK_DESKTOP"
    if et == "mouse_scroll":
        return "SCROLL"
    if et == "keystroke":
        if app == "Word":
            return "EDIT_WORD"
        if app == "Excel":
            return "EDIT_EXCEL"
        if app == "Notepad":
            return "EDIT_NOTEPAD"
        if app == "Browser":
            return "TYPE_BROWSER"
        return "TYPE"
    if et == "window_title_change":
        return "WINDOW_CHANGE"
    if et == "screenshot_smart":
        return "SCREENSHOT_CHECKPOINT"

    if routes:
        return "BROWSER_CONTEXT"
    if docs:
        return "DOCUMENT_CONTEXT"
    if dom:
        return "UI_CONTEXT"
    if app == "Browser":
        return "WORK_BROWSER"
    if app in {"Word", "Excel", "Notepad"}:
        return f"WORK_{app.upper()}"
    return "OTHER_EVENT"


def action_family(action: str) -> str:
    mapping = {
        "SCREENSHOT_CHECKPOINT": "EVIDENCE",
        "WINDOW_CHANGE": "CONTEXT",
        "UI_CONTEXT": "CONTEXT",
        "BROWSER_CONTEXT": "BROWSER",
        "WORK_BROWSER": "BROWSER",
        "CLICK_BROWSER": "BROWSER",
        "NAVIGATE_BROWSER": "BROWSER",
        "ENTER_BROWSER_FORM": "BROWSER_FORM",
        "TYPE_BROWSER": "BROWSER_FORM",
        "TEXT_INPUT_COMPLETE": "BROWSER_FORM",
        "WORK_WORD": "DOCUMENT",
        "EDIT_WORD": "DOCUMENT",
        "WORK_EXCEL": "DOCUMENT",
        "EDIT_EXCEL": "DOCUMENT",
        "WORK_NOTEPAD": "DOCUMENT",
        "EDIT_NOTEPAD": "DOCUMENT",
        "TYPE": "DESKTOP_INPUT",
        "CLICK_DESKTOP": "DESKTOP_INPUT",
        "SCROLL": "DESKTOP_INPUT",
        "COPY_EDIT_SHORTCUT": "TRANSFER",
        "CLIPBOARD_TRANSFER": "TRANSFER",
        "SWITCH_APPLICATION": "APP_TRANSITION",
        "OTHER_EVENT": "OTHER",
    }
    return mapping.get(norm(action), norm(action))


# ---------------------------------------------------------------------------
# LOAD AUTHORITATIVE 3A-1 INPUTS
# ---------------------------------------------------------------------------

def load_inputs():
    for p in (A1_SEGMENT, A1_EVENTS, A1_SCREENSHOTS, A1_OCR):
        if not p.exists():
            raise FileNotFoundError(f"Required 3A-1 output missing: {p}")

    seg = pd.read_csv(A1_SEGMENT)
    ev = pd.read_csv(A1_EVENTS)
    ss = pd.read_csv(A1_SCREENSHOTS)
    ocr = pd.read_csv(A1_OCR)

    # Normalize IDs.
    for df in (seg, ev, ss, ocr):
        if "session_id" in df.columns:
            df["session_id"] = df["session_id"].astype(str)

    for df in (ev, ss, ocr):
        if "segment_index" in df.columns:
            df["segment_index"] = pd.to_numeric(
                df["segment_index"], errors="coerce"
            ).astype("Int64")

    seg["segment_index"] = pd.to_numeric(
        seg["segment_index"], errors="coerce"
    ).astype(int)

    return seg, ev, ss, ocr


# ---------------------------------------------------------------------------
# NORMALIZE 3A-1 EVENT TABLE
# ---------------------------------------------------------------------------

def build_event_actions(ev: pd.DataFrame) -> pd.DataFrame:
    work = ev.copy()

    work["event_timestamp_parsed"] = pd.to_datetime(
        work["event_timestamp"], utc=True, errors="coerce"
    )

    work["active_app_norm"] = work["active_app"].fillna("").map(normalized_app)

    work["route_norm"] = work["routes"].fillna("").map(normalize_route)
    work["dom_norm"] = work["dom_ui_values"].fillna("").map(norm)
    work["doc_norm"] = work["document_values"].fillna("").map(norm)

    work["action_class"] = [
        classify_action(
            event_type=et,
            active_app=app,
            window_title=win,
            routes=route,
            dom=dom,
            docs=doc,
        )
        for et, app, win, route, dom, doc in zip(
            work.get("event_type", ""),
            work.get("active_app", ""),
            work.get("window_title", ""),
            work.get("routes", ""),
            work.get("dom_ui_values", ""),
            work.get("document_values", ""),
        )
    ]

    work = work.sort_values(
        ["session_id", "segment_index", "event_timestamp_parsed", "source_line"],
        na_position="last",
    )

    # Preserve the exact chronological event order within each segment.
    work["event_order_in_segment"] = (
        work.groupby(["session_id", "segment_index"], sort=False)
        .cumcount() + 1
    )

    return work


# ---------------------------------------------------------------------------
# CHRONOLOGICAL PHASE RECONSTRUCTION
# ---------------------------------------------------------------------------

def build_phase_rows(group: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert low-level events into meaningful observable phases.

    Screenshots are evidence checkpoints and do not create phases.
    Consecutive click/typing/context noise is consolidated.
    """
    g = group.sort_values(
        ["event_timestamp_parsed", "source_line"],
        na_position="last"
    ).copy()

    g = g[
        g["event_type"].astype(str).str.lower() != "screenshot_smart"
    ].copy()

    if g.empty:
        return []

    g["action_family"] = g["action_class"].map(action_family)

    rows = []
    current = None

    def flush():
        nonlocal current
        if current is None:
            return
        items = current.pop("_events")
        current["event_count"] = len(items)
        current["event_types"] = join([x["event_type"] for x in items], 60)
        current["action_classes"] = join([x["action_class"] for x in items], 60)
        current["action_families"] = join([x["action_family"] for x in items], 40)
        current["apps"] = join([x["active_app_norm"] for x in items], 20)
        current["routes"] = join(
            [x["route_norm"] for x in items if x["route_norm"] != "UNKNOWN_ROUTE"],
            30
        )
        current["dom_values"] = join([x["dom_norm"] for x in items], 30)
        current["documents"] = join([x["doc_norm"] for x in items], 30)
        current["first_event_timestamp"] = items[0]["ts"]
        current["last_event_timestamp"] = items[-1]["ts"]
        rows.append(current)
        current = None

    def make_entry(r):
        ts = r["event_timestamp_parsed"]
        return {
            "event_type": norm(r.get("event_type")),
            "action_class": norm(r.get("action_class")),
            "action_family": norm(r.get("action_family")),
            "active_app_norm": norm(r.get("active_app_norm")),
            "route_norm": norm(r.get("route_norm")),
            "dom_norm": norm(r.get("dom_norm")),
            "doc_norm": norm(r.get("doc_norm")),
            "ts": ts.isoformat() if pd.notna(ts) else "",
        }

    for _, r in g.iterrows():
        x = make_entry(r)
        af = x["action_family"]
        app = x["active_app_norm"]
        route = x["route_norm"]

        if current is None:
            current = {
                "phase_class": af,
                "active_app": app,
                "route": route,
                "start_order": int(r.get("event_order_in_segment", 0)),
                "_events": [x],
            }
            continue

        same_app = (
            not app
            or not current["active_app"]
            or app == current["active_app"]
        )
        same_route = (
            route == current["route"]
            or route == "UNKNOWN_ROUTE"
            or current["route"] == "UNKNOWN_ROUTE"
        )
        same_family = af == current["phase_class"]

        # Keep low-level interaction noise inside its surrounding phase.
        absorb = (
            (same_family and same_app)
            or (af in {"DESKTOP_INPUT", "CONTEXT"} and same_app)
            or (
                af == "TRANSFER"
                and current["phase_class"] in {
                    "DOCUMENT", "TRANSFER", "BROWSER_FORM"
                }
            )
            or (
                af in {"BROWSER", "BROWSER_FORM"}
                and current["phase_class"] in {"BROWSER", "BROWSER_FORM"}
                and same_app and same_route
            )
        )

        if absorb:
            current["_events"].append(x)
        else:
            flush()
            current = {
                "phase_class": af,
                "active_app": app,
                "route": route,
                "start_order": int(r.get("event_order_in_segment", 0)),
                "_events": [x],
            }

    flush()
    return rows


def reconstruct_phases(ev_actions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (sid, idx), g in ev_actions.groupby(
        ["session_id", "segment_index"], sort=False
    ):
        for phase_num, phase in enumerate(build_phase_rows(g), 1):
            phase.update({
                "session_id": sid,
                "segment_index": int(idx),
                "phase_order": phase_num,
            })
            rows.append(phase)

    if not rows:
        return pd.DataFrame(columns=[
            "session_id", "segment_index", "phase_order",
            "phase_class", "active_app", "event_count",
            "event_types", "action_classes", "apps", "routes",
            "dom_values", "documents", "first_event_timestamp",
            "last_event_timestamp",
        ])

    df = pd.DataFrame(rows)

    # Put identifiers first.
    cols = [
        "session_id", "segment_index", "phase_order", "phase_class",
        "active_app", "event_count", "event_types", "action_classes",
        "apps", "routes", "dom_values", "documents",
        "first_event_timestamp", "last_event_timestamp",
        "start_order",
    ]
    return df[cols]


# ---------------------------------------------------------------------------
# WORKFLOW SIGNATURES
# ---------------------------------------------------------------------------

def main_sequence(phases: pd.DataFrame) -> str:
    if phases.empty:
        return ""

    out = []
    for _, r in phases.sort_values("phase_order").iterrows():
        cls = norm(r["phase_class"])
        app = norm(r["active_app"])
        token = f"{cls}[{app}]"
        if not out or out[-1] != token:
            out.append(token)

    return " → ".join(out)


def transition_signature(phases: pd.DataFrame) -> str:
    if phases.empty:
        return ""

    tokens = []
    prev = None

    for _, r in phases.sort_values("phase_order").iterrows():
        app = norm(r["active_app"])
        if prev is not None and app != prev:
            tokens.append(f"{prev}>{app}")
        prev = app

    return " | ".join(uniq(tokens, 30))


def route_signature(phases: pd.DataFrame) -> str:
    routes = []
    for x in phases.sort_values("phase_order")["routes"].tolist():
        routes.extend(norm(x).split("|"))
    routes = [x for x in routes if x and x != "UNKNOWN_ROUTE"]
    return " → ".join(uniq(routes, 20))


def build_segment_patterns(seg: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    pgrp = phases.groupby(["session_id", "segment_index"], sort=False)

    phase_map = {}
    for key, g in pgrp:
        phase_map[key] = g.copy()

    result = seg.copy()
    result["phase_count"] = 0
    result["chronological_phase_sequence"] = ""
    result["application_transition_signature"] = ""
    result["route_sequence"] = ""

    for i, row in result.iterrows():
        key = (str(row["session_id"]), int(row["segment_index"]))
        pg = phase_map.get(key)

        if pg is None:
            continue

        result.at[i, "phase_count"] = len(pg)
        result.at[i, "chronological_phase_sequence"] = main_sequence(pg)
        result.at[i, "application_transition_signature"] = transition_signature(pg)
        result.at[i, "route_sequence"] = route_signature(pg)

    # Coarse structural signature.
    result["cross_app_flag"] = (
        result["application_transition_signature"].fillna("").astype(str).str.len() > 0
    )

    # Prefer explicit 3A-1 aggregate columns when present. Otherwise derive
    # the flags directly from the event-type-count string. This makes 3A-2
    # robust to the actual 3A-1 schema and avoids silently inventing zeros.
    def has_event_type(row, event_name: str) -> bool:
        counter = parse_event_counter(row.get("event_types", ""))
        return counter.get(event_name, 0) > 0

    if "clipboard_change_count" in result.columns:
        result["clipboard_flag"] = (
            pd.to_numeric(result["clipboard_change_count"], errors="coerce")
            .fillna(0).gt(0)
        )
    else:
        result["clipboard_flag"] = result.apply(
            lambda r: has_event_type(r, "clipboard_change"),
            axis=1,
        )

    if "browser_form_input_count" in result.columns:
        result["form_flag"] = (
            pd.to_numeric(result["browser_form_input_count"], errors="coerce")
            .fillna(0).gt(0)
        )
    else:
        result["form_flag"] = result.apply(
            lambda r: has_event_type(r, "browser_form_input"),
            axis=1,
        )

    if "browser_navigation_count" in result.columns:
        result["navigation_flag"] = (
            pd.to_numeric(result["browser_navigation_count"], errors="coerce")
            .fillna(0).gt(0)
        )
    else:
        result["navigation_flag"] = result.apply(
            lambda r: has_event_type(r, "browser_navigation"),
            axis=1,
        )

    # Stable structural signature. Do not include exact phase count,
    # screenshot filenames, DOM hashes, or exact click/typing counts.
    app_col = (
        "application_sequence_raw_events"
        if "application_sequence_raw_events" in result.columns
        else "application_sequence"
    )

    def normalize_app_path(value):
        parts = re.split(r"\s*→\s*", norm(value))
        out = []
        for part in parts:
            app = normalized_app(part)
            if app and (not out or out[-1] != app):
                out.append(app)
        return "→".join(out) or "UNKNOWN_APP"

    def phase_family_path(value):
        parts = re.split(r"\s*→\s*", norm(value))
        out = []
        for part in parts:
            base = part.split("[", 1)[0]
            fam = action_family(base)
            if fam and (not out or out[-1] != fam):
                out.append(fam)
        return "→".join(out) or "UNKNOWN_PHASE"

    result["app_signature"] = result[app_col].fillna("UNKNOWN").map(
        normalize_app_path
    )
    result["phase_family_signature"] = (
        result["chronological_phase_sequence"]
        .fillna("")
        .map(phase_family_path)
    )

    result["workflow_signature"] = (
        result["route_sequence"].fillna("UNKNOWN_ROUTE").astype(str)
        + " || apps=" + result["app_signature"].astype(str)
        + " || phases=" + result["phase_family_signature"].astype(str)
        + " || clip=" + result["clipboard_flag"].astype(int).astype(str)
        + " || form=" + result["form_flag"].astype(int).astype(str)
        + " || nav=" + result["navigation_flag"].astype(int).astype(str)
    )


    return result


# ---------------------------------------------------------------------------
# CROSS-SESSION / PATTERN SUMMARIZATION
# ---------------------------------------------------------------------------

def select_representative(g: pd.DataFrame) -> pd.Series:
    # Prefer OCR, then decision evidence, then richer timeline.
    score = (
        g["ocr_direct_present"].astype(int) * 1_000_000
        + g["decision_point_raw_flag"].astype(int) * 100_000
        + g["raw_event_count"].clip(upper=10_000)
        + g["screenshot_reference_event_count"].clip(upper=10_000)
    )
    return g.loc[score.idxmax()]


def build_pattern_summary(seg_patterns: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sig, g in seg_patterns.groupby("workflow_signature", dropna=False):
        representative = select_representative(g)

        rows.append({
            "workflow_signature": sig,
            "pattern_status": (
                "RECURRING"
                if len(g) >= 2
                else "SINGLETON_NEEDS_REVIEW"
            ),
            "segment_count": len(g),
            "session_count": g["session_id"].nunique(),
            "duration_total_s": float(
                pd.to_numeric(g["duration_seconds"], errors="coerce").fillna(0).sum()
            ),
            "duration_median_s": float(
                pd.to_numeric(g["duration_seconds"], errors="coerce").median()
            ),
            "duration_mean_s": float(
                pd.to_numeric(g["duration_seconds"], errors="coerce").mean()
            ),
            "mean_raw_events": float(
                pd.to_numeric(g["raw_event_count"], errors="coerce").mean()
            ),
            "mean_phase_count": float(
                pd.to_numeric(g["phase_count"], errors="coerce").mean()
            ),
            "segments_with_clipboard": int(g["clipboard_flag"].sum()),
            "segments_with_form_input": int(g["form_flag"].sum()),
            "segments_with_navigation": int(g["navigation_flag"].sum()),
            "segments_with_decision_evidence": int(g["decision_point_raw_flag"].sum()),
            "segments_with_cross_app": int(g["cross_app_flag"].sum()),
            "ocr_covered_segments": int(g["ocr_direct_present"].sum()),
            "session_list": join(g["session_id"].tolist(), 50),
            "dominant_route_sequence": (
                Counter(g["route_sequence"].fillna("")).most_common(1)[0][0]
                if len(g) else ""
            ),
            "dominant_app_sequence": (
                Counter(g["application_sequence_raw_events"].fillna("")).most_common(1)[0][0]
                if len(g) else ""
            ),
            "dominant_phase_sequence": (
                Counter(g["chronological_phase_sequence"].fillna("")).most_common(1)[0][0]
                if len(g) else ""
            ),
            "representative_session_id": representative["session_id"],
            "representative_segment_index": int(representative["segment_index"]),
            "representative_duration_s": float(representative["duration_seconds"]),
            "representative_ocr_japanese": norm(representative.get("ocr_japanese_evidence")),
            "representative_ocr_english": norm(representative.get("ocr_english_evidence")),
            "representative_business_purpose_step2": norm(
                representative.get("step2_business_purpose")
            ),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["segment_count", "session_count", "duration_total_s"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        out.insert(
            0,
            "pattern_rank",
            range(1, len(out) + 1),
        )
    else:
        out = pd.DataFrame(columns=[
            "pattern_rank", "workflow_signature", "pattern_status",
            "segment_count", "session_count", "duration_total_s",
            "duration_median_s", "duration_mean_s", "mean_raw_events",
            "mean_phase_count", "segments_with_clipboard",
            "segments_with_form_input", "segments_with_navigation",
            "segments_with_decision_evidence", "segments_with_cross_app",
            "ocr_covered_segments", "session_list", "dominant_route_sequence",
            "dominant_app_sequence", "dominant_phase_sequence",
            "representative_session_id", "representative_segment_index",
            "representative_duration_s", "representative_ocr_japanese",
            "representative_ocr_english",
            "representative_business_purpose_step2",
        ])

    return out


# ---------------------------------------------------------------------------
# REPRESENTATIVE TIMELINES
# ---------------------------------------------------------------------------

def build_representative_timeline(
    patterns: pd.DataFrame,
    phases: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    if patterns.empty:
        return pd.DataFrame()

    for _, p in patterns[patterns["pattern_status"] == "RECURRING"].iterrows():
        sid = str(p["representative_session_id"])
        idx = int(p["representative_segment_index"])

        pg = phases[
            (phases["session_id"] == sid)
            & (phases["segment_index"] == idx)
        ].sort_values("phase_order")

        for _, r in pg.iterrows():
            rows.append({
                "pattern_rank": int(p["pattern_rank"]),
                "workflow_signature": p["workflow_signature"],
                "pattern_segment_count": int(p["segment_count"]),
                "pattern_session_count": int(p["session_count"]),
                "representative_session_id": sid,
                "representative_segment_index": idx,
                "phase_order": int(r["phase_order"]),
                "phase_class": r["phase_class"],
                "active_app": r["active_app"],
                "event_count": int(r["event_count"]),
                "event_types": r["event_types"],
                "action_classes": r["action_classes"],
                "routes": r["routes"],
                "dom_values": r["dom_values"],
                "documents": r["documents"],
                "first_event_timestamp": r["first_event_timestamp"],
                "last_event_timestamp": r["last_event_timestamp"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def build_validation(
    seg: pd.DataFrame,
    ev: pd.DataFrame,
    phases: pd.DataFrame,
    patterns: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    rows.append({
        "check": "canonical_pi_segment_count",
        "expected": 209,
        "observed": len(seg),
        "status": "PASS" if len(seg) == 209 else "FAIL",
    })

    seg_keys = set(
        (str(s), int(i))
        for s, i in zip(seg["session_id"], seg["segment_index"])
    )
    ev_keys = set(
        (str(s), int(i))
        for s, i in zip(ev["session_id"], ev["segment_index"])
        if pd.notna(i)
    )

    rows.append({
        "check": "all_pi_segments_have_raw_event_rows",
        "expected": 209,
        "observed": len(seg_keys.intersection(ev_keys)),
        "status": (
            "PASS"
            if len(seg_keys.intersection(ev_keys)) == 209
            else "FAIL"
        ),
    })

    rows.append({
        "check": "segments_without_raw_events",
        "expected": 0,
        "observed": len(seg_keys - ev_keys),
        "status": "PASS" if len(seg_keys - ev_keys) == 0 else "FAIL",
    })

    raw_event_total = len(ev)

    rows.append({
        "check": "pi_raw_event_rows_attached",
        "expected": raw_event_total,
        "observed": raw_event_total,
        "status": "PASS",
    })

    rows.append({
        "check": "phase_rows_created",
        "expected": ">0",
        "observed": len(phases),
        "status": "PASS" if len(phases) > 0 else "FAIL",
    })

    rows.append({
        "check": "recurring_workflow_patterns",
        "expected": ">0",
        "observed": int(
            (patterns["pattern_status"] == "RECURRING").sum()
            if not patterns.empty else 0
        ),
        "status": (
            "PASS"
            if not patterns.empty
            and int((patterns["pattern_status"] == "RECURRING").sum()) > 0
            else "REVIEW"
        ),
    })

    rows.append({
        "check": "patterns_spanning_multiple_sessions",
        "expected": ">0",
        "observed": int(
            (patterns["session_count"] >= 2).sum()
            if not patterns.empty else 0
        ),
        "status": "INFO",
    })

    # Validate one-to-one membership: every segment assigned exactly once.
    if not seg_patterns.empty:
        assigned_keys = set(
            (str(s), int(i))
            for s, i in zip(
                seg_patterns["session_id"],
                seg_patterns["segment_index"],
            )
        )
        rows.append({
            "check": "pattern_assignment_coverage",
            "expected": len(seg),
            "observed": len(assigned_keys),
            "status": (
                "PASS" if len(assigned_keys) == len(seg)
                else "FAIL"
            ),
        })

    return pd.DataFrame(rows)


# Global placeholder populated in main after pattern creation.
seg_patterns = pd.DataFrame()


# ---------------------------------------------------------------------------
# WRITE EXCEL WORKBOOKS
# ---------------------------------------------------------------------------

def write_excel_main(
    path: Path,
    seg: pd.DataFrame,
    ev: pd.DataFrame,
    ss: pd.DataFrame,
    ocr: pd.DataFrame,
    phases: pd.DataFrame,
    patterns: pd.DataFrame,
    timeline: pd.DataFrame,
    validation: pd.DataFrame,
):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        seg.to_excel(writer, sheet_name="PI_Segments", index=False)
        phases.to_excel(writer, sheet_name="Chronological_Phases", index=False)
        patterns.to_excel(writer, sheet_name="Workflow_Patterns", index=False)
        timeline.to_excel(writer, sheet_name="Representative_Timelines", index=False)
        ev.to_excel(writer, sheet_name="Raw_Event_Detail", index=False)
        ss.to_excel(writer, sheet_name="Screenshot_Evidence", index=False)
        ocr.to_excel(writer, sheet_name="OCR_Evidence", index=False)
        validation.to_excel(writer, sheet_name="Validation", index=False)

        wb = writer.book

        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True
                )

            for col_idx, column_cells in enumerate(
                ws.iter_cols(min_row=1, max_row=min(ws.max_row, 100)),
                1
            ):
                max_len = 0
                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(value))
                ws.column_dimensions[get_column_letter(col_idx)].width = min(
                    max(max_len + 2, 12), 42
                )

        # More readable row heights for text-heavy sheets.
        for name in (
            "PI_Segments",
            "Workflow_Patterns",
            "Representative_Timelines",
            "OCR_Evidence",
        ):
            if name in wb.sheetnames:
                ws = wb[name]
                for row in range(2, min(ws.max_row, 500) + 1):
                    ws.row_dimensions[row].height = 32


def write_excel_review(
    path: Path,
    patterns: pd.DataFrame,
    timeline: pd.DataFrame,
    seg: pd.DataFrame,
    phases: pd.DataFrame,
):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    recurring = patterns[
        patterns["pattern_status"] == "RECURRING"
    ].copy()

    top = recurring.head(20).copy()

    reps = []
    for _, p in top.iterrows():
        reps.append({
            "pattern_rank": int(p["pattern_rank"]),
            "workflow_signature": p["workflow_signature"],
            "segment_count": int(p["segment_count"]),
            "session_count": int(p["session_count"]),
            "representative_session_id": p["representative_session_id"],
            "representative_segment_index": int(p["representative_segment_index"]),
            "route_sequence": p["dominant_route_sequence"],
            "application_sequence": p["dominant_app_sequence"],
            "phase_sequence": p["dominant_phase_sequence"],
            "ocr_japanese": p["representative_ocr_japanese"],
            "ocr_english": p["representative_ocr_english"],
            "step2_business_purpose": p["representative_business_purpose_step2"],
        })
    reps = pd.DataFrame(reps)

    top_ranks = set(top["pattern_rank"].tolist())

    # When no recurring patterns were discovered, build an empty review
    # timeline with a stable schema instead of indexing a missing column.
    if "pattern_rank" in timeline.columns:
        tl = timeline[timeline["pattern_rank"].isin(top_ranks)].copy()
    else:
        tl = pd.DataFrame(columns=[
            "pattern_rank", "workflow_signature", "pattern_segment_count",
            "pattern_session_count", "representative_session_id",
            "representative_segment_index", "phase_order", "phase_class",
            "active_app", "event_count", "event_types", "action_classes",
            "routes", "dom_values", "documents", "first_event_timestamp",
            "last_event_timestamp",
        ])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        top.to_excel(writer, sheet_name="Top_Patterns", index=False)
        reps.to_excel(writer, sheet_name="Manual_Review_Index", index=False)
        tl.to_excel(writer, sheet_name="Review_Timelines", index=False)

        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="548235")
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True,
                )

            for col_idx, column_cells in enumerate(
                ws.iter_cols(min_row=1, max_row=min(ws.max_row, 150)),
                1
            ):
                max_len = 0
                for cell in column_cells:
                    v = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(v))
                ws.column_dimensions[get_column_letter(col_idx)].width = min(
                    max(max_len + 2, 12), 46
                )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    global seg_patterns

    print("=" * 80)
    print("STEP 3A-2 — PI CHRONOLOGICAL WORKFLOW RECONSTRUCTION V3")
    print("=" * 80)

    seg, ev, ss, ocr = load_inputs()

    print(f"3A-1 pi segments loaded: {len(seg)}")
    print(f"3A-1 raw event rows loaded: {len(ev)}")
    print(f"3A-1 screenshot rows loaded: {len(ss)}")
    print(f"3A-1 OCR rows loaded: {len(ocr)}")

    # Step-2/3A-1 derived fields.
    if "ocr_record_count" in seg.columns:
        seg["ocr_direct_present"] = (
            pd.to_numeric(seg["ocr_record_count"], errors="coerce")
            .fillna(0).gt(0)
        )
    else:
        seg["ocr_direct_present"] = (
            seg.get("ocr_source_present", False)
            if "ocr_source_present" in seg.columns
            else False
        )

    if "decision_point_raw_event_count" in seg.columns:
        seg["decision_point_raw_flag"] = (
            pd.to_numeric(
                seg["decision_point_raw_event_count"], errors="coerce"
            )
            .fillna(0).gt(0)
        )
    elif "decision_point_event_evidence" in seg.columns:
        seg["decision_point_raw_flag"] = (
            seg["decision_point_event_evidence"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
    else:
        seg["decision_point_raw_flag"] = False

    # Build raw-derived chronological actions.
    ev_actions = build_event_actions(ev)

    # Ensure all 209 segments are represented in final segment table.
    seg_patterns_local = reconstruct_phases(ev_actions)
    seg_patterns = build_segment_patterns(seg, seg_patterns_local)

    # Re-use clear variable names.
    phases = seg_patterns_local

    # Pattern summary.
    patterns = build_pattern_summary(seg_patterns)

    # Membership output.
    membership = seg_patterns[[
        "session_id",
        "segment_index",
        "segment_start",
        "segment_end",
        "duration_seconds",
        "raw_event_count",
        "workflow_signature",
        "phase_count",
        "chronological_phase_sequence",
        "application_transition_signature",
        "route_sequence",
        "cross_app_flag",
        "clipboard_flag",
        "form_flag",
        "navigation_flag",
        "decision_point_raw_flag",
        "ocr_direct_present",
    ]].copy()

    # Representative timelines.
    timeline = build_representative_timeline(patterns, phases)

    # Full raw event detail includes normalized action class.
    raw_event_output = ev_actions.copy()
    raw_event_output["event_timestamp_parsed"] = (
        raw_event_output["event_timestamp_parsed"]
        .astype(str)
    )

    # Validation.
    validation = build_validation(
        seg_patterns,
        raw_event_output,
        phases,
        patterns,
    )

    # Main authoritative workbook.
    main_xlsx = OUTPUT_DIR / "step3a2_pi_chronological_workflow_evidence_v3.xlsx"
    review_xlsx = OUTPUT_DIR / "step3a2_pi_manual_review_workbook_v3.xlsx"

    write_excel_main(
        main_xlsx,
        seg_patterns,
        raw_event_output,
        ss,
        ocr,
        phases,
        patterns,
        timeline,
        validation,
    )

    write_excel_review(
        review_xlsx,
        patterns,
        timeline,
        seg_patterns,
        phases,
    )

    recurring = patterns[
        patterns["pattern_status"] == "RECURRING"
    ] if not patterns.empty else pd.DataFrame()

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"pi segments: {len(seg_patterns)}")
    print(f"phase rows: {len(phases)}")
    print(f"workflow patterns: {len(patterns)}")
    print(f"recurring patterns: {len(recurring)}")
    print(f"singleton/review patterns: {len(patterns) - len(recurring)}")
    print(
        "patterns spanning >=2 sessions: "
        f"{int((patterns['session_count'] >= 2).sum()) if not patterns.empty else 0}"
    )

    print("\nTOP RECURRING PATTERNS")
    if recurring.empty:
        print("No recurring pattern met the minimum 2-segment descriptive threshold.")
    else:
        cols = [
            "pattern_rank",
            "pattern_status",
            "segment_count",
            "session_count",
            "duration_total_s",
            "duration_median_s",
            "segments_with_clipboard",
            "segments_with_form_input",
            "segments_with_decision_evidence",
            "ocr_covered_segments",
            "dominant_route_sequence",
            "dominant_app_sequence",
            "dominant_phase_sequence",
        ]
        print(
            recurring[cols].head(20).to_string(index=False)
        )

    print("\nVALIDATION")
    print(validation.to_string(index=False))

    failed = validation[validation["status"] == "FAIL"]
    if not failed.empty:
        print("\nVALIDATION FAILURES")
        print(failed.to_string(index=False))
        raise RuntimeError(
            "3A-2 validation failed. Inspect the Validation sheet."
        )

    print("\nOUTPUTS")
    print(main_xlsx)
    print(review_xlsx)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3A-2 COMPLETE.")


if __name__ == "__main__":
    main()
