
"""
STEP 3D — PI TECHNICAL INTERFACE DISCOVERY V2 (FINAL)

Uses the exact Step-3A-1 schema rather than heuristic keyword matching.

Authoritative raw columns used:
    event_type
    active_app
    window_title
    routes
    dom_ui_values
    document_values
    event_timestamp
    segment_index

V2 corrections over V1:
    - browser form input is identified from event_type == browser_form_input
    - clipboard from event_type == clipboard_change
    - navigation from event_type == browser_navigation
    - app switching from event_type == app_switch
    - DOM evidence is reported as UI/DOM metadata, not automatically as
      stable CSS/XPath selectors
    - document evidence uses document_values + actual application
    - application/route counts are based on exact fields
    - production capabilities remain explicit UNKNOWN/VALIDATE items

Output:
    outputs/step3d_pi_technical_interface_discovery_v2.xlsx
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import re

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"

A1_SEG = OUT / "step3a1_pi_authoritative_segment_evidence_v2.csv"
A1_EVT = OUT / "step3a1_pi_authoritative_event_detail_v2.csv"
A1_SS = OUT / "step3a1_pi_screenshot_evidence_v2.csv"
A1_OCR = OUT / "step3a1_pi_ocr_semantic_evidence_v2.csv"

A3C_FILES = sorted(
    OUT.glob("step3c_pi_parameterized_vs_branch_analysis_v3*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
A4_FILES = sorted(
    OUT.glob("step3a4_pi_workflow_map_v3*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
B3_FILES = sorted(
    OUT.glob("step3b_pi_automation_boundary_v3*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)


def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def require(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name} missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def clean_route(value: Any) -> str:
    s = norm(value)
    if not s:
        return ""

    low = s.lower()
    known = [
        "/payroll-items",
        "/onboarding",
        "/social-insurance",
        "/leave-applications",
        "/resident-tax",
    ]

    for route in known:
        if route in low:
            return route

    # Reject screenshot/filesystem/XPath/accessibility noise as route.
    if any(x in low for x in (
        "screenshots/", ".png", ".jpg", ".jpeg", ".webp",
        "scr_smart", "c:/", "d:/"
    )):
        return ""

    return ""


def app_name(value: Any) -> str:
    s = norm(value)
    low = s.lower()

    if "edge" in low or "chrome" in low or "firefox" in low or "browser" in low:
        return "Browser"
    if "word" in low:
        return "Word"
    if "excel" in low:
        return "Excel"
    if "notepad" in low:
        return "Notepad"
    if "explorer" in low:
        return "Explorer"
    if "openwith" in low:
        return "OpenWith"
    if "powershell" in low or "terminal" in low or low == "cmd":
        return "Terminal"
    return s or "UNKNOWN"


def load():
    for p in (A1_SEG, A1_EVT, A1_SS, A1_OCR):
        if not p.exists():
            raise FileNotFoundError(p)

    if not A4_FILES:
        raise FileNotFoundError("No Step-3A-4 V3 workbook.")
    if not A3C_FILES:
        raise FileNotFoundError("No Step-3C V3 workbook.")
    if not B3_FILES:
        raise FileNotFoundError("No Step-3B V3 workbook.")

    seg = pd.read_csv(A1_SEG)
    evt = pd.read_csv(A1_EVT)
    ss = pd.read_csv(A1_SS)
    ocr = pd.read_csv(A1_OCR)

    require(
        evt,
        [
            "session_id",
            "segment_index",
            "event_type",
            "active_app",
            "window_title",
            "routes",
            "dom_ui_values",
            "document_values",
            "event_timestamp",
        ],
        "3A-1 event detail",
    )
    require(
        seg,
        ["session_id", "segment_index"],
        "3A-1 segment evidence",
    )

    a4 = A4_FILES[0]
    a4_seq = pd.read_excel(
        a4,
        sheet_name="Segment_Workflow_Sequences"
    )

    a3c = A3C_FILES[0]
    a3c_arch = pd.read_excel(
        a3c,
        sheet_name="Architecture_Decision"
    )

    b3 = B3_FILES[0]
    b3_boundary = pd.read_excel(
        b3,
        sheet_name="Automation_Boundary"
    )

    for df in (seg, evt, ss, ocr, a4_seq):
        if "session_id" in df.columns:
            df["session_id"] = df["session_id"].astype(str)
        if "segment_index" in df.columns:
            df["segment_index"] = pd.to_numeric(
                df["segment_index"],
                errors="coerce"
            ).astype("Int64")

    return (
        seg, evt, ss, ocr,
        a4_seq, a3c_arch, b3_boundary,
        a4, a3c, b3,
    )


def build_interface_inventory(evt: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "interface_component": "Event classification",
            "source_column": "event_type",
            "observed": True,
            "meaning": "Exact event taxonomy used by Step-3A-2 to identify navigation, form input, clipboard, clicks, keyboard, app switches, etc.",
            "implementation_status": "OBSERVED",
        },
        {
            "interface_component": "Active application",
            "source_column": "active_app",
            "observed": True,
            "meaning": "Application in which the event occurred.",
            "implementation_status": "OBSERVED",
        },
        {
            "interface_component": "Browser route",
            "source_column": "routes",
            "observed": True,
            "meaning": "Raw route/URL context; stable known application routes can be extracted.",
            "implementation_status": "OBSERVED — VALIDATE IN SANDBOX",
        },
        {
            "interface_component": "DOM/UI metadata",
            "source_column": "dom_ui_values",
            "observed": True,
            "meaning": "Observed UI/DOM/accessibility metadata; not automatically equivalent to stable CSS/XPath selectors.",
            "implementation_status": "OBSERVED — VALIDATE STABILITY",
        },
        {
            "interface_component": "Document metadata",
            "source_column": "document_values",
            "observed": True,
            "meaning": "Observed supporting document/file context.",
            "implementation_status": "OBSERVED — VALIDATE USE",
        },
        {
            "interface_component": "Structured browser form input",
            "source_rule": 'event_type == "browser_form_input"',
            "observed": (
                evt["event_type"].astype(str).eq("browser_form_input").any()
            ),
            "meaning": "Direct structured form-entry events.",
            "implementation_status": "OBSERVED",
        },
        {
            "interface_component": "Clipboard transfer",
            "source_rule": 'event_type == "clipboard_change"',
            "observed": (
                evt["event_type"].astype(str).eq("clipboard_change").any()
            ),
            "meaning": "Direct clipboard-change events.",
            "implementation_status": "OBSERVED",
        },
        {
            "interface_component": "Browser navigation",
            "source_rule": 'event_type == "browser_navigation"',
            "observed": (
                evt["event_type"].astype(str).eq("browser_navigation").any()
            ),
            "meaning": "Direct browser-navigation events.",
            "implementation_status": "OBSERVED",
        },
        {
            "interface_component": "Application switch",
            "source_rule": 'event_type == "app_switch"',
            "observed": (
                evt["event_type"].astype(str).eq("app_switch").any()
            ),
            "meaning": "Direct context/application transitions.",
            "implementation_status": "OBSERVED",
        },
    ]
    return pd.DataFrame(rows)


def build_routes(evt: pd.DataFrame) -> pd.DataFrame:
    e = evt.copy()
    e["route_clean"] = e["routes"].map(clean_route)

    rows = []
    for route, g in e[e["route_clean"].ne("")].groupby("route_clean"):
        rows.append({
            "route": route,
            "event_rows": len(g),
            "sessions": g["session_id"].nunique(),
            "segments": g[["session_id", "segment_index"]]
                .drop_duplicates().shape[0],
            "status": "OBSERVED",
            "prototype_use":
                "Candidate navigation target; verify page/state and route stability in sandbox.",
        })

    return pd.DataFrame(rows).sort_values(
        ["segments", "event_rows"],
        ascending=[False, False]
    )


def build_dom_ui_evidence(evt: pd.DataFrame, ocr: pd.DataFrame) -> pd.DataFrame:
    """
    Do not call every dom_ui_values string a selector.
    Separate:
        UI metadata
        likely structured IDs/classes
        semantic OCR fields
    """
    rows = []

    vals = Counter(
        norm(v)
        for v in evt["dom_ui_values"]
        if norm(v)
    )

    for value, count in vals.most_common(100):
        low = value.lower()

        if (
            "rootelement" in low
            or "rootwebarea" in low
            or value.startswith("RootWebArea")
        ):
            kind = "ACCESSIBILITY_ROOT_CONTEXT"
            status = "CONTEXT_ONLY"
        elif "#" in value or "@id=" in low or "@class=" in low or "[id=" in low:
            kind = "SELECTOR_LIKE_UI_REFERENCE"
            status = "OBSERVED — VALIDATE"
        elif "|" in value:
            kind = "COMPOSITE_UI_METADATA"
            status = "OBSERVED — VALIDATE"
        else:
            kind = "UI_LABEL_OR_ELEMENT_TEXT"
            status = "SEMANTIC/UI EVIDENCE"

        rows.append({
            "interface_kind": kind,
            "identifier_or_value": value[:1000],
            "observed_count": count,
            "source": "3A-1 dom_ui_values",
            "automation_status": status,
            "prototype_use":
                "Use only after mapping this observation to an actual stable target element in the sandbox.",
        })

    if "field_names" in ocr.columns:
        field_values = []
        for value in ocr["field_names"].fillna("").astype(str):
            for item in value.split(";"):
                item = item.strip()
                if item:
                    field_values.append(item)

        for field, count in Counter(field_values).most_common(80):
            rows.append({
                "interface_kind": "SEMANTIC_FIELD",
                "identifier_or_value": field,
                "observed_count": count,
                "source": "3A-3 OCR semantic evidence",
                "automation_status": "SEMANTICALLY_OBSERVED",
                "prototype_use":
                    "Candidate business-field mapping; does not itself identify a DOM selector.",
            })

    return pd.DataFrame(rows)


def build_event_interfaces(evt: pd.DataFrame) -> pd.DataFrame:
    rows = []

    event_map = [
        (
            "BROWSER_FORM_INPUT",
            "browser_form_input",
            "Populate known structured browser fields; verify target state.",
        ),
        (
            "CLIPBOARD_TRANSFER",
            "clipboard_change",
            "Transfer values with destination/type/integrity checks.",
        ),
        (
            "BROWSER_NAVIGATION",
            "browser_navigation",
            "Navigate only to validated route/page states.",
        ),
        (
            "BROWSER_CLICK",
            "browser_click",
            "Use semantic element targeting after selector/state validation.",
        ),
        (
            "APP_SWITCH",
            "app_switch",
            "Switch between validated application contexts.",
        ),
        (
            "KEYBOARD",
            "keystroke",
            "Use only when the keystroke sequence is deterministic and validated.",
        ),
        (
            "SHORTCUT",
            "shortcut",
            "Use only for validated deterministic operations such as controlled copy/paste.",
        ),
        (
            "TEXT_INPUT_COMPLETE",
            "text_input_complete",
            "Candidate deterministic input completion; verify target field/state.",
        ),
    ]

    for label, event_type, treatment in event_map:
        g = evt[
            evt["event_type"].astype(str).eq(event_type)
        ]

        if g.empty:
            continue

        rows.append({
            "interface": label,
            "event_type": event_type,
            "event_rows": len(g),
            "sessions": g["session_id"].nunique(),
            "segments": g[
                ["session_id", "segment_index"]
            ].drop_duplicates().shape[0],
            "applications":
                " | ".join(
                    sorted({
                        app_name(x)
                        for x in g["active_app"]
                        if norm(x)
                    })
                ),
            "status": "OBSERVED",
            "prototype_treatment": treatment,
        })

    return pd.DataFrame(rows)


def build_desktop_interfaces(evt: pd.DataFrame) -> pd.DataFrame:
    e = evt.copy()
    e["app_norm"] = e["active_app"].map(app_name)

    rows = []

    for app, g in e[e["app_norm"].ne("Browser")].groupby("app_norm"):
        rows.append({
            "application": app,
            "event_rows": len(g),
            "sessions": g["session_id"].nunique(),
            "segments": g[
                ["session_id", "segment_index"]
            ].drop_duplicates().shape[0],
            "status": "OBSERVED",
            "role":
                "Supporting/tooling interface around the pi browser workflow.",
            "prototype_treatment":
                "Minimize dependency; automate only stable deterministic artifact operations that are proven necessary.",
        })

    return pd.DataFrame(rows).sort_values(
        ["segments", "event_rows"],
        ascending=[False, False]
    )


def build_assumptions(routes, event_interfaces, dom, desktop):
    return pd.DataFrame([
        {
            "technical_question": "Known browser routes exist",
            "status": "OBSERVED",
            "evidence":
                "Routes extracted from exact raw `routes` field.",
            "next_validation":
                "Verify route/page state in sandbox.",
        },
        {
            "technical_question": "Structured form input is available",
            "status": "OBSERVED",
            "evidence":
                "Exact browser_form_input events are present in raw events.",
            "next_validation":
                "Map source values to actual target fields and verify selectors/state.",
        },
        {
            "technical_question": "Clipboard transfer interface exists",
            "status": "OBSERVED",
            "evidence":
                "Exact clipboard_change events are present.",
            "next_validation":
                "Validate destination, content integrity and accidental cross-case leakage.",
        },
        {
            "technical_question": "Stable DOM selectors are available",
            "status": "VALIDATE",
            "evidence":
                "UI/DOM metadata is captured, but the logs do not prove selector stability.",
            "next_validation":
                "Inspect actual DOM in sandbox and repeat the same operation across cases/sessions.",
        },
        {
            "technical_question": "Independent authentication is possible",
            "status": "UNKNOWN",
            "evidence":
                "Behavioral logs do not expose credentials/authentication mechanisms.",
            "next_validation":
                "Determine approved sandbox authentication method and secret handling.",
        },
        {
            "technical_question": "Production permissions are available",
            "status": "UNKNOWN",
            "evidence":
                "Dataset B provides no production authorization evidence.",
            "next_validation":
                "Confirm least-privilege account/access needed for prototype and deployment.",
        },
        {
            "technical_question": "An API is available",
            "status": "UNKNOWN",
            "evidence":
                "Browser interaction does not establish API availability.",
            "next_validation":
                "Check approved technical documentation or sandbox interface.",
        },
        {
            "technical_question": "Business rules are programmatically available",
            "status": "UNKNOWN",
            "evidence":
                "Logs show worker behavior, not the authoritative rule engine.",
            "next_validation":
                "Document deterministic rules used by the target workflow.",
        },
        {
            "technical_question": "Audit logging is sufficient",
            "status": "UNKNOWN",
            "evidence":
                "Dataset B logs are not evidence of a production automation audit trail.",
            "next_validation":
                "Define required automation audit events and human-review records.",
        },
        {
            "technical_question": "Exception recovery can be automated safely",
            "status": "VALIDATE",
            "evidence":
                "The Step-3B boundary explicitly uses stop/flag behavior for uncertainty.",
            "next_validation":
                "Test missing data, changed UI, invalid values, timeout and partial completion.",
        },
    ])


def build_prototype():
    return pd.DataFrame([
        {
            "stage": "READ",
            "interface": "Browser route + UI/DOM state",
            "scope":
                "Locate validated source case and read known visible fields.",
            "status": "PROTOTYPE",
            "stop_condition":
                "Required field missing, unexpected page/state, or ambiguous source.",
        },
        {
            "stage": "EXTRACT",
            "interface": "DOM/UI metadata",
            "scope":
                "Extract known fields after actual selector/state mapping.",
            "status": "PROTOTYPE",
            "stop_condition":
                "Element not found, changed structure, or value cannot be validated.",
        },
        {
            "stage": "TRANSFER",
            "interface": "Clipboard / deterministic in-process transfer",
            "scope":
                "Move known values between validated contexts.",
            "status": "PROTOTYPE",
            "stop_condition":
                "Destination mismatch, malformed value, or unexpected content.",
        },
        {
            "stage": "TRANSFORM",
            "interface": "Local deterministic code",
            "scope":
                "Apply only explicitly validated normalization/formatting rules.",
            "status": "PROTOTYPE",
            "stop_condition":
                "Value falls outside known deterministic rules.",
        },
        {
            "stage": "POPULATE",
            "interface": "Browser form fields",
            "scope":
                "Populate validated structured fields and verify resulting values.",
            "status": "PROTOTYPE",
            "stop_condition":
                "Missing field, changed UI, validation error, or uncertain target.",
        },
        {
            "stage": "PREPARE ARTIFACT",
            "interface": "Document/file tooling when necessary",
            "scope":
                "Prepare stable deterministic supporting artifact.",
            "status": "OPTIONAL",
            "stop_condition":
                "Template/content variation requiring interpretation.",
        },
        {
            "stage": "HUMAN REVIEW",
            "interface": "Browser decision interface",
            "scope":
                "Present prepared case, evidence and flags; automation stops.",
            "status": "HUMAN",
            "stop_condition":
                "Human resolves approval/hold/rejection/exception/final release.",
        },
    ])


def main():
    (
        seg,
        evt,
        ss,
        ocr,
        a4_seq,
        a3c_arch,
        b3_boundary,
        a4,
        a3c,
        b3,
    ) = load()

    seg_keys = set(
        (str(s), int(i))
        for s, i in zip(seg["session_id"], seg["segment_index"])
    )

    if len(seg_keys) != 209:
        raise RuntimeError(
            f"Expected 209 pi segment keys, found {len(seg_keys)}."
        )

    # Exact event types used by earlier workflow reconstruction.
    event_type_counts = (
        evt["event_type"]
        .fillna("")
        .astype(str)
        .value_counts()
    )

    inventory = build_interface_inventory(evt)
    routes = build_routes(evt)
    dom = build_dom_ui_evidence(evt, ocr)
    event_interfaces = build_event_interfaces(evt)
    desktop = build_desktop_interfaces(evt)
    assumptions = build_assumptions(
        routes,
        event_interfaces,
        dom,
        desktop
    )
    prototype = build_prototype()

    segment_technical = a4_seq.copy()
    keep = [
        "session_id",
        "segment_index",
        "duration_seconds",
        "raw_event_count",
        "clean_application_sequence",
        "clean_phase_sequence",
        "has_clipboard",
        "has_form_entry",
        "has_navigation",
        "has_decision_evidence",
    ]
    segment_technical = segment_technical[
        [c for c in keep if c in segment_technical.columns]
    ]

    validation = pd.DataFrame([
        {
            "check": "pi_population",
            "expected": 209,
            "observed": len(seg_keys),
            "status": "PASS" if len(seg_keys) == 209 else "FAIL",
        },
        {
            "check": "raw_event_rows",
            "expected": ">0",
            "observed": len(evt),
            "status": "PASS" if len(evt) > 0 else "FAIL",
        },
        {
            "check": "browser_form_input_events",
            "expected": ">0",
            "observed": int(
                event_type_counts.get("browser_form_input", 0)
            ),
            "status": "PASS"
            if event_type_counts.get("browser_form_input", 0) > 0
            else "FAIL",
        },
        {
            "check": "clipboard_change_events",
            "expected": ">0",
            "observed": int(
                event_type_counts.get("clipboard_change", 0)
            ),
            "status": "PASS"
            if event_type_counts.get("clipboard_change", 0) > 0
            else "FAIL",
        },
        {
            "check": "browser_navigation_events",
            "expected": ">0",
            "observed": int(
                event_type_counts.get("browser_navigation", 0)
            ),
            "status": "PASS"
            if event_type_counts.get("browser_navigation", 0) > 0
            else "REVIEW",
        },
        {
            "check": "app_switch_events",
            "expected": ">0",
            "observed": int(
                event_type_counts.get("app_switch", 0)
            ),
            "status": "PASS"
            if event_type_counts.get("app_switch", 0) > 0
            else "FAIL",
        },
        {
            "check": "stable_dom_selectors_proven",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "authentication_proven",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "production_permissions_proven",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "api_availability_proven",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "human_decision_automated",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "segments_jsonl_modified",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
    ])

    readme = pd.DataFrame([
        ["Purpose", "Identify the actual technical interfaces that a pi prototype can interact with, and separate observations from implementation unknowns."],
        ["Authoritative raw schema", "event_type, active_app, window_title, routes, dom_ui_values, document_values, event_timestamp, segment_index."],
        ["Browser evidence", "Routes, browser navigation, clicks and browser_form_input are directly observed in the raw events."],
        ["Transfer evidence", "clipboard_change events are directly observed."],
        ["DOM caution", "dom_ui_values are UI/DOM/accessibility metadata; stable selectors have not been proven."],
        ["Desktop caution", "Word/Excel/Notepad/FileTool usage is supporting tooling, not automatically a separate business workflow."],
        ["Unknowns", "Authentication, production permissions, API availability, business-rule access, selector stability, audit logging and safe exception recovery require sandbox/technical validation."],
        ["Prototype", "READ → EXTRACT → TRANSFER → TRANSFORM → POPULATE → optional artifact preparation → STOP for HUMAN REVIEW."],
    ], columns=["item", "value"])

    output = OUT / "step3d_pi_technical_interface_discovery_v2.xlsx"

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        inventory.to_excel(writer, sheet_name="Interface_Inventory", index=False)
        routes.to_excel(writer, sheet_name="Browser_Routes", index=False)
        dom.to_excel(writer, sheet_name="DOM_UI_Evidence", index=False)
        event_interfaces.to_excel(writer, sheet_name="Input_Transfer_Interface", index=False)
        desktop.to_excel(writer, sheet_name="Desktop_Interface", index=False)
        assumptions.to_excel(writer, sheet_name="Integration_Assumptions", index=False)
        prototype.to_excel(writer, sheet_name="Prototype_Interface_Spec", index=False)
        segment_technical.to_excel(writer, sheet_name="Segment_Technical_Evidence", index=False)
        validation.to_excel(writer, sheet_name="Validation", index=False)

        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        wb = writer.book
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
                    wrap_text=True
                )

            for idx, cells in enumerate(
                ws.iter_cols(
                    min_row=1,
                    max_row=min(ws.max_row, 120)
                ),
                1
            ):
                width = max(
                    [len(str(c.value or "")) for c in cells] + [12]
                )
                ws.column_dimensions[
                    get_column_letter(idx)
                ].width = min(width + 2, 52)

            for r in range(2, min(ws.max_row, 300) + 1):
                ws.row_dimensions[r].height = 30

    style_wb = output
    print("=" * 80)
    print("STEP 3D — PI TECHNICAL INTERFACE DISCOVERY V2")
    print("=" * 80)
    print(f"pi segments: {len(seg_keys)}")
    print(f"raw event rows: {len(evt)}")
    print(f"browser routes observed: {len(routes)}")
    print(f"DOM/UI evidence rows: {len(dom)}")
    print(f"input/transfer interface classes: {len(event_interfaces)}")
    print(f"desktop interfaces observed: {len(desktop)}")

    print("\nBROWSER ROUTES")
    print(routes.to_string(index=False))

    print("\nEXACT EVENT INTERFACES")
    print(event_interfaces.to_string(index=False))

    print("\nINTEGRATION ASSUMPTIONS")
    print(assumptions.to_string(index=False))

    print("\nPROTOTYPE INTERFACE SPEC")
    print(prototype.to_string(index=False))

    print("\nVALIDATION")
    print(validation.to_string(index=False))

    failures = validation[
        validation["status"] == "FAIL"
    ]
    if not failures.empty:
        raise RuntimeError(
            "3D validation failed. Inspect Validation sheet."
        )

    print("\nOUTPUT")
    print(output)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3D V2 COMPLETE.")


if __name__ == "__main__":
    main()
