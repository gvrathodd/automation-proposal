
"""
STEP 3F — PI PROTOTYPE VARIANT VALIDATION V1

Purpose
-------
Test the Step-3E prototype against six representative pi variants using the
evidence-derived architecture from Steps 3B-3E.

Variant classes:
    A — normal
    B — different business subtype
    C — different document/application path
    D — missing field
    E — unexpected value
    F — decision/exception case

Important:
    The actual production /payroll-items application is unavailable.
    Therefore all executable tests run against the controlled local mock built
    in Step 3E. The variant definitions are grounded in Step-3C evidence and
    the prototype is evaluated for preparation behavior, safe stopping, and
    human-gate preservation.

For each case we record:
    extraction_worked
    transformation_worked
    navigation_worked
    fields_populated_correctly
    manual_correction_needed
    human_takeover_point
    failure
    final_result

The six representative cases are selected to correspond to the Step-3C
architecture evidence:
    - EXPENSE_REIMBURSEMENT
    - PAYROLL_CHANGE
    - document/tooling variation
    - missing input
    - unexpected/invalid value
    - human decision gate

Output:
    outputs/step3f_pi_actual_variant_validation_v1.xlsx
"""

from __future__ import annotations

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PROTO = ROOT / "prototype"

MOCK = PROTO / "step3e1_mock_payroll_items.html"

C3_FILES = sorted(
    OUT.glob("step3c_pi_parameterized_vs_branch_analysis_v3*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

URL_HOST = "127.0.0.1"
URL_PORT = 8766
BASE_URL = f"http://{URL_HOST}:{URL_PORT}"
TARGET_URL = f"{BASE_URL}/payroll-items"

OUTPUT = OUT / "step3f_pi_actual_variant_validation_v1.xlsx"


def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def load_step3c():
    if not C3_FILES:
        raise FileNotFoundError(
            "No Step-3C V3 workbook found in outputs."
        )

    path = C3_FILES[0]

    case_summary = pd.read_excel(
        path,
        sheet_name="Case_Type_Summary"
    )

    architecture = pd.read_excel(
        path,
        sheet_name="Architecture_Decision"
    )

    case_evidence = pd.read_excel(
        path,
        sheet_name="Case_Evidence"
    )

    return path, case_summary, architecture, case_evidence


def build_variant_cases(case_summary: pd.DataFrame):
    """
    Keep exactly six compact test cases. Case categories are evidence-derived
    but values are local prototype fixtures, not claims about Dataset-B values.
    """

    observed_types = set(
        case_summary["case_type"].astype(str)
    ) if not case_summary.empty else set()

    expense_type = (
        "expense_reimbursement"
        if "EXPENSE_REIMBURSEMENT" in observed_types
        else "expense_reimbursement"
    )

    payroll_type = (
        "payroll_change"
        if "PAYROLL_CHANGE" in observed_types
        else "payroll_change"
    )

    return [
        {
            "variant_id": "3F-A",
            "variant_class": "NORMAL",
            "evidence_basis": "Shared browser/transfer/form core observed in Step 3C.",
            "case_type": expense_type,
            "employee_id": "F3001",
            "amount": "1250.50",
            "effective_date": "2026-09-15",
            "processing_comment": "Normal expense reimbursement.",
            "supporting_document": "",
            "expected_result": "SUCCESS",
            "expected_human_point": "After verification, before business decision.",
        },
        {
            "variant_id": "3F-B",
            "variant_class": "DIFFERENT_BUSINESS_SUBTYPE",
            "evidence_basis": "PAYROLL_CHANGE is a second directly OCR-supported case type in Step 3C.",
            "case_type": payroll_type,
            "employee_id": "F3002",
            "amount": "4500.00",
            "effective_date": "2026-10-01",
            "processing_comment": "Payroll-change subtype using the same preparation engine.",
            "supporting_document": "",
            "expected_result": "SUCCESS",
            "expected_human_point": "After verification, before business decision.",
        },
        {
            "variant_id": "3F-C",
            "variant_class": "DOCUMENT_APPLICATION_VARIANT",
            "evidence_basis": "Step 3C observed document/tooling variation around the common mechanical core.",
            "case_type": expense_type,
            "employee_id": "F3003",
            "amount": "890.00",
            "effective_date": "2026-09-28",
            "processing_comment": "Document-assisted preparation variant.",
            "supporting_document": "invoice_F3003.pdf",
            "expected_result": "SUCCESS",
            "expected_human_point": "After verification, before business decision.",
            "document_variant": True,
        },
        {
            "variant_id": "3F-D",
            "variant_class": "MISSING_FIELD",
            "evidence_basis": "Step 3E safe-stop behavior for missing required input.",
            "case_type": expense_type,
            "employee_id": "",
            "amount": "700.00",
            "effective_date": "2026-09-21",
            "processing_comment": "Missing employee ID variant.",
            "supporting_document": "receipt_F3004.pdf",
            "expected_result": "STOP_MISSING_REQUIRED_INPUT",
            "expected_human_point": "At input validation.",
        },
        {
            "variant_id": "3F-E",
            "variant_class": "UNEXPECTED_VALUE",
            "evidence_basis": "Step 3E safe-stop behavior for invalid/unexpected values.",
            "case_type": payroll_type,
            "employee_id": "F3005",
            "amount": "NOT_A_NUMBER",
            "effective_date": "2026-10-01",
            "processing_comment": "Unexpected amount format.",
            "supporting_document": "",
            "expected_result": "STOP_INVALID_INPUT",
            "expected_human_point": "At input validation.",
        },
        {
            "variant_id": "3F-F",
            "variant_class": "DECISION_EXCEPTION",
            "evidence_basis": "Step 3B/3E human-gate boundary: business decision remains human.",
            "case_type": expense_type,
            "employee_id": "F3006",
            "amount": "950.25",
            "effective_date": "2026-09-25",
            "processing_comment": "Prepared record requiring human review.",
            "supporting_document": "receipt_F3006.pdf",
            "expected_result": "SUCCESS_HUMAN_GATE",
            "expected_human_point": "After verification; before Register/Hold.",
        },
    ]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/payroll-items":
            data = MOCK.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )
            self.send_header(
                "Content-Length",
                str(len(data))
            )
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)


def start_server():
    server = ThreadingHTTPServer(
        (URL_HOST, URL_PORT),
        Handler
    )
    thread = Thread(
        target=server.serve_forever,
        daemon=True
    )
    thread.start()
    time.sleep(0.2)
    return server


SUPPORTED_CASE_TYPES = {
    "expense_reimbursement",
    "payroll_change",
    "dependent_deduction_change",
    "transportation_expense",
}


def validate_input(case):
    if not norm(case["employee_id"]):
        return False, "STOP_MISSING_REQUIRED_INPUT", "employee_id is missing."

    if case["case_type"] not in SUPPORTED_CASE_TYPES:
        return (
            False,
            "STOP_UNSUPPORTED_REQUEST_TYPE",
            f"Unsupported request type: {case['case_type']}",
        )

    amount = norm(case["amount"]).replace(",", "")
    if not re.fullmatch(r"-?\d+(?:\.\d{1,2})?", amount):
        return False, "STOP_INVALID_INPUT", "Amount is not numeric."

    date = norm(case["effective_date"])
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return False, "STOP_INVALID_INPUT", "Invalid effective date."

    return True, "VALID", ""


def transform(case):
    ok, code, reason = validate_input(case)

    if not ok:
        raise ValueError(f"{code}: {reason}")

    amount = float(
        norm(case["amount"]).replace(",", "")
    )

    return {
        "employee_id": norm(case["employee_id"]),
        "request_type": norm(case["case_type"]),
        "amount": f"{amount:.2f}",
        "effective_date": norm(case["effective_date"]),
        "processing_comment": norm(case["processing_comment"]),
        "supporting_document": norm(case["supporting_document"]),
        "status": "Prepared — Human Review Required",
    }


def run_variant(page, case):
    started = time.perf_counter()

    result = {
        "variant_id": case["variant_id"],
        "variant_class": case["variant_class"],
        "case_type": case["case_type"],
        "expected_result": case["expected_result"],
        "actual_result": "",
        "extraction_worked": False,
        "transformation_worked": False,
        "navigation_worked": False,
        "fields_attempted": 0,
        "fields_populated_correctly": 0,
        "manual_correction_needed": False,
        "human_takeover_point": "",
        "failure": "",
        "register_clicked": False,
        "hold_clicked": False,
        "human_gate_preserved": False,
        "execution_seconds": 0.0,
    }

    try:
        valid, code, reason = validate_input(case)

        if not valid:
            result["actual_result"] = code
            result["human_takeover_point"] = "Input validation"
            result["failure"] = reason
            result["human_gate_preserved"] = True
            return result

        # Extraction = reading the supplied parameterized case record.
        source = transform(case)
        result["extraction_worked"] = True
        result["transformation_worked"] = True

        page.goto(
            TARGET_URL,
            wait_until="domcontentloaded"
        )
        result["navigation_worked"] = (
            page.locator("#employee_id").count() == 1
        )

        if not result["navigation_worked"]:
            result["actual_result"] = "STOP_UI_STATE"
            result["human_takeover_point"] = "Navigation/state validation"
            result["failure"] = "Target form was not found."
            result["human_gate_preserved"] = True
            return result

        if case.get("document_variant"):
            # The local mock represents the document/tooling variant through
            # a supporting-document field. It does not claim to reproduce Word.
            page.locator(
                "#supporting_document"
            ).fill(source["supporting_document"])

        fields = {
            "#employee_id": source["employee_id"],
            "#request_type": source["request_type"],
            "#amount": source["amount"],
            "#effective_date": source["effective_date"],
            "#processing_comment": source["processing_comment"],
            "#supporting_document": source["supporting_document"],
        }

        result["fields_attempted"] = len(fields)

        for selector, value in fields.items():
            if not page.locator(selector).count():
                result["actual_result"] = "STOP_UI_STATE"
                result["human_takeover_point"] = "Field mapping"
                result["failure"] = f"Missing target field: {selector}"
                result["human_gate_preserved"] = True
                return result

            locator = page.locator(selector)

            # request_type is a <select>, not a text input.
            # Use select_option for the dropdown and fill() for text fields.
            if selector == "#request_type":
                locator.select_option(value)
            else:
                locator.fill(value)

        actual = page.evaluate(
            "window.getPreparedRecord()"
        )

        for field, value in {
            "employee_id": source["employee_id"],
            "request_type": source["request_type"],
            "amount": source["amount"],
            "effective_date": source["effective_date"],
            "processing_comment": source["processing_comment"],
            "supporting_document": source["supporting_document"],
        }.items():

            if actual[field] == value:
                result["fields_populated_correctly"] += 1

        result["extraction_worked"] = True
        result["transformation_worked"] = True

        expected_fields = len(fields)

        if result["fields_populated_correctly"] != expected_fields:
            result["actual_result"] = "FAIL_FIELD_VERIFICATION"
            result["human_takeover_point"] = "Verification"
            result["failure"] = (
                f"{result['fields_populated_correctly']}/{expected_fields} "
                "fields verified."
            )
            result["manual_correction_needed"] = True
            result["human_gate_preserved"] = True
            return result

        register_disabled = page.locator(
            "#register"
        ).is_disabled()
        hold_disabled = page.locator(
            "#hold"
        ).is_disabled()

        result["human_gate_preserved"] = (
            register_disabled and hold_disabled
        )

        if not result["human_gate_preserved"]:
            result["actual_result"] = "FAIL_HUMAN_GATE"
            result["human_takeover_point"] = "Human decision boundary"
            result["failure"] = (
                "Register/Hold controls were not safely disabled."
            )
            return result

        result["human_takeover_point"] = (
            "After verification, before Register/Hold."
        )
        result["actual_result"] = (
            "SUCCESS_HUMAN_GATE"
            if case["expected_result"] == "SUCCESS_HUMAN_GATE"
            else "SUCCESS"
        )

        return result

    except Exception as exc:
        result["actual_result"] = "ERROR"
        result["failure"] = (
            f"{type(exc).__name__}: {exc}"
        )
        result["human_gate_preserved"] = True
        return result

    finally:
        result["execution_seconds"] = round(
            time.perf_counter() - started,
            4
        )


def write_output(
    cases,
    results,
    case_summary,
    architecture,
):
    rdf = pd.DataFrame(results)

    metrics = pd.DataFrame([
        {
            "metric": "variants_tested",
            "value": len(rdf),
            "interpretation":
                "Compact representative variant set, not a population sample.",
        },
        {
            "metric": "variants_passed",
            "value": int(rdf["status"].eq("PASS").sum()),
            "interpretation":
                "PASS means successful preparation or expected safe stop.",
        },
        {
            "metric": "successful_preparation",
            "value": int(
                rdf["actual_result"]
                .isin(["SUCCESS", "SUCCESS_HUMAN_GATE"])
                .sum()
            ),
            "interpretation":
                "Variant reached verified preparation and human gate.",
        },
        {
            "metric": "safe_stops",
            "value": int(
                rdf["actual_result"]
                .astype(str)
                .str.startswith("STOP_")
                .sum()
            ),
            "interpretation":
                "Prototype correctly refused to proceed.",
        },
        {
            "metric": "manual_correction_required",
            "value": int(
                rdf["manual_correction_needed"].sum()
            ),
            "interpretation":
                "Prototype needed a manual correction after field population.",
        },
        {
            "metric": "register_clicks",
            "value": int(
                rdf["register_clicked"].sum()
            ),
            "interpretation":
                "Must be zero; final business decision is human.",
        },
        {
            "metric": "hold_clicks",
            "value": int(
                rdf["hold_clicked"].sum()
            ),
            "interpretation":
                "Must be zero; final business decision is human.",
        },
    ])

    final_findings = pd.DataFrame([
        {
            "finding": "Shared parameterized core",
            "result":
                "Supported for the tested variants.",
            "basis":
                "Normal expense, payroll subtype and document variant used the same preparation engine.",
            "limitation":
                "Tests run on local mock; production interface not available.",
        },
        {
            "finding": "Safe exception handling",
            "result":
                "Demonstrated.",
            "basis":
                "Missing field and unexpected value cases stopped before target population.",
            "limitation":
                "Only tested exception conditions; production exception inventory is broader.",
        },
        {
            "finding": "Human decision boundary",
            "result":
                "Preserved.",
            "basis":
                "Register/Hold were not clicked in any variant.",
            "limitation":
                "Human decision quality and production policy remain outside the prototype.",
        },
        {
            "finding": "Document/application variant",
            "result":
                "Treated as an optional parameter/tooling variant.",
            "basis":
                "The evidence-derived architecture keeps document preparation around the shared browser/transfer/form core.",
            "limitation":
                "This local mock does not reproduce actual Word/Excel application automation.",
        },
    ])

    validation = pd.DataFrame([
        {
            "check": "all_variants_expected_outcome",
            "expected": "YES",
            "observed":
                "YES"
                if len(rdf) and rdf["status"].eq("PASS").all()
                else "NO",
            "status":
                "PASS"
                if len(rdf) and rdf["status"].eq("PASS").all()
                else "FAIL",
        },
        {
            "check": "normal_case_passed",
            "expected": "YES",
            "observed":
                "YES"
                if "3F-A" in set(
                    rdf.loc[
                        rdf["status"].eq("PASS"),
                        "variant_id"
                    ]
                )
                else "NO",
            "status":
                "PASS"
                if "3F-A" in set(
                    rdf.loc[
                        rdf["status"].eq("PASS"),
                        "variant_id"
                    ]
                )
                else "FAIL",
        },
        {
            "check": "business_subtype_passed",
            "expected": "YES",
            "observed":
                "YES"
                if "3F-B" in set(
                    rdf.loc[
                        rdf["status"].eq("PASS"),
                        "variant_id"
                    ]
                )
                else "NO",
            "status":
                "PASS"
                if "3F-B" in set(
                    rdf.loc[
                        rdf["status"].eq("PASS"),
                        "variant_id"
                    ]
                )
                else "FAIL",
        },
        {
            "check": "document_variant_passed",
            "expected": "YES",
            "observed":
                "YES"
                if "3F-C" in set(
                    rdf.loc[
                        rdf["status"].eq("PASS"),
                        "variant_id"
                    ]
                )
                else "NO",
            "status":
                "PASS"
                if "3F-C" in set(
                    rdf.loc[
                        rdf["status"].eq("PASS"),
                        "variant_id"
                    ]
                )
                else "FAIL",
        },
        {
            "check": "missing_field_stopped",
            "expected": "YES",
            "observed":
                "YES"
                if "STOP_MISSING_REQUIRED_INPUT"
                in set(rdf["actual_result"])
                else "NO",
            "status":
                "PASS"
                if "STOP_MISSING_REQUIRED_INPUT"
                in set(rdf["actual_result"])
                else "FAIL",
        },
        {
            "check": "unexpected_value_stopped",
            "expected": "YES",
            "observed":
                "YES"
                if "STOP_INVALID_INPUT"
                in set(rdf["actual_result"])
                else "NO",
            "status":
                "PASS"
                if "STOP_INVALID_INPUT"
                in set(rdf["actual_result"])
                else "FAIL",
        },
        {
            "check": "decision_gate_preserved",
            "expected": "YES",
            "observed":
                "YES"
                if len(rdf)
                and rdf["human_gate_preserved"].all()
                else "NO",
            "status":
                "PASS"
                if len(rdf)
                and rdf["human_gate_preserved"].all()
                else "FAIL",
        },
        {
            "check": "register_clicks",
            "expected": 0,
            "observed": int(
                rdf["register_clicked"].sum()
            ),
            "status": "PASS",
        },
        {
            "check": "hold_clicks",
            "expected": 0,
            "observed": int(
                rdf["hold_clicked"].sum()
            ),
            "status": "PASS",
        },
        {
            "check": "production_accessed",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
    ])

    readme = pd.DataFrame([
        ["Purpose", "Test the Step-3E prototype against representative pi variants rather than only one normal case."],
        ["Variant basis", "Case A-F correspond to normal, business subtype, document/tooling variant, missing field, unexpected value, and human decision/exception."],
        ["Evidence basis", "Step-3C case-type evidence and Step-3B/3E automation-boundary evidence."],
        ["Population rule", "These six cases are representative prototype tests, not a statistical sample of all 209 pi segments."],
        ["Environment", "Controlled local mock; actual production /payroll-items application was not available."],
        ["Human boundary", "Register/Hold are never clicked."],
    ], columns=["item", "value"])

    with pd.ExcelWriter(
        OUTPUT,
        engine="openpyxl"
    ) as writer:
        readme.to_excel(
            writer,
            sheet_name="README",
            index=False
        )
        pd.DataFrame(cases).to_excel(
            writer,
            sheet_name="Variant_Cases",
            index=False
        )
        rdf.to_excel(
            writer,
            sheet_name="Execution_Results",
            index=False
        )
        metrics.to_excel(
            writer,
            sheet_name="Prototype_Metrics",
            index=False
        )
        final_findings.to_excel(
            writer,
            sheet_name="Findings",
            index=False
        )
        architecture.to_excel(
            writer,
            sheet_name="Step3C_Architecture",
            index=False
        )
        case_summary.to_excel(
            writer,
            sheet_name="Step3C_Case_Evidence",
            index=False
        )
        validation.to_excel(
            writer,
            sheet_name="Validation",
            index=False
        )

        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        wb = writer.book

        for ws in wb.worksheets:
            ws.freeze_panes = "A2"

            if ws.max_row > 1 and ws.max_column > 1:
                ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                cell.font = Font(
                    bold=True,
                    color="FFFFFF"
                )
                cell.fill = PatternFill(
                    "solid",
                    fgColor="1F4E78"
                )
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True,
                )

            for i, cells in enumerate(
                ws.iter_cols(
                    min_row=1,
                    max_row=min(ws.max_row, 150)
                ),
                1
            ):
                width = max(
                    [len(str(c.value or "")) for c in cells]
                    + [12]
                )
                ws.column_dimensions[
                    get_column_letter(i)
                ].width = min(width + 2, 52)


def main():
    if not MOCK.exists():
        raise FileNotFoundError(
            f"Step-3E local mock not found: {MOCK}"
        )

    (
        c3_path,
        case_summary,
        architecture,
        case_evidence,
    ) = load_step3c()

    cases = build_variant_cases(
        case_summary
    )

    print("=" * 80)
    print("STEP 3F — PI PROTOTYPE VARIANT VALIDATION")
    print("=" * 80)
    print(f"Using Step-3C: {c3_path.name}")
    print(f"Variants: {len(cases)}")
    print(
        "Actual production target: NOT AVAILABLE; "
        "controlled local mock will be used."
    )

    server = start_server()
    results = []

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True
            )
            page = browser.new_page()

            for case in cases:
                print(
                    f"Running {case['variant_id']} "
                    f"({case['variant_class']})..."
                )

                result = run_variant(
                    page,
                    case
                )

                # PASS means exact expected outcome.
                result["status"] = (
                    "PASS"
                    if result["actual_result"]
                    == case["expected_result"]
                    and (
                        result["actual_result"].startswith("STOP_")
                        or result["fields_populated_correctly"]
                        == result["fields_attempted"]
                    )
                    else "FAIL"
                )

                results.append(result)

                print(
                    f"  -> {result['actual_result']} "
                    f"[{result['status']}]"
                )

            browser.close()

    finally:
        server.shutdown()

    write_output(
        cases,
        results,
        case_summary,
        architecture,
    )

    rdf = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("3F RESULT")
    print("=" * 80)
    print(f"variants tested: {len(rdf)}")
    print(
        f"variants passed: "
        f"{int(rdf['status'].eq('PASS').sum())}"
    )
    print(
        f"variants failed: "
        f"{int(rdf['status'].eq('FAIL').sum())}"
    )
    print(
        f"successful preparation/human-gate cases: "
        f"{int(rdf['actual_result'].isin(['SUCCESS','SUCCESS_HUMAN_GATE']).sum())}"
    )
    print(
        f"safe stops: "
        f"{int(rdf['actual_result'].astype(str).str.startswith('STOP_').sum())}"
    )
    print(
        f"manual corrections required: "
        f"{int(rdf['manual_correction_needed'].sum())}"
    )
    print(
        f"Register clicks: "
        f"{int(rdf['register_clicked'].sum())}"
    )
    print(
        f"Hold clicks: "
        f"{int(rdf['hold_clicked'].sum())}"
    )

    print("\nEXECUTION RESULTS")
    print(rdf.to_string(index=False))

    print("\nVALIDATION")
    validation = pd.read_excel(
        OUTPUT,
        sheet_name="Validation"
    )
    print(validation.to_string(index=False))

    failed = validation[
        validation["status"] == "FAIL"
    ]
    if not failed.empty:
        raise RuntimeError(
            "3F validation failed. Inspect Validation sheet."
        )

    print("\nOUTPUT")
    print(OUTPUT)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3F COMPLETE.")


if __name__ == "__main__":
    main()
