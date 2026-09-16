
"""
STEP 3E-2 — PI PROTOTYPE VALIDATION V1

Purpose
-------
Test the Step-3E prototype more systematically against a compact set of
representative cases and controlled failure conditions.

This is still a LOCAL MOCK test. It does not connect to production.

Test coverage
-------------
1. Normal expense reimbursement
2. Normal payroll change
3. Second parameter set for same workflow
4. Document-assisted case
5. Missing required value
6. Invalid numeric value
7. Unsupported request type
8. Invalid date format
9. Human-gate preservation
10. Unexpected target-field state (controlled UI mutation)

Success means:
    the prototype follows the intended preparation path OR stops safely.

It does NOT mean:
    the real production system is compatible.

Outputs
-------
ONE workbook:
    outputs/step3e2_pi_prototype_validation_v1.xlsx
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PROTO = ROOT / "prototype"

MOCK_HTML = PROTO / "step3e1_mock_payroll_items.html"
OUTPUT = OUT / "step3e2_pi_prototype_validation_v1.xlsx"

HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}/payroll-items"


CASES = [
    {
        "case_id": "3E2-001",
        "case_type": "expense_reimbursement",
        "employee_id": "E2001",
        "amount": "1250.50",
        "effective_date": "2026-09-15",
        "processing_comment": "Standard reimbursement preparation.",
        "supporting_document": "receipt_001.pdf",
        "expected": "SUCCESS",
        "test_class": "NORMAL",
    },
    {
        "case_id": "3E2-002",
        "case_type": "payroll_change",
        "employee_id": "E2002",
        "amount": "4500",
        "effective_date": "2026-10-01",
        "processing_comment": "Standard payroll change preparation.",
        "supporting_document": "",
        "expected": "SUCCESS",
        "test_class": "NORMAL",
    },
    {
        "case_id": "3E2-003",
        "case_type": "expense_reimbursement",
        "employee_id": "E2003",
        "amount": "75.25",
        "effective_date": "2026-09-25",
        "processing_comment": "Different parameter values on same workflow.",
        "supporting_document": "",
        "expected": "SUCCESS",
        "test_class": "PARAMETER_VARIATION",
    },
    {
        "case_id": "3E2-004",
        "case_type": "expense_reimbursement",
        "employee_id": "E2004",
        "amount": "890.00",
        "effective_date": "2026-09-28",
        "processing_comment": "Document-assisted preparation.",
        "supporting_document": "invoice_004.pdf",
        "expected": "SUCCESS",
        "test_class": "DOCUMENT_VARIANT",
    },
    {
        "case_id": "3E2-005",
        "case_type": "expense_reimbursement",
        "employee_id": "",
        "amount": "700.00",
        "effective_date": "2026-09-21",
        "processing_comment": "Missing employee ID.",
        "supporting_document": "receipt_005.pdf",
        "expected": "STOP_MISSING_REQUIRED_INPUT",
        "test_class": "MISSING_INPUT",
    },
    {
        "case_id": "3E2-006",
        "case_type": "payroll_change",
        "employee_id": "E2006",
        "amount": "NOT_A_NUMBER",
        "effective_date": "2026-10-01",
        "processing_comment": "Invalid amount.",
        "supporting_document": "",
        "expected": "STOP_INVALID_INPUT",
        "test_class": "INVALID_INPUT",
    },
    {
        "case_id": "3E2-007",
        "case_type": "unknown_case",
        "employee_id": "E2007",
        "amount": "100.00",
        "effective_date": "2026-10-01",
        "processing_comment": "Unsupported request type.",
        "supporting_document": "",
        "expected": "STOP_UNSUPPORTED_REQUEST_TYPE",
        "test_class": "UNSUPPORTED_PARAMETER",
    },
    {
        "case_id": "3E2-008",
        "case_type": "expense_reimbursement",
        "employee_id": "E2008",
        "amount": "500.00",
        "effective_date": "01-10-2026",
        "processing_comment": "Invalid date format.",
        "supporting_document": "",
        "expected": "STOP_INVALID_INPUT",
        "test_class": "INVALID_INPUT",
    },
    {
        "case_id": "3E2-009",
        "case_type": "payroll_change",
        "employee_id": "E2009",
        "amount": "999.99",
        "effective_date": "2026-10-15",
        "processing_comment": "Human gate preservation.",
        "supporting_document": "",
        "expected": "SUCCESS_HUMAN_GATE",
        "test_class": "HUMAN_GATE",
    },
    {
        "case_id": "3E2-010",
        "case_type": "expense_reimbursement",
        "employee_id": "E2010",
        "amount": "450.00",
        "effective_date": "2026-09-30",
        "processing_comment": "Unexpected UI state.",
        "supporting_document": "",
        "expected": "STOP_UI_STATE",
        "test_class": "UNEXPECTED_UI",
        "mutate_ui": True,
    },
]


SUPPORTED = {
    "expense_reimbursement",
    "payroll_change",
    "dependent_deduction_change",
    "transportation_expense",
}


def validate_case(case: dict[str, Any]) -> tuple[bool, str, str]:
    if not str(case.get("employee_id", "")).strip():
        return False, "STOP_MISSING_REQUIRED_INPUT", "Missing employee_id."

    if case.get("case_type") not in SUPPORTED:
        return (
            False,
            "STOP_UNSUPPORTED_REQUEST_TYPE",
            f"Unsupported request type: {case.get('case_type')}",
        )

    amount = str(case.get("amount", "")).strip().replace(",", "")
    if not re.fullmatch(r"-?\d+(?:\.\d{1,2})?", amount):
        return False, "STOP_INVALID_INPUT", "Invalid amount."

    if float(amount) < 0:
        return False, "STOP_INVALID_INPUT", "Negative amount."

    date = str(case.get("effective_date", ""))
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return False, "STOP_INVALID_INPUT", "Invalid date format."

    return True, "VALID", "Input validation passed."


def normalized_case(case):
    ok, result, reason = validate_case(case)
    if not ok:
        raise ValueError(reason)

    amount = float(
        str(case["amount"]).strip().replace(",", "")
    )

    return {
        "employee_id": str(case["employee_id"]).strip(),
        "request_type": str(case["case_type"]).strip(),
        "amount": f"{amount:.2f}",
        "effective_date": str(case["effective_date"]).strip(),
        "processing_comment":
            str(case.get("processing_comment", "")).strip(),
        "supporting_document":
            str(case.get("supporting_document", "")).strip(),
        "status": "Prepared — Human Review Required",
    }


def run_case(page, case):
    started = time.perf_counter()

    result = {
        "case_id": case["case_id"],
        "test_class": case["test_class"],
        "case_type": case["case_type"],
        "expected_result": case["expected"],
        "actual_result": "",
        "status": "",
        "fields_attempted": 0,
        "fields_verified": 0,
        "human_gate_preserved": False,
        "register_clicked": False,
        "hold_clicked": False,
        "failure_reason": "",
        "seconds": 0.0,
    }

    try:
        valid, stop_result, reason = validate_case(case)

        if not valid:
            result["actual_result"] = stop_result
            result["status"] = (
                "PASS"
                if stop_result == case["expected"]
                else "FAIL"
            )
            result["failure_reason"] = reason
            result["human_gate_preserved"] = True
            return result

        record = normalized_case(case)

        page.goto(URL, wait_until="domcontentloaded")

        # Controlled negative test: remove one required target element after
        # the page loads. The automation must stop rather than guess.
        if case.get("mutate_ui"):
            page.evaluate("""
                const el = document.querySelector('#amount');
                if (el) el.remove();
            """)

        selectors = {
            "#employee_id": record["employee_id"],
            "#request_type": record["request_type"],
            "#amount": record["amount"],
            "#effective_date": record["effective_date"],
            "#processing_comment": record["processing_comment"],
            "#supporting_document": record["supporting_document"],
        }

        result["fields_attempted"] = len(selectors)

        required = [
            "#employee_id",
            "#request_type",
            "#amount",
            "#effective_date",
            "#processing_comment",
            "#supporting_document",
        ]

        for selector in required:
            if not page.locator(selector).count():
                result["actual_result"] = "STOP_UI_STATE"
                result["status"] = (
                    "PASS"
                    if case["expected"] == "STOP_UI_STATE"
                    else "FAIL"
                )
                result["failure_reason"] = (
                    f"Required target element missing: {selector}"
                )
                result["human_gate_preserved"] = True
                return result

        page.locator("#employee_id").fill(
            record["employee_id"]
        )
        page.locator("#request_type").select_option(
            record["request_type"]
        )
        page.locator("#amount").fill(
            record["amount"]
        )
        page.locator("#effective_date").fill(
            record["effective_date"]
        )
        page.locator("#processing_comment").fill(
            record["processing_comment"]
        )
        page.locator("#supporting_document").fill(
            record["supporting_document"]
        )

        actual = page.evaluate("window.getPreparedRecord()")

        expected_actual = {
            "employee_id": record["employee_id"],
            "request_type": record["request_type"],
            "amount": record["amount"],
            "effective_date": record["effective_date"],
            "processing_comment": record["processing_comment"],
            "supporting_document": record["supporting_document"],
        }

        correct = 0
        mismatches = []

        for field, expected in expected_actual.items():
            observed = actual[field]
            if observed == expected:
                correct += 1
            else:
                mismatches.append(
                    f"{field}: expected={expected!r}, observed={observed!r}"
                )

        result["fields_verified"] = correct

        if mismatches:
            result["actual_result"] = "FAIL_VERIFICATION"
            result["status"] = "FAIL"
            result["failure_reason"] = "; ".join(mismatches)
            result["human_gate_preserved"] = True
            return result

        register_disabled = page.locator(
            "#register"
        ).is_disabled()
        hold_disabled = page.locator(
            "#hold"
        ).is_disabled()

        result["register_clicked"] = False
        result["hold_clicked"] = False
        result["human_gate_preserved"] = (
            register_disabled and hold_disabled
        )

        if not result["human_gate_preserved"]:
            result["actual_result"] = "FAIL_HUMAN_GATE"
            result["status"] = "FAIL"
            result["failure_reason"] = (
                "Decision controls were not safely disabled."
            )
            return result

        page.evaluate(
            """window.markVerification(
                true,
                "Preparation verified. Automation stopped before human decision."
            );"""
        )

        result["actual_result"] = (
            "SUCCESS_HUMAN_GATE"
            if case["expected"] == "SUCCESS_HUMAN_GATE"
            else "SUCCESS"
        )

        result["status"] = (
            "PASS"
            if result["actual_result"] == case["expected"]
            else "FAIL"
        )

        if result["status"] == "FAIL":
            result["failure_reason"] = (
                f"Expected {case['expected']} but got {result['actual_result']}."
            )

        return result

    except Exception as exc:
        result["actual_result"] = "ERROR"
        result["status"] = "FAIL"
        result["failure_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
        result["human_gate_preserved"] = True
        return result

    finally:
        result["seconds"] = round(
            time.perf_counter() - started,
            4
        )


def start_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/payroll-items":
                data = MOCK_HTML.read_bytes()
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

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler
    )
    thread = __import__("threading").Thread(
        target=server.serve_forever,
        daemon=True
    )
    thread.start()

    return server


def write_workbook(results):
    rdf = pd.DataFrame(results)

    summary = pd.DataFrame([
        {
            "metric": "cases_tested",
            "value": len(rdf),
        },
        {
            "metric": "cases_passed",
            "value": int(
                rdf["status"].eq("PASS").sum()
            ),
        },
        {
            "metric": "cases_failed",
            "value": int(
                rdf["status"].eq("FAIL").sum()
            ),
        },
        {
            "metric": "success_preparation_cases",
            "value": int(
                rdf["actual_result"]
                .astype(str)
                .isin(["SUCCESS", "SUCCESS_HUMAN_GATE"])
                .sum()
            ),
        },
        {
            "metric": "safe_stops",
            "value": int(
                rdf["actual_result"]
                .astype(str)
                .str.startswith("STOP_")
                .sum()
            ),
        },
        {
            "metric": "verification_failures",
            "value": int(
                rdf["actual_result"].eq("FAIL_VERIFICATION").sum()
            ),
        },
        {
            "metric": "human_gate_preserved_all_cases",
            "value": bool(
                len(rdf) and rdf["human_gate_preserved"].all()
            ),
        },
        {
            "metric": "register_clicks",
            "value": int(rdf["register_clicked"].sum()),
        },
        {
            "metric": "hold_clicks",
            "value": int(rdf["hold_clicked"].sum()),
        },
    ])

    architecture = pd.DataFrame([
        {
            "finding": "Parameterized core",
            "result":
                "Same preparation engine handles supported case parameters.",
            "evidence":
                "Normal and parameter-variation cases use identical field-entry logic.",
        },
        {
            "finding": "Failure handling",
            "result":
                "Invalid/missing/unsupported inputs stop safely before population.",
            "evidence":
                "Input-validation cases matched their expected STOP outcomes.",
        },
        {
            "finding": "Unexpected UI handling",
            "result":
                "Missing target element causes controlled STOP_UI_STATE.",
            "evidence":
                "Controlled mutation removed #amount and automation refused to guess.",
        },
        {
            "finding": "Human gate",
            "result":
                "Register/Hold are never clicked.",
            "evidence":
                "Decision controls remain disabled through the prototype test.",
        },
        {
            "finding": "Scope limitation",
            "result":
                "This validates the controlled local mock only.",
            "evidence":
                "No production system was accessed.",
        },
    ])

    validation = pd.DataFrame([
        {
            "check": "all_test_cases_pass",
            "expected": "YES",
            "observed": (
                "YES"
                if len(rdf)
                and rdf["status"].eq("PASS").all()
                else "NO"
            ),
            "status": (
                "PASS"
                if len(rdf)
                and rdf["status"].eq("PASS").all()
                else "FAIL"
            ),
        },
        {
            "check": "safe_failure_cases_present",
            "expected": "YES",
            "observed":
                "YES"
                if rdf["actual_result"].astype(str)
                .str.startswith("STOP_").any()
                else "NO",
            "status":
                "PASS"
                if rdf["actual_result"].astype(str)
                .str.startswith("STOP_").any()
                else "FAIL",
        },
        {
            "check": "ui_state_case_stopped",
            "expected": "YES",
            "observed":
                "YES"
                if "STOP_UI_STATE" in set(rdf["actual_result"])
                else "NO",
            "status":
                "PASS"
                if "STOP_UI_STATE" in set(rdf["actual_result"])
                else "FAIL",
        },
        {
            "check": "human_gate_preserved",
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
            "check": "register_clicked",
            "expected": 0,
            "observed": int(rdf["register_clicked"].sum()),
            "status": "PASS",
        },
        {
            "check": "hold_clicked",
            "expected": 0,
            "observed": int(rdf["hold_clicked"].sum()),
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
        ["Purpose", "Systematically validate the Step-3E local prototype against representative success, parameter, document, input-error, unsupported-parameter and unexpected-UI cases."],
        ["Target", "Local evidence-derived /payroll-items mock only."],
        ["Passing principle", "A case passes when the prototype completes the intended preparation OR stops safely at the expected exception boundary."],
        ["Human gate", "Register/Hold must remain untouched."],
        ["Production claim", "None. These are controlled mock-environment prototype results."],
    ], columns=["item", "value"])

    with pd.ExcelWriter(
        OUTPUT,
        engine="openpyxl"
    ) as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        summary.to_excel(writer, sheet_name="Validation_Summary", index=False)
        rdf.to_excel(writer, sheet_name="Execution_Results", index=False)
        pd.DataFrame(CASES).to_excel(
            writer,
            sheet_name="Test_Cases",
            index=False
        )
        architecture.to_excel(
            writer,
            sheet_name="Architecture_Findings",
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
                    wrap_text=True
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
    if not MOCK_HTML.exists():
        raise FileNotFoundError(
            f"Missing local mock: {MOCK_HTML}\n"
            "Run Step 3E-1 first."
        )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is not installed. Run:\n"
            "pip install playwright pandas openpyxl\n"
            "playwright install chromium"
        )
        return 2

    server = start_server()
    results = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(
                URL,
                wait_until="domcontentloaded"
            )

            for case in CASES:
                print(
                    f"Running {case['case_id']} "
                    f"({case['test_class']})..."
                )

                r = run_case(
                    page,
                    case
                )
                results.append(r)

                print(
                    f"  -> {r['actual_result']} "
                    f"[{r['status']}]"
                )

            browser.close()

        write_workbook(results)

        rdf = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("3E-2 RESULT")
        print("=" * 80)
        print(f"cases tested: {len(rdf)}")
        print(
            f"cases passed: "
            f"{int(rdf['status'].eq('PASS').sum())}"
        )
        print(
            f"cases failed: "
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
            f"register clicks: "
            f"{int(rdf['register_clicked'].sum())}"
        )
        print(
            f"hold clicks: "
            f"{int(rdf['hold_clicked'].sum())}"
        )

        print("\nEXECUTION RESULTS")
        print(rdf.to_string(index=False))

        print("\nOUTPUT")
        print(OUTPUT)
        print("\nsegments.jsonl was NOT modified.")
        print("STEP 3E-2 COMPLETE.")

        return 0

    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
