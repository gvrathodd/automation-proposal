
"""
STEP 3E-0 — ENVIRONMENT DISCOVERY

Run from the automation-proposal repository root.

Purpose:
    Determine whether a runnable local implementation/sandbox/mock of the
    observed pi /payroll-items interface exists BEFORE writing automation.

Output:
    outputs/step3e0_environment_discovery.xlsx
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

TEXT_EXT = {
    ".py",".js",".jsx",".ts",".tsx",".json",".html",".htm",".css",".scss",
    ".md",".txt",".yaml",".yml",".toml",".ini",".env.example"
}

IGNORE_DIRS = {
    ".git",".venv","venv","node_modules","__pycache__",".pytest_cache",
    ".mypy_cache","dist","build",".idea",".vscode","coverage"
}

PATTERNS = {
    "payroll_route": re.compile(r"/payroll-items|payroll-items", re.I),
    "web_router": re.compile(
        r"<Route\b|createBrowserRouter|Routes\b|router\b|FastAPI\b|Flask\b|"
        r"@app\.(get|post|put|patch|delete)\b", re.I
    ),
    "localhost": re.compile(r"localhost|127\.0\.0\.1", re.I),
    "mock_fixture": re.compile(
        r"\bmock\b|\bfixture\b|\bseed\b|\bfake\b|\bdummy\b|\bstub\b",
        re.I
    ),
    "package": re.compile(
        r'"scripts"\s*:|"dependencies"\s*:|"devDependencies"\s*:',
        re.I
    ),
}

def scan_files():
    records = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        rel = path.relative_to(ROOT)
        suffix = path.suffix.lower()

        if suffix not in TEXT_EXT:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        hits = []
        for name, pat in PATTERNS.items():
            if pat.search(text):
                hits.append(name)

        if hits:
            records.append({
                "file": str(rel),
                "extension": suffix,
                "size_bytes": path.stat().st_size,
                "signals": " | ".join(hits),
                "payroll_route_hits": len(
                    PATTERNS["payroll_route"].findall(text)
                ),
                "router_hits": len(
                    PATTERNS["web_router"].findall(text)
                ),
                "localhost_hits": len(
                    PATTERNS["localhost"].findall(text)
                ),
                "mock_hits": len(
                    PATTERNS["mock_fixture"].findall(text)
                ),
            })
    return pd.DataFrame(records)

def inspect_top_level():
    rows = []
    for p in sorted(ROOT.iterdir(), key=lambda x: x.name.lower()):
        rows.append({
            "name": p.name,
            "type": "DIR" if p.is_dir() else "FILE",
            "extension": p.suffix.lower(),
        })
    return pd.DataFrame(rows)

def classify(files, top):
    names = set(top["name"].astype(str).str.lower()) if not top.empty else set()

    has_package = "package.json" in names
    has_requirements = "requirements.txt" in names
    has_docker = any(
        n.startswith("dockerfile") or n.startswith("docker-compose")
        for n in names
    )

    route_files = files[
        files["payroll_route_hits"].gt(0)
    ] if not files.empty else pd.DataFrame()

    router_files = files[
        files["router_hits"].gt(0)
    ] if not files.empty else pd.DataFrame()

    local_files = files[
        files["localhost_hits"].gt(0)
    ] if not files.empty else pd.DataFrame()

    mock_files = files[
        files["mock_hits"].gt(0)
    ] if not files.empty else pd.DataFrame()

    # Strongest evidence: actual route + router/application structure.
    if not route_files.empty and not router_files.empty:
        classification = "POTENTIAL EXECUTABLE LOCAL TARGET"
        reason = (
            "The repository contains /payroll-items references and executable "
            "web/application routing signals. A runtime test is still required."
        )
    elif not route_files.empty and (has_package or has_requirements or has_docker):
        classification = "PARTIAL REPRODUCTION"
        reason = (
            "The repository references /payroll-items and has application "
            "infrastructure, but no sufficiently strong route/router pairing was found."
        )
    elif not mock_files.empty:
        classification = "MOCK/TEST ENVIRONMENT CANDIDATE"
        reason = (
            "Mock/fixture/seed/test signals exist, but an executable payroll route "
            "was not established by repository inspection."
        )
    else:
        classification = "NO EXECUTABLE SANDBOX EVIDENCE"
        reason = (
            "Repository scan did not establish a runnable /payroll-items "
            "implementation. This is not proof that no external sandbox exists."
        )

    return classification, reason, {
        "route_files": len(route_files),
        "router_files": len(router_files),
        "localhost_files": len(local_files),
        "mock_files": len(mock_files),
        "package_json": has_package,
        "requirements_txt": has_requirements,
        "docker_files": has_docker,
    }

def main():
    files = scan_files()
    top = inspect_top_level()
    classification, reason, stats = classify(files, top)

    result = pd.DataFrame([{
        "classification": classification,
        "reason": reason,
        **stats
    }])

    next_steps = pd.DataFrame([
        {
            "step": 1,
            "action": "Inspect candidate files containing /payroll-items",
            "purpose": "Confirm whether the string is an actual route/component or merely dataset evidence."
        },
        {
            "step": 2,
            "action": "Identify application start command",
            "purpose": "Determine whether a local web application can actually be launched."
        },
        {
            "step": 3,
            "action": "Launch only local/sandbox target",
            "purpose": "Never use production credentials or an unknown external system for the prototype."
        },
        {
            "step": 4,
            "action": "Open /payroll-items",
            "purpose": "Verify that the observed target page exists in the local environment."
        },
        {
            "step": 5,
            "action": "Compare visible fields against Step-3A OCR evidence",
            "purpose": "Establish interface fidelity before writing automation."
        },
    ])

    readme = pd.DataFrame([
        ["Question", "Does the current project contain a reproducible local sandbox/mock target for pi?"],
        ["Observed route from Dataset B", "/payroll-items"],
        ["Rule", "Repository text alone does not prove an executable application."],
        ["Safety", "Do not connect the prototype to an unknown production system."],
        ["Next", "If an executable target exists, verify it locally; otherwise use a faithful local mock."],
    ], columns=["item","value"])

    out = OUT / "step3e0_environment_discovery.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        result.to_excel(writer, sheet_name="Result", index=False)
        top.to_excel(writer, sheet_name="Repository_Overview", index=False)
        files.to_excel(writer, sheet_name="Candidate_Files", index=False)
        next_steps.to_excel(writer, sheet_name="Next_Steps", index=False)

    print("=" * 80)
    print("STEP 3E-0 — ENVIRONMENT DISCOVERY")
    print("=" * 80)
    print(f"Repository: {ROOT}")
    print(f"Candidate text files with signals: {len(files)}")
    print("\nRESULT")
    print(result.to_string(index=False))
    print("\nTOP CANDIDATE FILES")
    if files.empty:
        print("No candidate files found.")
    else:
        print(files.sort_values(
            ["payroll_route_hits","router_hits","localhost_hits"],
            ascending=False
        ).head(30).to_string(index=False))
    print("\nOUTPUT")
    print(out)

if __name__ == "__main__":
    main()
