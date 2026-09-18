
from __future__ import annotations
import json, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "raw_data" / "dataset-downloads"
SEGMENTS = ROOT / "segments.jsonl"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()

def low(v: Any) -> str:
    return re.sub(r"\s+", " ", norm(v).lower()).strip()

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def discover_event_files(root: Path = DATA_ROOT) -> list[Path]:
    return sorted(root.rglob("events.jsonl"))

def identify_dataset(path: Path) -> str | None:
    for part in path.parts:
        p = part.lower()
        if p == "dataset_a":
            return "A"
        if p == "dataset_b":
            return "B"
    return None

def load_b_events() -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    files = [p for p in discover_event_files() if identify_dataset(p) == "B"]
    if not files:
        raise FileNotFoundError(
            f"No Dataset-B events.jsonl found below {DATA_ROOT}. "
            "Place the original raw dataset under raw_data/dataset-downloads/."
        )
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in files:
        for e in read_jsonl(p):
            sid = e.get("session_id")
            if sid:
                e["_source_file"] = str(p)
                e["_dataset"] = "B"
                sessions[str(sid)].append(e)
    for sid, evs in sessions.items():
        evs.sort(key=lambda e: (
            int(e.get("timestamp_ms", 0) or 0),
            int((e.get("correlation") or {}).get("sequence_number", 0) or 0),
        ))
    return sessions, files

def event_ts(e: dict[str, Any]) -> int:
    return int(e.get("timestamp_ms", 0) or 0)

def payload(e: dict[str, Any]) -> dict[str, Any]:
    p = e.get("payload")
    return p if isinstance(p, dict) else {}

def context(e: dict[str, Any]) -> dict[str, Any]:
    c = e.get("context")
    return c if isinstance(c, dict) else {}

def active_app(e: dict[str, Any]) -> str:
    a = context(e).get("active_app") or {}
    if isinstance(a, dict):
        return norm(a.get("name") or a.get("app_name") or a.get("process_name") or a.get("title"))
    return norm(a)

def routes_from(e: dict[str, Any]) -> list[str]:
    vals = []
    c = context(e)
    p = payload(e)
    for obj in (c, p, e):
        if isinstance(obj, dict):
            for key in ("routes", "route", "url", "href", "page_url"):
                v = obj.get(key)
                if isinstance(v, list):
                    vals.extend(norm(x) for x in v)
                else:
                    vals.append(norm(v))
    return [x for x in vals if x]

def event_type(e: dict[str, Any]) -> str:
    return low(e.get("event_type"))

def file_refs(e: dict[str, Any]) -> list[str]:
    vals = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "filename" and isinstance(v, str):
                    vals.append(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(payload(e))
    return list(dict.fromkeys(vals))

def get_field(e: dict[str, Any], *names: str) -> str:
    p = payload(e)
    c = context(e)
    for obj in (p, c, e):
        if isinstance(obj, dict):
            for n in names:
                if n in obj and norm(obj[n]):
                    return norm(obj[n])
    return ""

def load_la_segments() -> pd.DataFrame:
    if not SEGMENTS.exists():
        raise FileNotFoundError(f"Missing {SEGMENTS}")
    rows = []
    for x in read_jsonl(SEGMENTS):
        if norm(x.get("label")).lower() != "la":
            continue
        rows.append({
            "session_id": norm(x.get("session_id")),
            "start": norm(x.get("start")),
            "end": norm(x.get("end")),
            "label": "la",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No label='la' segments found in segments.jsonl.")
    df["segment_index"] = df.groupby("session_id").cumcount()
    df["start_dt"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df["start_ms"] = (df["start_dt"].astype("int64") // 10**6)
    df["end_ms"] = (df["end_dt"].astype("int64") // 10**6)
    df["duration_seconds"] = (df["end_ms"] - df["start_ms"]) / 1000.0
    df["segment_key"] = df["session_id"] + "::" + df["segment_index"].astype(str)
    return df

def assign_events(segments: pd.DataFrame, sessions: dict[str, list[dict[str, Any]]]):
    rows = []
    for _, s in segments.iterrows():
        for e in sessions.get(s["session_id"], []):
            ts = event_ts(e)
            if int(s["start_ms"]) <= ts <= int(s["end_ms"]):
                r = dict(e)
                r["_segment_index"] = int(s["segment_index"])
                r["_segment_key"] = s["segment_key"]
                rows.append(r)
    return rows

def classify_app(s: str) -> str:
    q = low(s)
    if any(x in q for x in ("edge", "chrome", "firefox", "browser")):
        return "Browser"
    if "word" in q or "winword" in q:
        return "Word"
    if "excel" in q:
        return "Excel"
    if "notepad" in q:
        return "Notepad"
    if "explorer" in q:
        return "Explorer"
    if "powershell" in q or "terminal" in q or q == "cmd":
        return "Terminal"
    return s or "UNKNOWN"

def phase_for_event(e: dict[str, Any]) -> str:
    et = event_type(e)
    if et == "browser_navigation":
        return "BROWSER_NAVIGATION"
    if et in {"browser_click", "browser_form_input"}:
        return "BROWSER_WORK"
    if et == "clipboard_change":
        return "DATA_TRANSFER"
    if et in {"app_switch", "window_change"}:
        return "APPLICATION_SWITCH"
    if et in {"keystroke", "keyboard", "mouse_click", "mouse_scroll", "shortcut"}:
        return "DESKTOP_INPUT"
    if et.startswith("document") or et in {"file_open", "file_save"}:
        return "DOCUMENT_WORK"
    return "BROWSER_WORK" if "browser" in et else "DESKTOP_INPUT"

def excel_write(path: Path, sheets: dict[str, pd.DataFrame]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            safe = str(name)[:31]
            df.to_excel(w, sheet_name=safe, index=False)
    return path
