"""
STEP 3A-1 — EXTRACT ALL PI SEGMENT EVIDENCE

Purpose:
    Use the canonical segments.jsonl to isolate all 209 `pi` segments and
    reconstruct their available event/screenshot/UI evidence from Dataset-B.

Inputs expected in the local repository:
    ./segments.jsonl
    ./raw_data/dataset-downloads/**/dataset_b/**/events.jsonl
    ./outputs/dataset_b_ocr_translated_20pct.jsonl   (optional)

Outputs:
    ./outputs/step3a1_pi_segment_evidence.csv
    ./outputs/step3a1_pi_segment_event_detail.csv
    ./outputs/step3a1_pi_segment_summary.csv

Important:
    - Does NOT change segments.jsonl.
    - Clipboard payload text is not written to output; only length/type/hash.
    - Actual OCR text is not written to the per-segment table by default.
      OCR-derived Japanese/English snippets are summarized separately.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
SEGMENTS_FILE = REPO / "segments.jsonl"
OUTPUT_DIR = REPO / "outputs"

# Optional OCR source. The script will also search outputs recursively.
OCR_CANDIDATES = [
    OUTPUT_DIR / "dataset_b_ocr_translated_20pct.jsonl",
    OUTPUT_DIR / "dataset_b_ocr_translated.jsonl",
]

EVENT_TYPE_ORDER = [
    "app_switch",
    "browser_navigation",
    "browser_click",
    "browser_form_input",
    "clipboard_change",
    "keystroke",
    "shortcut",
    "mouse_click",
    "mouse_scroll",
    "screenshot_smart",
    "text_input_complete",
    "window_title_change",
    "window_state_change",
]


def parse_dt(value: str) -> float:
    """Parse ISO timestamp to epoch seconds without requiring pandas."""
    from datetime import datetime, timezone

    s = str(value).strip()
    if not s:
        raise ValueError("empty timestamp")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()


def walk_strings(obj: Any, max_items: int = 2000) -> list[str]:
    out: list[str] = []

    def rec(x: Any) -> None:
        if len(out) >= max_items:
            return
        if isinstance(x, str):
            if x.strip():
                out.append(x.strip())
        elif isinstance(x, dict):
            for v in x.values():
                rec(v)
                if len(out) >= max_items:
                    break
        elif isinstance(x, list):
            for v in x:
                rec(v)
                if len(out) >= max_items:
                    break

    rec(obj)
    return out


def flatten_dict(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p, v
            yield from flatten_dict(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield p, v
            yield from flatten_dict(v, p)


def collect_matching_values(obj: Any, key_patterns: list[str]) -> list[str]:
    found: list[str] = []
    pats = [re.compile(p, re.I) for p in key_patterns]
    for path, value in flatten_dict(obj):
        leaf = path.rsplit(".", 1)[-1]
        if isinstance(value, (str, int, float, bool)) and any(p.search(leaf) for p in pats):
            s = str(value).strip()
            if s:
                found.append(s)
    return found


def unique_join(values: Iterable[str], limit: int = 25) -> str:
    seen = []
    done = set()
    for v in values:
        s = str(v).strip()
        if s and s not in done:
            done.add(s)
            seen.append(s)
            if len(seen) >= limit:
                break
    return " | ".join(seen)


def find_dataset_b_event_files() -> list[Path]:
    roots = [
        REPO / "raw_data",
        REPO / "data",
    ]
    files: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue
        # Restrict to Dataset-B event files.
        for p in root.rglob("events.jsonl"):
            s = str(p).lower().replace("\\", "/")
            if "/dataset_b/" in s or "dataset_b-" in s or "dataset-b" in s:
                files.add(p)

    # Fallback if the folder naming differs.
    if not files:
        for root in roots:
            if not root.exists():
                continue
            files.update(root.rglob("events.jsonl"))

    return sorted(files)


def load_pi_segments() -> list[dict[str, Any]]:
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing canonical segments file: {SEGMENTS_FILE}")

    rows = []
    with SEGMENTS_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = obj.get("label")
            if label != "pi":
                continue

            # Canonical segment artifact may use different timestamp key names.
            start = obj.get("start") or obj.get("start_ts") or obj.get("segment_start")
            end = obj.get("end") or obj.get("end_ts") or obj.get("segment_end")
            if not start or not end:
                raise ValueError(
                    f"pi segment missing start/end at segments.jsonl line {line_no}"
                )

            rows.append({
                "session_id": obj["session_id"],
                "label": "pi",
                "start": start,
                "end": end,
                "start_s": parse_dt(start),
                "end_s": parse_dt(end),
                "duration_s": parse_dt(end) - parse_dt(start),
                "application_variant": obj.get("application_variant", ""),
                "prefix_strength": obj.get("prefix_strength", ""),
                "_source_line": line_no,
            })

    # segments.jsonl contains session_id/start/end/label only. Reconstruct the
    # canonical 1-based segment index from chronological segment order within
    # each session. This matches the index used by the earlier Step-2 outputs.
    rows.sort(key=lambda r: (r["session_id"], r["start_s"], r["end_s"], r["_source_line"]))
    counters = defaultdict(int)
    for r in rows:
        counters[r["session_id"]] += 1
        r["segment_index"] = counters[r["session_id"]]
        r.pop("_source_line", None)
    return rows


def load_events_for_pi(
    pi_segments: list[dict[str, Any]],
    event_files: list[Path],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in pi_segments:
        by_session[seg["session_id"]].append(seg)

    attached: dict[tuple[str, int], list[dict[str, Any]]] = {
        (s["session_id"], s["segment_index"]): [] for s in pi_segments
    }

    needed_sessions = set(by_session)

    for path in event_files:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sid = e.get("session_id")
                if sid not in needed_sessions:
                    continue

                ts = e.get("timestamp_iso")
                if not ts:
                    ms = e.get("timestamp_ms")
                    if ms is None:
                        continue
                    ts_s = float(ms) / 1000.0
                else:
                    try:
                        ts_s = parse_dt(ts)
                    except ValueError:
                        continue

                for seg in by_session[sid]:
                    if seg["start_s"] <= ts_s <= seg["end_s"]:
                        attached[(sid, seg["segment_index"])].append({
                            "event": e,
                            "source_file": str(path),
                            "source_line": line_no,
                            "timestamp_s": ts_s,
                        })
                        break

    for key in attached:
        attached[key].sort(key=lambda x: x["timestamp_s"])

    return attached


def load_ocr() -> dict[tuple[str, int], list[dict[str, Any]]]:
    path = next((p for p in OCR_CANDIDATES if p.exists()), None)
    if path is None:
        # Search more generally for a translated Dataset-B OCR JSONL.
        candidates = list(OUTPUT_DIR.rglob("*ocr*translated*.jsonl"))
        path = candidates[0] if candidates else None

    if path is None:
        return {}

    out: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            sid = obj.get("session_id")
            idx = obj.get("segment_index")
            if sid is None or idx is None:
                continue

            if obj.get("label") != "pi":
                continue

            out[(sid, int(idx))].append(obj)

    return dict(out)


def summarize_segment(
    seg: dict[str, Any],
    entries: list[dict[str, Any]],
    ocr_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    events = [x["event"] for x in entries]
    counts = Counter(e.get("event_type", "") for e in events)
    apps = []
    transitions = []
    routes = []
    dom_ids = []
    fields = []
    buttons = []
    docs = []
    windows = []
    decision_events = []
    screenshots = []
    clipboard_lengths = []
    clipboard_hashes = []
    forms = []
    typed_targets = []

    prev_app = None

    for e in events:
        et = e.get("event_type", "")
        ctx = e.get("context") or {}
        active = ctx.get("active_app") or {}
        app_name = active.get("app_name") or active.get("process_name")
        window_title = active.get("window_title") or ""
        if app_name:
            apps.append(app_name)
        if window_title:
            windows.append(window_title)

        if et == "app_switch":
            payload = e.get("payload") or {}
            new_app = (payload.get("new_app") or {}).get("app_name")
            old_app = (payload.get("previous_app") or {}).get("app_name")
            if old_app or new_app:
                transitions.append(f"{old_app or '?'} -> {new_app or '?'}")

        if et == "browser_navigation":
            payload = e.get("payload") or {}
            # Collect URL-like / route-like strings from navigation payload.
            for _, v in flatten_dict(payload):
                if isinstance(v, str):
                    if "/" in v or v.startswith("http"):
                        routes.append(v)

        if et == "browser_click":
            payload = e.get("payload") or {}
            target = payload.get("target_element") or {}
            dom_ids.extend(collect_matching_values(target, [r"automation_id", r"stable_key"]))
            n = target.get("name")
            if n:
                buttons.append(str(n))

        if et == "browser_form_input":
            payload = e.get("payload") or {}
            forms.append(unique_join(walk_strings(payload), limit=12))
            target = payload.get("target_element") or payload.get("field") or {}
            dom_ids.extend(collect_matching_values(target, [r"automation_id", r"stable_key", r"field"]))
            fields.extend(collect_matching_values(payload, [r"field", r"label", r"name", r"placeholder"]))

        if et == "text_input_complete":
            payload = e.get("payload") or {}
            tf = payload.get("target_field") or {}
            typed_targets.extend(collect_matching_values(tf, [r"automation_id", r"name"]))
            dom_ids.extend(collect_matching_values(tf, [r"automation_id", r"stable_key"]))
            if payload.get("final_text"):
                # Keep only a length/hash for potentially sensitive typed content.
                forms.append(
                    f"typed_text_len={len(str(payload['final_text']))};"
                    f"typed_text_sha1={sha1_text(str(payload['final_text']))}"
                )

        if et == "clipboard_change":
            payload = e.get("payload") or {}
            if payload.get("text_length") is not None:
                clipboard_lengths.append(int(payload["text_length"]))
            for x in payload.get("formats_available") or []:
                forms.append(f"clipboard_format={x}")
            ext = e.get("extensions") or {}
            uia = ext.get("uia_v2") or {}
            target = uia.get("target") or {}
            dom_ids.extend(collect_matching_values(target, [r"automation_id", r"stable_key"]))

        if et == "screenshot_smart":
            payload = e.get("payload") or {}
            fr = payload.get("file_reference") or {}
            fn = fr.get("filename") or fr.get("relative_path")
            if fn:
                screenshots.append(fn)

        # Generic structural harvesting.
        dom_ids.extend(collect_matching_values(e, [r"automation_id", r"dom_id", r"stable_key"]))
        fields.extend(collect_matching_values(e, [r"field_label", r"field_name", r"placeholder"]))
        docs.extend(collect_matching_values(e, [r"file_name", r"filename", r"document", r"doc_name"]))
        routes.extend(collect_matching_values(e, [r"url", r"href", r"route", r"path"]))

        # Do not use a button name alone. Capture the decision event as evidence only.
        decision_pattern = re.compile(
            r"approve|approval|reject|rejection|hold|register|confirm|"
            r"登録|承認|却下|保留|確定|決裁",
            re.I,
        )
        text_blob = " ".join(walk_strings(e, max_items=120))
        if decision_pattern.search(text_blob):
            decision_events.append(f"{et}:{text_blob[:180]}")

        # Word/Excel/Notepad document evidence from window titles.
        for t in (window_title,):
            low = t.lower()
            if any(k in low for k in ("word", "excel", "notepad")):
                docs.append(t)

    # OCR summary.
    ocr_jp = []
    ocr_en = []
    ocr_scores = []
    for o in ocr_entries:
        for row in o.get("japanese_ocr_lines") or []:
            jp = row.get("japanese")
            en = row.get("english")
            if jp:
                ocr_jp.append(str(jp))
            if en:
                ocr_en.append(str(en))
        if o.get("ocr_mean_score") is not None:
            ocr_scores.append(float(o["ocr_mean_score"]))

    row = dict(seg)
    row.update({
        "event_count": len(events),
        "event_types": unique_join(
            [f"{k}={counts[k]}" for k in sorted(counts) if k], limit=60
        ),
        "application_sequence": unique_join(apps, limit=30),
        "application_count": len(set(a for a in apps if a)),
        "application_transitions": unique_join(transitions, limit=30),
        "browser_route_or_url_evidence": unique_join(routes, limit=30),
        "dom_ids_or_stable_keys": unique_join(dom_ids, limit=40),
        "field_evidence": unique_join(fields, limit=40),
        "button_or_target_names": unique_join(buttons, limit=30),
        "document_or_window_evidence": unique_join(
            list(docs) + list(windows), limit=40
        ),
        "form_input_evidence": unique_join(forms, limit=30),
        "typed_target_fields": unique_join(typed_targets, limit=30),
        "clipboard_change_count": counts.get("clipboard_change", 0),
        "clipboard_text_lengths": unique_join(map(str, clipboard_lengths), limit=30),
        "browser_navigation_count": counts.get("browser_navigation", 0),
        "browser_click_count": counts.get("browser_click", 0),
        "browser_form_input_count": counts.get("browser_form_input", 0),
        "keyboard_event_count": counts.get("keystroke", 0),
        "shortcut_count": counts.get("shortcut", 0),
        "mouse_click_count": counts.get("mouse_click", 0),
        "scroll_count": counts.get("mouse_scroll", 0),
        "screenshot_event_count": counts.get("screenshot_smart", 0),
        "screenshot_files": unique_join(screenshots, limit=40),
        "decision_point_event_evidence": unique_join(decision_events, limit=20),
        "ocr_record_count": len(ocr_entries),
        "ocr_japanese_terms": unique_join(ocr_jp, limit=40),
        "ocr_english_terms": unique_join(ocr_en, limit=40),
        "ocr_mean_confidence": (
            sum(ocr_scores) / len(ocr_scores) if ocr_scores else ""
        ),
        "ocr_source_present": bool(ocr_entries),
    })
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    print("=" * 80)
    print("STEP 3A-1 — PI SEGMENT EVIDENCE EXTRACTION")
    print("=" * 80)

    pi_segments = load_pi_segments()
    print(f"pi segments found in segments.jsonl: {len(pi_segments)}")

    event_files = find_dataset_b_event_files()
    print(f"Dataset-B events.jsonl files discovered: {len(event_files)}")
    if not event_files:
        raise RuntimeError(
            "No Dataset-B events.jsonl files found. "
            "Run this from the repository containing raw_data/."
        )

    attached = load_events_for_pi(pi_segments, event_files)
    ocr = load_ocr()

    evidence_rows = []
    event_rows = []

    for seg in pi_segments:
        key = (seg["session_id"], seg["segment_index"])
        entries = attached.get(key, [])
        ocr_entries = ocr.get(key, [])
        evidence_rows.append(summarize_segment(seg, entries, ocr_entries))

        for x in entries:
            e = x["event"]
            event_rows.append({
                "session_id": seg["session_id"],
                "segment_index": seg["segment_index"],
                "label": "pi",
                "segment_start": seg["start"],
                "segment_end": seg["end"],
                "event_timestamp": e.get("timestamp_iso", ""),
                "event_type": e.get("event_type", ""),
                "event_id": e.get("event_id", ""),
                "active_app": ((e.get("context") or {}).get("active_app") or {}).get("app_name", ""),
                "window_title": ((e.get("context") or {}).get("active_app") or {}).get("window_title", ""),
                "source_file": x["source_file"],
                "source_line": x["source_line"],
                "payload_keys": unique_join((e.get("payload") or {}).keys(), limit=50),
                "has_uia_v2": bool(((e.get("extensions") or {}).get("uia_v2"))),
                "has_dom": bool(
                    ((e.get("extensions") or {}).get("capture_quality_v1") or {}).get("dom_available")
                ),
            })

    write_csv(OUTPUT_DIR / "step3a1_pi_segment_evidence.csv", evidence_rows)
    write_csv(OUTPUT_DIR / "step3a1_pi_segment_event_detail.csv", event_rows)

    # Compact validation summary.
    summary = []
    for col, label in [
        ("event_count", "Mean events / pi segment"),
        ("duration_s", "Mean duration / pi segment"),
        ("clipboard_change_count", "Mean clipboard changes / pi segment"),
        ("browser_navigation_count", "Mean browser navigations / pi segment"),
        ("browser_click_count", "Mean browser clicks / pi segment"),
        ("browser_form_input_count", "Mean browser form inputs / pi segment"),
        ("keyboard_event_count", "Mean keystrokes / pi segment"),
        ("screenshot_event_count", "Mean screenshot events / pi segment"),
    ]:
        vals = [float(r[col]) for r in evidence_rows if r.get(col) not in ("", None)]
        summary.append({
            "metric": label,
            "value": (sum(vals) / len(vals) if vals else 0.0),
        })

    families = Counter(r.get("application_variant", "") for r in evidence_rows)
    for k, v in sorted(families.items()):
        summary.append({"metric": f"Segments — application_variant={k or 'unknown'}", "value": v})

    summary.append({
        "metric": "Segments with any raw events",
        "value": sum(1 for r in evidence_rows if int(r["event_count"]) > 0),
    })
    summary.append({
        "metric": "Segments with OCR records",
        "value": sum(1 for r in evidence_rows if int(r["ocr_record_count"]) > 0),
    })
    summary.append({
        "metric": "Segments with decision-point evidence",
        "value": sum(1 for r in evidence_rows if r["decision_point_event_evidence"]),
    })

    write_csv(OUTPUT_DIR / "step3a1_pi_segment_summary.csv", summary)

    print("\nRESULT")
    print("-" * 80)
    print(f"pi segments: {len(pi_segments)}")
    print(f"segments with events: {sum(bool(attached.get((s['session_id'], s['segment_index']))) for s in pi_segments)}")
    print(f"event records attached to pi segments: {len(event_rows)}")
    print(f"OCR-covered pi segments: {sum(bool(ocr.get((s['session_id'], s['segment_index']))) for s in pi_segments)}")
    print(f"decision-evidence pi segments: {sum(1 for r in evidence_rows if r['decision_point_event_evidence'])}")

    print("\nOUTPUTS")
    print(OUTPUT_DIR / "step3a1_pi_segment_evidence.csv")
    print(OUTPUT_DIR / "step3a1_pi_segment_event_detail.csv")
    print(OUTPUT_DIR / "step3a1_pi_segment_summary.csv")
    print("\nsegments.jsonl was NOT modified.")


if __name__ == "__main__":
    main()
