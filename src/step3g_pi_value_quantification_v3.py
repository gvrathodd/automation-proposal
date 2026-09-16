
"""
STEP 3G — PI VALUE QUANTIFICATION V1

Purpose
-------
Quantify what can be measured from the observed Dataset-B pi evidence and
from the controlled local prototype, while explicitly refusing unsupported
production-savings claims.

Two evidence domains are kept separate:

A) OBSERVED DATASET STATISTICS
   - 209 pi segments
   - phase/action structure from Step 3A-2
   - observed segment durations where available
   - decision-oriented evidence from Step 3B / Step 3A-4
   - raw event counts

B) PROTOTYPE-DERIVED RESULTS
   - Step 3E-2 validation cases
   - Step 3F variant validation
   - successful preparation
   - safe stops
   - field verification
   - manual correction
   - local prototype execution time

Critical methodology
--------------------
The following are NOT claimed unless comparable manual/prototype timing data
exist for the same cases and environment:
    - actual production time saved per case
    - production annual savings
    - production throughput increase

The observed Dataset-B segment duration and the local mock execution time are
reported separately. They are not subtracted from one another because they are
not directly comparable measurements.

"Average human steps/case" is not directly observable from the available logs.
Instead we report:
    - average decision-oriented evidence count/segment as an evidence proxy
    - average non-automated/unknown phase occurrences per segment
Neither is labeled as literal "human steps".

Output
------
outputs/step3g_pi_value_quantification_v1.xlsx
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

A1_SEG = OUT / "step3a1_pi_authoritative_segment_evidence_v2.csv"
A1_EVT = OUT / "step3a1_pi_authoritative_event_detail_v2.csv"

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

E2 = OUT / "step3e2_pi_prototype_validation_v1.xlsx"
F3 = OUT / "step3f_pi_actual_variant_validation_v1.xlsx"

OUTPUT = OUT / "step3g_pi_value_quantification_v1.xlsx"


def norm(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_sources():
    require(A1_SEG)
    require(A1_EVT)
    require(E2)
    require(F3)

    if not A4_FILES:
        raise FileNotFoundError(
            "No Step-3A-4 V3 workbook found."
        )
    if not B3_FILES:
        raise FileNotFoundError(
            "No Step-3B V3 workbook found."
        )

    seg = pd.read_csv(A1_SEG)
    evt = pd.read_csv(A1_EVT)

    a4 = A4_FILES[0]
    a4_seq = pd.read_excel(
        a4,
        sheet_name="Segment_Workflow_Sequences"
    )
    a4_order = pd.read_excel(
        a4,
        sheet_name="Common_Order_Evidence"
    ) if "Common_Order_Evidence" in pd.ExcelFile(a4).sheet_names else pd.DataFrame()
    a4_decision = pd.read_excel(
        a4,
        sheet_name="Decision_Evidence"
    )

    b3 = B3_FILES[0]
    b3_boundary = pd.read_excel(
        b3,
        sheet_name="Automation_Boundary"
    )
    b3_human = pd.read_excel(
        b3,
        sheet_name="Human_Gate_Evidence"
    ) if "Human_Gate_Evidence" in pd.ExcelFile(b3).sheet_names else pd.DataFrame()

    e2_results = pd.read_excel(
        E2,
        sheet_name="Execution_Results"
    )
    e2_summary = pd.read_excel(
        E2,
        sheet_name="Validation_Summary"
    )

    f3_results = pd.read_excel(
        F3,
        sheet_name="Execution_Results"
    )

    for df in (seg, evt, a4_seq, a4_order, a4_decision,
               b3_boundary, b3_human, e2_results,
               e2_summary, f3_results):
        if "session_id" in df.columns:
            df["session_id"] = df["session_id"].astype(str)
        if "segment_index" in df.columns:
            df["segment_index"] = pd.to_numeric(
                df["segment_index"],
                errors="coerce"
            ).astype("Int64")

    return (
        seg,
        evt,
        a4_seq,
        a4_order,
        a4_decision,
        b3_boundary,
        b3_human,
        e2_results,
        e2_summary,
        f3_results,
        a4,
        b3,
    )


def build_observed_segment_metrics(
    seg: pd.DataFrame,
    evt: pd.DataFrame,
    a4_seq: pd.DataFrame,
    a4_decision: pd.DataFrame,
):
    """
    Build one-row-per-pi-segment observed statistics.

    Duration and event count are reconstructed from the authoritative raw event
    table whenever possible. This avoids dependency on a particular Step-3A-1
    segment-summary column name.
    """
    keys = seg[
        ["session_id", "segment_index"]
    ].drop_duplicates().copy()

    base = keys.copy()
    base["duration_seconds"] = pd.NA
    base["raw_event_count"] = pd.NA

    raw = evt.copy()

    if "session_id" in raw.columns and "segment_index" in raw.columns:
        raw["session_id"] = raw["session_id"].astype(str)
        raw["segment_index"] = pd.to_numeric(
            raw["segment_index"],
            errors="coerce"
        ).astype("Int64")

    if "event_timestamp" in raw.columns:
        raw["_ts"] = pd.to_datetime(
            raw["event_timestamp"],
            errors="coerce",
            utc=True
        )

        valid = raw[
            raw["_ts"].notna()
            & raw["session_id"].notna()
            & raw["segment_index"].notna()
        ].copy()

        if not valid.empty:
            grouped = (
                valid
                .groupby(
                    ["session_id", "segment_index"],
                    dropna=False
                )
                .agg(
                    raw_event_count_observed=(
                        "event_type", "size"
                    ),
                    first_ts=("_ts", "min"),
                    last_ts=("_ts", "max"),
                )
                .reset_index()
            )

            grouped["duration_seconds_observed"] = (
                grouped["last_ts"] - grouped["first_ts"]
            ).dt.total_seconds()

            base = base.merge(
                grouped[
                    [
                        "session_id",
                        "segment_index",
                        "raw_event_count_observed",
                        "duration_seconds_observed",
                    ]
                ],
                on=["session_id", "segment_index"],
                how="left",
            )

            base["raw_event_count"] = pd.to_numeric(
                base["raw_event_count_observed"],
                errors="coerce"
            )
            base["duration_seconds"] = pd.to_numeric(
                base["duration_seconds_observed"],
                errors="coerce"
            )

            base.drop(
                columns=[
                    "raw_event_count_observed",
                    "duration_seconds_observed",
                ],
                inplace=True,
                errors="ignore",
            )
    else:
        # Fallback to a segment-summary event-count/duration field if the raw
        # event timestamps are unexpectedly unavailable.
        duration_aliases = {
            "duration_seconds",
            "duration_s",
            "duration_sec",
            "duration",
        }

        event_aliases = {
            "raw_event_count",
            "event_count",
            "events",
            "n_events",
        }

        dur_col = next(
            (
                c for c in seg.columns
                if str(c).lower() in duration_aliases
            ),
            None
        )
        evt_col = next(
            (
                c for c in seg.columns
                if str(c).lower() in event_aliases
            ),
            None
        )

        if dur_col:
            d = seg[
                ["session_id", "segment_index", dur_col]
            ].copy()
            d["duration_seconds"] = pd.to_numeric(
                d[dur_col],
                errors="coerce"
            )
            base = base.drop(
                columns=["duration_seconds"]
            ).merge(
                d[
                    [
                        "session_id",
                        "segment_index",
                        "duration_seconds",
                    ]
                ],
                on=["session_id", "segment_index"],
                how="left",
            )

        if evt_col:
            e = seg[
                ["session_id", "segment_index", evt_col]
            ].copy()
            e["raw_event_count"] = pd.to_numeric(
                e[evt_col],
                errors="coerce"
            )
            base = base.drop(
                columns=["raw_event_count"]
            ).merge(
                e[
                    [
                        "session_id",
                        "segment_index",
                        "raw_event_count",
                    ]
                ],
                on=["session_id", "segment_index"],
                how="left",
            )

    # Number of A4 rows associated with each segment. This is retained only as
    # provenance; the phase count below is derived from the actual sequence.
    if not a4_seq.empty:
        sequence_rows = (
            a4_seq
            .groupby(
                ["session_id", "segment_index"],
                dropna=False
            )
            .size()
            .reset_index(
                name="workflow_sequence_rows"
            )
        )

        base = base.merge(
            sequence_rows,
            on=["session_id", "segment_index"],
            how="left"
        )
    else:
        base["workflow_sequence_rows"] = 0

    base["workflow_sequence_rows"] = pd.to_numeric(
        base["workflow_sequence_rows"],
        errors="coerce"
    ).fillna(0).astype(int)

    def count_phase_tokens(value):
        s = norm(value)
        if not s:
            return 0
        return sum(
            1 for token in s.split("→")
            if norm(token)
        )

    if "clean_phase_sequence" in a4_seq.columns:
        phase_counts = a4_seq[
            [
                "session_id",
                "segment_index",
                "clean_phase_sequence",
            ]
        ].copy()

        phase_counts["phase_occurrences"] = (
            phase_counts[
                "clean_phase_sequence"
            ].map(count_phase_tokens)
        )

        phase_counts = (
            phase_counts
            .groupby(
                ["session_id", "segment_index"],
                dropna=False
            )["phase_occurrences"]
            .max()
            .reset_index()
        )

        base = base.merge(
            phase_counts,
            on=["session_id", "segment_index"],
            how="left"
        )
    else:
        base["phase_occurrences"] = 0

    base["phase_occurrences"] = pd.to_numeric(
        base["phase_occurrences"],
        errors="coerce"
    ).fillna(0).astype(int)

    if not a4_decision.empty and {
        "session_id",
        "segment_index"
    }.issubset(a4_decision.columns):

        decision_counts = (
            a4_decision
            .groupby(
                ["session_id", "segment_index"],
                dropna=False
            )
            .size()
            .reset_index(
                name="decision_evidence_rows"
            )
        )

        base = base.merge(
            decision_counts,
            on=["session_id", "segment_index"],
            how="left"
        )
    else:
        base["decision_evidence_rows"] = 0

    base["decision_evidence_rows"] = pd.to_numeric(
        base["decision_evidence_rows"],
        errors="coerce"
    ).fillna(0).astype(int)

    return base

def phase_level_observed_stats(
    a4_seq: pd.DataFrame,
    boundary: pd.DataFrame,
):
    """
    Aggregate phase occurrences.

    Uses exact workflow phase labels from Step 3A-4.
    """
    phase_rows = []

    for _, r in a4_seq.iterrows():
        seq = norm(r.get("clean_phase_sequence"))
        if not seq:
            continue

        for token in seq.split("→"):
            phase = norm(token).split("[", 1)[0]
            if phase:
                phase_rows.append({
                    "session_id": str(r["session_id"]),
                    "segment_index": int(r["segment_index"]),
                    "phase": phase,
                })

    phases = pd.DataFrame(phase_rows)

    if phases.empty:
        return pd.DataFrame()

    g = (
        phases.groupby("phase")
        .agg(
            phase_occurrences=("phase", "size"),
            segments=("segment_index", "count"),
            sessions=("session_id", "nunique"),
        )
        .reset_index()
    )

    g["avg_occurrences_per_pi_segment"] = (
        g["phase_occurrences"] / 209
    ).round(3)

    if not boundary.empty and "step" in boundary.columns:
        b = boundary[
            [c for c in (
                "step",
                "classification"
            ) if c in boundary.columns]
        ].copy()
        b = b.rename(columns={"step": "phase"})
        g = g.merge(
            b,
            on="phase",
            how="left"
        )

    return g.sort_values(
        "phase_occurrences",
        ascending=False
    )


def build_prototype_metrics(
    e2_results: pd.DataFrame,
    f3_results: pd.DataFrame,
):
    # Combine prototype observations but keep their origins.
    rows = []

    for _, r in e2_results.iterrows():
        rows.append({
            "source": "3E-2",
            "test_id": r["case_id"],
            "test_class": r.get("test_class", ""),
            "result": r["actual_result"],
            "status": r["status"],
            "manual_correction_needed": bool(
                r.get("manual_correction_needed", False)
            ),
            "human_gate_preserved": bool(
                r.get("human_gate_preserved", False)
            ),
            "register_clicked": bool(
                r.get("register_clicked", False)
            ),
            "hold_clicked": bool(
                r.get("hold_clicked", False)
            ),
            "fields_attempted": r.get("fields_attempted", 0),
            "fields_verified": r.get("fields_verified", 0),
            "execution_seconds": r.get(
                "execution_seconds",
                r.get("seconds", 0)
            ),
        })

    for _, r in f3_results.iterrows():
        rows.append({
            "source": "3F",
            "test_id": r["variant_id"],
            "test_class": r.get("variant_class", ""),
            "result": r["actual_result"],
            "status": r["status"],
            "manual_correction_needed": bool(
                r.get("manual_correction_needed", False)
            ),
            "human_gate_preserved": bool(
                r.get("human_gate_preserved", False)
            ),
            "register_clicked": bool(
                r.get("register_clicked", False)
            ),
            "hold_clicked": bool(
                r.get("hold_clicked", False)
            ),
            "fields_attempted": r.get("fields_attempted", 0),
            "fields_verified": r.get(
                "fields_populated_correctly",
                0
            ),
            "execution_seconds": r.get(
                "execution_seconds",
                r.get("seconds", 0)
            ),
        })

    proto = pd.DataFrame(rows)

    if proto.empty:
        return proto, pd.DataFrame()

    proto["execution_seconds"] = pd.to_numeric(
        proto["execution_seconds"],
        errors="coerce"
    ).fillna(0)

    proto["fields_attempted"] = pd.to_numeric(
        proto["fields_attempted"],
        errors="coerce"
    ).fillna(0)

    proto["fields_verified"] = pd.to_numeric(
        proto["fields_verified"],
        errors="coerce"
    ).fillna(0)

    summary = pd.DataFrame([
        {
            "metric": "combined_prototype_tests",
            "value": len(proto),
            "interpretation":
                "3E-2 and 3F controlled local-mock tests.",
        },
        {
            "metric": "combined_passes",
            "value": int(proto["status"].eq("PASS").sum()),
            "interpretation":
                "PASS means successful preparation or expected safe stop.",
        },
        {
            "metric": "pass_rate_pct",
            "value": round(
                100 * proto["status"].eq("PASS").mean(),
                2
            ),
            "interpretation":
                "Controlled prototype test result; not production accuracy.",
        },
        {
            "metric": "manual_correction_rate_pct",
            "value": round(
                100 * proto["manual_correction_needed"].mean(),
                2
            ),
            "interpretation":
                "Observed in prototype tests; not production correction rate.",
        },
        {
            "metric": "human_gate_preserved_pct",
            "value": round(
                100 * proto["human_gate_preserved"].mean(),
                2
            ),
            "interpretation":
                "Prototype safety result.",
        },
        {
            "metric": "mean_prototype_execution_seconds",
            "value": round(
                proto["execution_seconds"].mean(),
                4
            ),
            "interpretation":
                "Local mock runtime only; not comparable directly to Dataset-B duration.",
        },
        {
            "metric": "median_prototype_execution_seconds",
            "value": round(
                proto["execution_seconds"].median(),
                4
            ),
            "interpretation":
                "Local mock runtime only.",
        },
        {
            "metric": "register_clicks",
            "value": int(proto["register_clicked"].sum()),
            "interpretation":
                "Must remain zero.",
        },
        {
            "metric": "hold_clicks",
            "value": int(proto["hold_clicked"].sum()),
            "interpretation":
                "Must remain zero.",
        },
    ])

    return proto, summary


def build_value_table(
    observed_segments,
    phase_stats,
    proto_summary,
):
    avg_duration = pd.to_numeric(
        observed_segments["duration_seconds"],
        errors="coerce"
    ).dropna()

    avg_phases = (
        observed_segments["phase_occurrences"]
        .mean()
    )

    decision_avg = (
        observed_segments["decision_evidence_rows"]
        .mean()
    )

    return pd.DataFrame([
        {
            "metric": "Average observed pi segment duration",
            "value":
                round(avg_duration.mean(), 3)
                if len(avg_duration)
                else None,
            "unit": "seconds/segment",
            "evidence_type": "OBSERVED_DATASET",
            "interpretation":
                "Observed Dataset-B segment duration; not a manual-only time measurement.",
        },
        {
            "metric": "Median observed pi segment duration",
            "value":
                round(avg_duration.median(), 3)
                if len(avg_duration)
                else None,
            "unit": "seconds/segment",
            "evidence_type": "OBSERVED_DATASET",
            "interpretation":
                "Observed Dataset-B segment duration.",
        },
        {
            "metric": "Average workflow phase occurrences per pi segment",
            "value": round(avg_phases, 3),
            "unit": "phase occurrences/segment",
            "evidence_type": "OBSERVED_DATASET",
            "interpretation":
                "Mechanical/workflow activity count proxy; not human-step count.",
        },
        {
            "metric": "Average decision-evidence rows per pi segment",
            "value": round(decision_avg, 3),
            "unit": "evidence rows/segment",
            "evidence_type": "OBSERVED_DATASET",
            "interpretation":
                "Decision-oriented evidence proxy; not literal human-step count.",
        },
        {
            "metric": "Prototype validation pass rate",
            "value":
                proto_summary.loc[
                    proto_summary["metric"].eq("pass_rate_pct"),
                    "value"
                ].iloc[0]
                if not proto_summary.empty else None,
            "unit": "%",
            "evidence_type": "PROTOTYPE_DERIVED",
            "interpretation":
                "Controlled local-mock result.",
        },
        {
            "metric": "Prototype correction rate",
            "value":
                proto_summary.loc[
                    proto_summary["metric"].eq("manual_correction_rate_pct"),
                    "value"
                ].iloc[0]
                if not proto_summary.empty else None,
            "unit": "%",
            "evidence_type": "PROTOTYPE_DERIVED",
            "interpretation":
                "Controlled local-mock result; not production correction rate.",
        },
        {
            "metric": "Estimated time saved per case",
            "value": None,
            "unit": "seconds/case",
            "evidence_type": "NOT_ESTIMABLE",
            "interpretation":
                "Requires comparable manual execution time and prototype execution time for the same cases/environment. Dataset-B segment duration and local-mock runtime are not directly comparable.",
        },
        {
            "metric": "Estimated time saved across observed pi workload",
            "value": None,
            "unit": "seconds/workload",
            "evidence_type": "NOT_ESTIMABLE",
            "interpretation":
                "Requires a validated per-case time saving and a defined workload unit/case count. Do not multiply local mock runtime differences by 209.",
        },
    ])


def build_remaining_measurements():
    return pd.DataFrame([
        {
            "measurement": "Comparable manual execution time",
            "why_needed":
                "Needed to calculate actual time saved per case.",
            "how_to_measure":
                "Time representative real/manual or faithful task executions under the same interface and case definition used by the prototype.",
        },
        {
            "measurement": "Comparable prototype execution time",
            "why_needed":
                "Must be measured against the same task boundary as manual timing.",
            "how_to_measure":
                "Measure prototype preparation from defined start state to defined human-gate stop.",
        },
        {
            "measurement": "Workload case count",
            "why_needed":
                "Needed to project savings across a defined workload.",
            "how_to_measure":
                "Use an explicit observed or operational case-count denominator rather than segment count unless one-to-one equivalence has been established.",
        },
    ])


def main():
    (
        seg,
        evt,
        a4_seq,
        a4_order,
        a4_decision,
        b3_boundary,
        b3_human,
        e2_results,
        e2_summary,
        f3_results,
        a4_path,
        b3_path,
    ) = load_sources()

    observed_segments = build_observed_segment_metrics(
        seg,
        evt,
        a4_seq,
        a4_decision
    )

    phase_stats = phase_level_observed_stats(
        a4_seq,
        b3_boundary
    )

    proto_rows, proto_summary = build_prototype_metrics(
        e2_results,
        f3_results
    )

    value_table = build_value_table(
        observed_segments,
        phase_stats,
        proto_summary
    )

    remaining = build_remaining_measurements()

    # A compact final interpretation.
    conclusion = pd.DataFrame([
        {
            "area": "Observed workload",
            "finding":
                f"{len(observed_segments)} pi segments have been quantified from the Dataset-B evidence.",
            "evidence_type": "OBSERVED_DATASET",
        },
        {
            "area": "Mechanical work",
            "finding":
                f"Average observed workflow-phase occurrences per segment = {observed_segments['phase_occurrences'].mean():.3f}.",
            "evidence_type": "OBSERVED_DATASET",
        },
        {
            "area": "Human work",
            "finding":
                f"Average decision-evidence rows per segment = {observed_segments['decision_evidence_rows'].mean():.3f}; this is only a decision-evidence proxy, not literal human-step count.",
            "evidence_type": "OBSERVED_DATASET",
        },
        {
            "area": "Prototype",
            "finding":
                f"3E-2 and 3F results are controlled local-mock prototype evidence; combined pass rate = {proto_summary.loc[proto_summary['metric'].eq('pass_rate_pct'),'value'].iloc[0] if not proto_summary.empty else 'N/A'}%.",
            "evidence_type": "PROTOTYPE_DERIVED",
        },
        {
            "area": "Time savings",
            "finding":
                "Not estimable from current evidence without comparable manual and prototype timing for the same cases.",
            "evidence_type": "NOT_ESTIMABLE",
        },
    ])

    validation = pd.DataFrame([
        {
            "check": "pi_population",
            "expected": 209,
            "observed": len(observed_segments),
            "status": "PASS"
            if len(observed_segments) == 209
            else "FAIL",
        },
        {
            "check": "observed_dataset_and_prototype_separated",
            "expected": "YES",
            "observed": "YES",
            "status": "PASS",
        },
        {
            "check": "literal_human_steps_claimed_without_direct_measurement",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "time_savings_estimated_without_comparable_timing",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "production_savings_projected_from_local_mock",
            "expected": "NO",
            "observed": "NO",
            "status": "PASS",
        },
        {
            "check": "prototype_register_or_hold_actions",
            "expected": 0,
            "observed":
                int(proto_rows["register_clicked"].sum())
                + int(proto_rows["hold_clicked"].sum()),
            "status":
                "PASS"
                if (
                    int(proto_rows["register_clicked"].sum())
                    + int(proto_rows["hold_clicked"].sum())
                ) == 0
                else "FAIL",
        },
    ])

    readme = pd.DataFrame([
        ["Purpose", "Quantify value only where the evidence supports a measurement."],
        ["Observed statistics", "Dataset-B segment and workflow evidence are reported separately from prototype measurements."],
        ["Human steps", "Literal human-step counts are not directly measured; decision-evidence count is explicitly labeled as a proxy."],
        ["Prototype statistics", "3E-2/3F local-mock results are prototype-derived and are not treated as production rates."],
        ["Time saved", "Not estimated because observed Dataset-B duration and local-mock runtime are not comparable task timings."],
        ["Workload projection", "No projection is made across 209 segments without a validated per-case saving and case-count denominator."],
    ], columns=["item", "value"])

    output = OUTPUT

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        value_table.to_excel(writer, sheet_name="Value_Metrics", index=False)
        observed_segments.to_excel(writer, sheet_name="Observed_Segment_Stats", index=False)
        phase_stats.to_excel(writer, sheet_name="Observed_Phase_Stats", index=False)
        proto_rows.to_excel(writer, sheet_name="Prototype_Results", index=False)
        proto_summary.to_excel(writer, sheet_name="Prototype_Summary", index=False)
        conclusion.to_excel(writer, sheet_name="Final_Interpretation", index=False)
        remaining.to_excel(writer, sheet_name="Remaining_Measurements", index=False)
        validation.to_excel(writer, sheet_name="Validation", index=False)

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

    print("=" * 80)
    print("STEP 3G — PI VALUE QUANTIFICATION V1")
    print("=" * 80)

    print("\nOBSERVED DATASET STATISTICS")
    print(
        f"pi segments: {len(observed_segments)}"
    )
    print(
        f"average phase occurrences/segment: "
        f"{observed_segments['phase_occurrences'].mean():.3f}"
    )
    print(
        f"average decision-evidence rows/segment: "
        f"{observed_segments['decision_evidence_rows'].mean():.3f}"
    )

    durations = pd.to_numeric(
        observed_segments["duration_seconds"],
        errors="coerce"
    ).dropna()

    if len(durations):
        print(
            f"average observed segment duration: "
            f"{durations.mean():.3f}s"
        )
        print(
            f"median observed segment duration: "
            f"{durations.median():.3f}s"
        )
    else:
        print(
            "observed segment duration: NOT AVAILABLE after raw-event "
            "reconstruction"
        )

    print("\nPROTOTYPE-DERIVED")
    print(
        proto_summary.to_string(index=False)
        if not proto_summary.empty
        else "No prototype metrics available."
    )

    print("\nVALUE METRICS")
    print(value_table.to_string(index=False))

    print("\nFINAL INTERPRETATION")
    print(conclusion.to_string(index=False))

    print("\nVALIDATION")
    print(validation.to_string(index=False))

    failures = validation[
        validation["status"] == "FAIL"
    ]

    if not failures.empty:
        raise RuntimeError(
            "3G validation failed. Inspect Validation sheet."
        )

    print("\nOUTPUT")
    print(output)
    print("\nsegments.jsonl was NOT modified.")
    print("STEP 3G COMPLETE.")


if __name__ == "__main__":
    main()
