
from __future__ import annotations
import pandas as pd
from collections import Counter
from la_common import OUT, load_b_events, load_la_segments, assign_events, event_type, routes_from, active_app, get_field, excel_write

A1 = OUT / "step3a1_la_authoritative_event_detail_v1.csv"
OUTPUT = OUT / "step3d_la_technical_interface_discovery_v1.xlsx"

KNOWN_ROUTE = "/leave-applications"

def main():
    ev = pd.read_csv(A1)
    routes=Counter(); apps=Counter(); types=Counter()
    for _, r in ev.iterrows():
        for rt in str(r.get("routes","")).split(" | "):
            if KNOWN_ROUTE in rt.lower():
                routes[KNOWN_ROUTE]+=1
        if str(r.get("active_app")):
            apps[str(r.get("active_app"))]+=1
        types[str(r.get("event_type"))]+=1

    route_df = pd.DataFrame([{"route":k,"event_rows":v} for k,v in routes.items()])
    event_df = pd.DataFrame([{"event_type":k,"rows":v} for k,v in types.items()]).sort_values("rows",ascending=False)

    observed_targets = ev.loc[ev["target_field"].fillna("").astype(str)!="","target_field"].value_counts().reset_index()
    observed_targets.columns=["target_field","rows"]

    assumptions = pd.DataFrame([
        ["Production implementation","UNKNOWN","Real /leave-applications executable implementation was not available to the analysis repository."],
        ["Authentication","UNKNOWN","Must be validated in pilot."],
        ["Permissions","UNKNOWN","Must be validated for employee attendance/leave data."],
        ["API availability","UNKNOWN","No API contract inferred from logs."],
        ["Selector stability","VALIDATE","Historical DOM/target metadata is evidence, not a production selector guarantee."],
        ["Business-rule logic","UNKNOWN","Human decision remains outside automation."],
    ], columns=["technical_question","status","evidence_note"])

    proto = pd.DataFrame([
        ["route",KNOWN_ROUTE],
        ["target_core","employee_id, employee_name, request_type, department, attendance, leave_type, leave_start, leave_end, status, processing_comment"],
        ["decision_gate","Approve / Return for correction / Hold remain human-controlled"],
        ["prototype_type","Controlled local reconstruction; not production integration"],
    ], columns=["item","value"])

    excel_write(OUTPUT, {"README":pd.DataFrame([["Family","LA"],["Route",KNOWN_ROUTE],["Population",len(ev)]],columns=["item","value"]),
                         "Browser_Routes":route_df,"Input_Transfer_Interface":event_df,
                         "Observed_Target_Fields":observed_targets,"Integration_Assumptions":assumptions,
                         "Prototype_Interface_Spec":proto})
    print(OUTPUT)

if __name__ == "__main__":
    main()
