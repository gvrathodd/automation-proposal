
from __future__ import annotations
from pathlib import Path
import pandas as pd
from la_common import OUT, excel_write

SEG=OUT/"step3a1_la_authoritative_segment_evidence_v1.csv"
E2=OUT/"step3e2_la_prototype_validation_v1.xlsx"
F3=OUT/"step3f_la_actual_variant_validation_v1.xlsx"
OUTPUT=OUT/"step3g_la_value_quantification_v1.xlsx"

def main():
    seg=pd.read_csv(SEG); e2=pd.read_excel(E2,sheet_name="Execution_Results"); f3=pd.read_excel(F3,sheet_name="Execution_Results")
    total_seconds=seg["duration_seconds"].sum()
    summary=pd.DataFrame([
        ["LA segments",len(seg)],
        ["Observed segmented time (s)",round(float(total_seconds),3)],
        ["Observed segmented time (min)",round(float(total_seconds)/60,3)],
        ["Time share from final Step-2",24.18],
        ["Median segment duration (s)",round(float(seg["duration_seconds"].median()),3)],
        ["Prototype validation pass rate",round(100*(e2.status=="PASS").mean(),2) if not e2.empty else 0],
        ["Variant validation pass rate",round(100*(f3.status=="PASS").mean(),2) if not f3.empty else 0],
        ["Production savings estimate","NOT ESTIMABLE from current evidence"],
        ["Reason","No comparable manual-vs-production automation timing study"],
    ],columns=["metric","value"])
    notes=pd.DataFrame([
        ["Use of prototype runtime","Do not multiply local mock runtime by historical workload."],
        ["Observed workload","LA represents 24.18% of all segmented Dataset-B time."],
        ["Impact boundary","Mechanical preparation may be accelerated; business review/decision remains human."],
    ],columns=["topic","interpretation"])
    excel_write(OUTPUT,{"Observed_Value":summary,"Interpretation":notes})
    print(OUTPUT)
if __name__=="__main__": main()
