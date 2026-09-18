
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=[
"step3a1_authoritative_la_evidence_reconstruction_v1.py",
"step3a2_la_chronological_workflow_reconstruction_v1.py",
"step3a3_la_ocr_semantic_evidence_v1.py",
"step3a4_la_workflow_map_v1.py",
"step3b_la_automation_boundary_v1.py",
"step3c_la_parameterized_vs_branch_analysis_v1.py",
"step3d_la_technical_interface_discovery_v1.py",
"step3e0_application_target_discovery_v1.py",
"step3e1_la_local_prototype_v1.py",
"step3e2_la_prototype_validation_v1.py",
"step3e3_la_final_prototype_assessment_v1.py",
"step3f_la_prototype_variant_validation_v1.py",
"step3g_la_value_quantification_v1.py",
"step3h_la_real_dataset_replay_v1.py",
"step3i_la_real_data_extraction_similarity_v1.py",
]
for name in SCRIPTS:
    print("\n===",name,"===")
    subprocess.run([sys.executable,str(ROOT/"src"/name)],check=True)
