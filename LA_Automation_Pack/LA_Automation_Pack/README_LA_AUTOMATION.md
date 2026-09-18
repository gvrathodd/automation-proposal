# LA Automation Pack

This pack is the LA analogue of the PI Step-3 rebuild.

LA = the existing Dataset-B `la` family:
- 99 inferred work units
- 24.18% of observed segmented time
- 14 observed sessions in the family-level inventory
- `/leave-applications` is the dominant observed route
- direct OCR evidence includes attendance / leave / paid-leave terminology

## Files

`src/` contains the evidence, workflow, automation-boundary, prototype-validation, replay, and value-quantification scripts.

`prototype/` contains:
- `la_leave_applications.html`
- `la_test_cases.json`
- `la_demo_runner.py`
- `la_demo_runner_live.py`

## Run order

From the repository root:

```bash
python src/step3a1_authoritative_la_evidence_reconstruction_v1.py
python src/step3a2_la_chronological_workflow_reconstruction_v1.py
python src/step3a3_la_ocr_semantic_evidence_v1.py
python src/step3a4_la_workflow_map_v1.py
python src/step3b_la_automation_boundary_v1.py
python src/step3c_la_parameterized_vs_branch_analysis_v1.py
python src/step3d_la_technical_interface_discovery_v1.py
python src/step3e0_application_target_discovery_v1.py
python src/step3e1_la_local_prototype_v1.py
python src/step3e2_la_prototype_validation_v1.py
python src/step3e3_la_final_prototype_assessment_v1.py
python src/step3f_la_prototype_variant_validation_v1.py
python src/step3g_la_value_quantification_v1.py
python src/step3h_la_real_dataset_replay_v1.py
python src/step3i_la_real_data_extraction_similarity_v1.py
```

Or run the full sequence:

```bash
python src/la_run_pipeline.py
```

## Prototype

Install:

```bash
pip install playwright pandas openpyxl
playwright install chromium
```

Visible demo:

```bash
python prototype/la_demo_runner_live.py
```

Headless validation:

```bash
python prototype/la_demo_runner.py
```

## Safety boundary

The prototype prepares attendance/leave data and stops at the human decision gate.

It never clicks:
- Approve
- Return for correction
- Hold

Missing or invalid inputs are safe-stop conditions.

## Important limitation

The production `/leave-applications` application was not available as an executable target in the project evidence. The prototype is therefore a controlled local reconstruction, not production integration.

No production savings or production accuracy are claimed.
