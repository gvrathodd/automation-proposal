# From Operation Logs to an Automation Proposal

## Author

**Name:** Gaurav Girish Rathod  
**University:** IIT Mandi  
**Program:** B.Tech in Civil Engineering  
**Email:** [b24076@students.iitmandi.ac.in](mailto:b24076@students.iitmandi.ac.in)

A seven-day FDE-style investigation that turns raw desktop-operation logs into:

1. coherent work units,
2. evidence-supported process families,
3. an automation candidate decision, and
4. a working human-in-the-loop automation prototype.

The project uses **Dataset A** as the labelled development/evaluation dataset and **Dataset B** as the unlabelled production-style dataset.

## Automation Demo

**Live automation demo:**  
https://youtu.be/LL4OfRidPTM

The demo shows the PI automation prototype preparing structured data on a reconstructed local interface while preserving the human decision gate.

---

## Project Objective

The raw logs contain low-level desktop activity such as keyboard events, mouse clicks, application switches, browser interactions, clipboard changes, and screenshots. They do not directly provide business-process boundaries.

The project therefore follows this sequence:

```text
Raw operation logs
        |
        v
Session reconstruction
        |
        v
Boundary discovery on Dataset A
        |
        v
Validated segmentation model
        |
        v
540 inferred Dataset-B work units
        |
        v
Process-family / semantic analysis
        |
        v
Automation candidate analysis
        |
        v
PI workflow reconstruction
        |
        v
Human-in-the-loop automation prototype
        |
        v
Validation + real Dataset-B replay + risk analysis
```

The core design principle is:

> **Automate mechanical preparation; keep business decisions human-controlled.**

---

# 1. Dataset Overview

The complete raw-data inventory identified:

| Dataset | Sessions | Events | Event Files | Ground Truth |
|---|---:|---:|---:|---|
| Dataset A | 63 | 162,768 | 117 | Yes |
| Dataset B | 15 | 20,477 | 20 | No |
| Total | 78 | 183,245 | 137 | A only |

Additional integrity findings:

- 39,309 screenshots were present in the original inventory.
- No duplicate event IDs were found.
- Screenshot references were resolved without missing references in the final integrity check.
- Dataset A contained 53 multi-chunk sessions.
- Dataset B contained 5 multi-chunk sessions.
- Physical chunk-folder naming was not always identical to `correlation.chunk_id`; the raw data was not modified.
- Timestamp order was used as the authoritative chronological order when sequence metadata disagreed.

Raw data is intentionally kept outside Git and is ignored through:

```text
raw_data/
```

Do not commit raw operational logs, screenshots, credentials, or other sensitive source data.

---

# 2. Step 1 — Recover Coherent Work Units

## 2.1 Session Reconstruction

A single recording chunk is **not** treated as a business task boundary.

Sessions can contain:

- multiple recording chunks,
- application switches,
- temporary departures from a task,
- returns to a previous workflow,
- repeated process executions.

A reusable loader was therefore built to:

1. discover event files recursively,
2. identify the dataset from the path,
3. load events,
4. group them by `session_id`,
5. normalize missing context fields,
6. sort chronologically,
7. preserve chunk metadata,
8. retain source-file references.

A loader bug caused failures when `context.active_app` was `null`. The loader was corrected to tolerate missing application context.

---

## 2.2 Ground Truth Extraction

Dataset A ground truth contains process metadata including process code/name, variant, case identifiers, start/end timestamps, and execution information.

A total of **257 executions were missing end timestamps**. These were retained as incomplete GT records rather than assigning fabricated end timestamps.

Boundary events extracted from GT included:

- `process_started`
- `process_switched_out`
- `process_suspended`
- `process_resumed`

Counts:

| Boundary Event | Count |
|---|---:|
| Process started | 1,819 |
| Process switched out | 1,590 |
| Process resumed | 190 |
| Process suspended | 99 |
| Total raw boundary events | 3,698 |
| Unique `(session_id, timestamp)` boundaries | 3,583 |

Duplicate timestamps were collapsed only for evaluation.

---

## 2.3 Boundary-Signal Analysis

True-boundary windows were compared with random control windows.

The strongest raw signals were:

| Signal | Relative boundary-window rate |
|---|---:|
| Application switch | ~2.26× |
| Browser navigation | ~2.18× |
| Browser click | ~1.47× |
| Mouse click | ~1.44× |

The conclusion was that application/context transitions contain real boundary information, but raw activity volume alone cannot define a business boundary.

---

## 2.4 Baseline V1

A lightweight heuristic was built using two-second bins and weighted activity signals such as:

- application switches,
- browser navigation,
- browser clicks,
- mouse clicks,
- window changes.

Result:

| Metric | Result |
|---|---:|
| GT boundaries | 3,583 |
| Predicted boundaries | 2,770 |
| Precision | 0.5560 |
| Recall | 0.4298 |
| F1 | 0.4848 |
| Mean boundary error | 1.50 s |

The main failure mode was a combination of false positives and missed genuine process transitions.

---

## 2.5 Baseline V2 — Before/After Context

A second heuristic compared event activity immediately before and after candidate timestamps.

Features included:

- before/after event counts,
- application switches,
- browser navigation,
- browser clicks,
- mouse clicks,
- window changes,
- unique applications,
- keyboard activity.

Result:

| Metric | Result |
|---|---:|
| Candidate points | 45,782 |
| Predicted boundaries | 4,557 |
| GT boundaries | 3,583 |
| Precision | 0.4215 |
| Recall | 0.5361 |
| F1 | 0.4720 |
| Mean error | 2.29 s |

Recall increased, but over-segmentation became worse. This justified moving to supervised ML.

---

## 2.6 Supervised Boundary Model

A candidate dataset of **45,782 timestamps** was created from Dataset A.

A candidate was considered positive when it fell within **±5 seconds** of a GT boundary.

Class balance:

- Boundary / positive: **22.31%**
- Non-boundary / negative: **77.69%**

To avoid session leakage, splitting was done by whole session:

- 44 training sessions
- 9 validation sessions
- 10 test sessions

Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting

HistGradientBoosting performed best at the candidate-classification layer:

| Metric | Test |
|---|---:|
| Precision | 0.8864 |
| Recall | 0.9246 |
| F1 | 0.9051 |
| PR-AUC | 0.9652 |

Important interpretation:

> **Candidate-level F1 is not the same as final segmentation F1.**

The classifier can identify a region around a transition while producing several nearby positive candidates for the same underlying business boundary.

This separated the problem into:

**A. Boundary-region detection**

and

**B. Boundary timestamp consolidation**

This distinction became one of the main findings of Step 1.

---

## 2.7 Boundary Consolidation

Using the HGB predictions:

- raw positive candidates: **1,646**
- consolidated boundaries: **364**
- test GT boundaries: **536**
- precision: **0.8516**
- recall: **0.5784**
- F1: **0.6889**
- mean boundary error: **1.10 s**

A diagnostic showed:

> **533 / 536 = 99.44%** of test GT boundaries had a nearby positive classifier candidate within ±5 seconds.

Therefore the principal bottleneck was not boundary-region detection; it was selecting the correct number and location of boundaries during consolidation.

A merge-gap sweep was performed and a **2-second merge gap** was retained as the practical configuration because it preserved more distinct boundaries without materially sacrificing precision.

---

## 2.8 Raw-JSON Rebuild and Capacity Experiment

The ML pipeline was rebuilt directly from Dataset A raw JSON rather than relying only on a previously materialized feature table.

This produced approximately **46,160 candidate rows** while preserving the 44/9/10 session-level split.

The number of HGB boosting iterations was also tested:

| `max_iter` | Validation F1 |
|---:|---:|
| 200 | ~0.6975 |
| 300 | ~0.6861 |
| 500 | ~0.6861 |
| 800 | ~0.6844 |

Increasing model capacity did not solve the segmentation bottleneck.

The final raw-derived evaluation was approximately:

| Tolerance | F1 |
|---|---:|
| ±1 s | 0.602 |
| ±2 s | 0.684 |
| ±3 s | 0.688 |
| ±5 s | 0.691 |

At ±5 seconds:

- precision: ~0.973
- recall: ~0.535
- mean error: ~0.60 s

A local-peak consolidation experiment achieved approximately:

- precision: 0.9783
- recall: 0.5388
- F1: 0.6949
- mean error: 0.646 s

It was not adopted because it did not materially improve the validated pipeline.

---

## 2.9 Final Step-1 Configuration

The frozen configuration was:

```text
candidate spacing:       2 seconds
context window:          ±6 seconds
model:                   HistGradientBoosting
max_iter:                200
learning_rate:           0.08
max_leaf_nodes:          31
classification threshold: 0.20
merge gap:               2 seconds
```

No claim of perfect business-boundary recovery is made.

---

# 3. Dataset-B Segmentation

The final Step-1 segmentation was applied to Dataset B.

Dataset B:

- 15 sessions
- 20,477 events
- 5,294 candidate points
- 772 positive candidates
- 525 predicted boundary points
- **540 inferred work units**

Because Dataset B has **no ground truth**, no B-side F1 score is claimed.

## Critical deliverable

The authoritative:

```text
segments.jsonl
```

represents **all 540 inferred Dataset-B work units**.

It is **not** a PI-only file.

Downstream PI analysis uses a subset of those 540 work units:

```text
Dataset B
   |
   +--> 540 total inferred work units
             |
             +--> PI family = 209
             |
             +--> LA = 99
             +--> OB = 77
             +--> SI = 68
             +--> RT = 55
             +--> Other = 32
```

Expected JSONL record structure:

```json
{
  "session_id": "...",
  "start": "2026-...Z",
  "end": "2026-...Z",
  "label": "..."
}
```

---

# 4. Step 2 — Process Discovery

Step 1 answers:

> **Where are the work units?**

Step 2 answers:

> **What do those work units represent, and which are plausible automation candidates?**

The 540 segments were treated as **inferred work units**, not as 540 confirmed unique business processes.

---

## 4.1 Segment Inventory

Each segment was described using evidence such as:

- duration,
- raw event count,
- event types,
- application sequence,
- application transitions,
- browser route/path,
- navigation,
- browser clicks,
- form inputs,
- keyboard activity,
- clipboard activity,
- documents,
- screenshots,
- cross-application behavior,
- page classification.

---

## 4.2 Application Patterns and False Leads

Repeated application sequences such as:

```text
Edge → Word → Edge
Edge → Notepad → Edge
```

were investigated.

They were useful behavioral evidence but were **not** treated as business-process identities.

The same application transition can occur in different business contexts.

Therefore:

> **Application sequence ≠ business process**

Semantic and structural evidence was required.

---

## 4.3 Screenshot Evidence Strategy

A first screenshot selector was built using:

- transitions,
- interaction events,
- temporal position,
- application diversity.

The first selector over-relied on temporal position and was rejected.

A second selector prioritized:

1. application transitions,
2. meaningful interactions,
3. application diversity,
4. temporal position only as fallback.

It produced approximately **2,199 representative screenshots**, with approximately **99.6%** selected using event/transition evidence.

However, even that was not treated as the final semantic evidence source.

For semantic interpretation, all screenshots associated with a complete segment were preserved.

---

## 4.4 Representative Manual Review

Four provisional behavioral families were selected for manual inspection:

```text
PF004
PF005
PF008
PF009
```

Three executions from each family were selected:

- 12 representative segments
- multiple sessions
- repeated behavior
- cross-application activity
- clipboard/form activity
- screenshot evidence

The review focused on:

- business identity,
- consistency,
- variants,
- repeating actions,
- human judgement,
- automation feasibility,
- operational risk.

For these 12 segments:

- 571 raw events
- 137 screenshot references
- 137 resolved
- 0 missing
- 0 ambiguous

Screenshot resolution followed:

```text
session
  -> chunk
  -> events.jsonl
  -> segment time range
  -> screenshot reference
  -> originating screenshot file
```

Identical screenshot timestamps were explicitly preserved.

---

# 5. OCR and Semantic Interpretation

The raw environment contained Japanese business-process screens. Local OCR was introduced because low-level event metadata did not fully describe the business semantics.

A dedicated `.venv_vision` environment was used so that OCR dependencies did not disturb the primary environment.

Installed/tested components included:

- PaddlePaddle
- PaddleOCR
- PyTorch
- Transformers
- SentencePiece

A five-screenshot test recovered business-specific labels including:

- 経費精算・給与変更
- 扶養控除変更
- 役職手当新設
- 年次有給休暇
- 半日有給申請
- 特別休暇（慶弔）

The initial OCR test produced high mean confidence, approximately **0.9765–0.9929**.

---

## 5.1 What OCR Revealed

Screenshots exposed information such as:

- employee IDs,
- employee names,
- request types,
- amounts,
- statuses,
- processing comments,
- structured business pages.

Example fields included:

```text
社員 ID
氏名
区分
ステータス
処理コメント
登録確定
保留
```

This demonstrated that the logs contained structured business-processing screens rather than only generic desktop activity.

---

## 5.2 Translation Experiment

A local Japanese-to-English translation model was tested.

Directly translating large raw OCR dumps failed because the input mixed:

- UI labels,
- employee IDs,
- names,
- URLs,
- numbers,
- statuses,
- browser chrome,
- OCR noise.

The outputs were not reliable enough to be used as evidence.

The pipeline therefore changed from:

```text
raw OCR -> full-document translation
```

to:

```text
screenshot
 -> OCR regions
 -> noise filtering
 -> relevant Japanese labels
 -> targeted terminology
 -> structured UI interpretation
 -> business-process interpretation
```

The analytical glossary included terms such as:

| Japanese | Interpretation |
|---|---|
| 申請 | Application |
| 承認 | Approval |
| 処理待ち | Pending processing |
| 登録確定 | Confirm/finalize registration |
| 保留 | Hold |
| 社員 ID | Employee ID |
| 金額 | Amount |
| 給与変更 | Payroll change |
| 経費精算 | Expense reimbursement |
| 勤怠 | Attendance |
| 休暇申請 | Leave application |

The original Japanese was retained alongside analytical English translation.

---

## 5.3 OCR Coverage Limitation

The larger OCR sample produced:

- 1,113 OCR records
- 1,104 records containing Japanese
- 3,366 unique Japanese OCR lines
- 283,417 Japanese characters
- 0 missing translation lines
- 1,104 records with English translations

Coverage across all 540 segments was only:

**160 / 540 = 29.63%**

Family-level coverage varied significantly.

Therefore OCR-derived subprocess counts were **not extrapolated** to all 540 segments.

---

## 5.4 Semantic Propagation Experiment

A structural propagation method attempted to infer semantics for OCR-uncovered segments using:

- routes,
- DOM fields,
- button families,
- window/document patterns,
- application patterns,
- related structural evidence.

A held-out evaluation was performed.

Result:

- 40 held-out cases
- 17 accepted as high-confidence
- 8 correct
- **47.1% precision** among accepted cases

Propagation was therefore rejected as a full-dataset labelling mechanism.

Unresolved semantics were retained instead of forcing labels.

---

# 6. Final Process Families

The authoritative quantitative layer is the six-family segmentation over all 540 work units:

| Family | Segments | Segment Share | Time Share |
|---|---:|---:|---:|
| PI | 209 | 38.7% | 35.78% |
| LA | 99 | 18.3% | 24.18% |
| OB | 77 | 14.3% | 16.80% |
| SI | 68 | 12.6% | 11.68% |
| RT | 55 | 10.2% | 8.61% |
| Other | 32 | 5.9% | 2.95% |

Observed total segmented time:

**10,574.831 seconds = approximately 176.25 minutes = 2.94 hours**

Approximate family absolute time:

| Family | Absolute Time |
|---|---:|
| PI | 3,783.777 s / 63.06 min |
| LA | 2,557.133 s / 42.62 min |
| OB | 1,776.118 s / 29.60 min |
| SI | 1,235.493 s / 20.59 min |
| RT | 910.264 s / 15.17 min |
| Other | 312.046 s / 5.20 min |

These figures represent **observed segmented workload**, not automation savings.

---

# 7. Mechanical Work vs Human Judgment

Each family was decomposed into two major categories.

## Mechanical / repetitive

```text
read
  -> copy
  -> transfer
  -> transform
  -> enter
  -> prepare documents
  -> navigate
  -> prepare submission
```

## Human judgment

```text
review
  -> policy interpretation
  -> approve / reject
  -> hold
  -> exception handling
  -> ambiguous decisions
  -> final business release
```

A key rule was established:

> A button labelled “Approve”, “Register”, or “Hold” is not evidence that the surrounding workflow is automatically safe.

The preceding context and actual business decision must be evaluated.

---

# 8. Step 2 Candidate Decision

The final candidate decision was intentionally **evidence-based rather than dependent on an opaque composite score**.

Dimensions considered included:

- workload volume,
- observed time,
- repetitive/mechanical work,
- human judgment,
- structured inputs,
- cross-application transfer,
- implementation complexity,
- integration assumptions,
- governance risk,
- prototype feasibility.

## Priority 1 — PI

PI was selected for the first automation pilot because it combines:

- 209 / 540 segments,
- 38.7% segment share,
- 35.78% time share,
- high browser activity,
- high clipboard activity,
- repetitive preparation/transfer behavior,
- identifiable human decision gates,
- a workflow that can be separated into deterministic preparation and human review.

Direct OCR evidence includes PI case types such as:

- expense reimbursement,
- payroll change.

These are treated as **observed branch evidence**, not an exhaustive decomposition of all 209 PI segments.

## Priority 2 — LA

LA remains the next candidate because:

- it has 99 segments,
- 24.18% time share,
- strong clipboard/browser interaction,
- lower decision-point evidence than PI,
- meaningful repetitive transfer activity.

## Deferred

OB, SI, RT, and Other were deferred primarily because of:

- governance sensitivity,
- exception/access complexity,
- lower relative first-build opportunity,
- or insufficient evidence for a safe first implementation.

---

# 9. Step 3 — PI Automation Scope

The selected automation is explicitly a:

> **Human-in-the-loop PI automation assistant**

It is not an autonomous decision system.

## In scope

```text
READ / RETRIEVE
       |
       v
COPY / TRANSFER
       |
       v
TRANSFORM / FORMAT
       |
       v
FORM ENTRY
       |
       v
DOCUMENT PREPARATION
       |
       v
PREDICTABLE NAVIGATION
       |
       v
SUBMISSION PREPARATION
       |
       v
HUMAN REVIEW / DECISION
```

## Out of scope

- approval,
- hold,
- rejection,
- policy interpretation,
- ambiguous cases,
- exception resolution,
- final business release.

---

# 10. Prototype Architecture

The prototype uses a parameterized-core design:

```text
                +----------------------+
                |     Input / Case     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Read / Extract Data  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Validate / Transform |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Populate PI Fields   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Verify Prepared Data |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Human Decision Gate  |
                +----------+-----------+
                           |
                   +-------+-------+
                   |               |
                 APPROVE         HOLD /
                 (human)       EXCEPTION
```

Design rule:

> Same mechanical workflow + different data = parameter variation.

> Different mechanical workflow = validated branch.

> Different business decision = human gate.

---

# 11. Technical Interface Discovery

The PI subset contains:

- 209 canonical PI segments
- 7,325 raw PI event rows

Observed browser routes included:

| Route | Events | Sessions | Segments |
|---|---:|---:|---:|
| `/payroll-items` | 3,523 | 14 | 190 |
| `/onboarding` | 42 | 4 | 7 |
| `/social-insurance` | 36 | 4 | 6 |
| `/leave-applications` | 18 | 2 | 4 |
| `/resident-tax` | 1 | 1 | 1 |

Interface evidence included:

| Event / Interface | Rows | Sessions | Segments |
|---|---:|---:|---:|
| Browser form input | 234 | 14 | 154 |
| Clipboard transfer | 336 | 15 | 184 |
| Browser navigation | 57 | 13 | 42 |
| Browser click | 726 | 14 | 188 |
| App switch | 436 | 15 | 150 |
| Keyboard | 1,618 | 15 | 190 |
| Shortcut | 497 | 15 | 188 |
| Text input complete | 48 | 9 | 20 |

The repository did **not** contain an executable implementation of the real `/payroll-items` production portal.

The technical discovery therefore classified the production target as:

```text
NO EXECUTABLE TARGET EVIDENCE
```

A controlled local reconstruction was used for prototyping.

---

# 12. Prototype Validation

The prototype was tested with normal cases, parameter variations, document variants, missing inputs, invalid inputs, unsupported parameters, and human-gate cases.

## Step 3E2 validation

10 / 10 cases passed.

Cases covered:

- normal PI cases,
- parameter variation,
- document variation,
- missing input,
- invalid input,
- unsupported parameter,
- unexpected invalid input,
- human-gate case,
- unexpected UI state.

Successful preparations preserved the human gate.

No final business-decision button was clicked by the automation.

## Combined prototype validation

Across Step 3E1 + Step 3E2:

- **16 / 16 tests passed**
- **0 manual corrections**
- **0 Register clicks**
- **0 Hold clicks**

## Step 3F actual variant validation

6 / 6 passed:

1. NORMAL expense reimbursement
2. Different business subtype — payroll change
3. Document/application variant
4. Missing field — safe stop
5. Unexpected/invalid value — safe stop
6. Decision exception — human gate

All validated cases preserved the decision boundary.

---

# 13. Real Dataset-B PI Replay

A real-data offline replay was built against the observed Dataset-B PI event stream.

Results:

- 209 canonical PI segments
- 15 sessions
- 20 event files
- 20,477 Dataset-B events
- 7,325 real PI event rows
- 77 direct OCR-supported PI segments
- exact timestamp-based mapping
- 0 OCR extrapolation
- 0 production access
- no changes to `segments.jsonl`

Replay classification:

| Replay Class | Segments | Share |
|---|---:|---:|
| STRUCTURAL_ONLY | 132 | 63.16% |
| HUMAN_GATE | 55 | 26.32% |
| REPLAYABLE_WITH_VARIANT | 14 | 6.70% |
| INSUFFICIENT_TARGET_EVIDENCE | 7 | 3.35% |
| REPLAYABLE | 1 | 0.48% |

Coverage across the 209 PI segments:

| Evidence | Coverage |
|---|---:|
| Raw event mapping | 209 / 209 |
| Browser target | 186 / 209 |
| Navigation | 42 / 209 |
| Form input | 154 / 209 |
| Clipboard | 184 / 209 |
| App switch | 150 / 209 |
| Direct OCR | 77 / 209 |
| Transformation evidence | 190 / 209 |
| Human-gate evidence | 59 / 209 |

The 132 `STRUCTURAL_ONLY` segments are not treated as failed automation cases. They are underdetermined because the operation logs do not always contain enough screen-content information to reconstruct the exact business values.

---

# 14. Why Raw Logs Cannot Reconstruct All Business Fields

A key finding from the real-data replay was that raw browser events often expose **interaction metadata** but not the actual case values.

For example, event metadata could expose targets such as:

```text
pi-note
btn-pi-ok
```

but not necessarily:

```text
employee ID
employee name
amount
category
status
```

Those values were visible in the rendered screen rather than the interaction payload.

`context.extracted_text` was documented by the event schema but was observed as absent in the delivered PI interaction logs.

Therefore:

> Structured case data could not be fully recovered from the raw operation log payload alone. Screenshots were the primary source of case-content evidence in the available logs.

For production automation, the authoritative data should be read from the live DOM or an approved API at runtime rather than reconstructed from historical event payloads.

---

# 15. Automation Value

The PI family represents:

- **209 observed segments**
- **35.78% of total segmented workload time**
- approximately **63.06 minutes** of observed segmented PI activity

This is workload share, not guaranteed savings.

The project deliberately does **not** estimate time saved by multiplying prototype runtime by the number of historical cases because:

- the local prototype runs against a lightweight reconstructed interface,
- no comparable manual-vs-automated timing study was performed,
- production authentication/network/UI latency was unavailable,
- the actual production decision and exception paths remain human.

A valid savings estimate should be produced later using controlled production timing:

```text
baseline manual handling time
            -
human-assisted automated handling time
            =
observed time reduction
```

---

# 16. Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Production UI differs from reconstructed UI | Prototype selectors/workflow may not transfer | Validate selectors and page states against production |
| Selector instability | Automation breaks after UI changes | Stable DOM identifiers and UI-change monitoring |
| Authentication/session expiry | Workflow interruption | Detect authentication loss and hand off to human |
| Missing source values | Incorrect field preparation | Validate required fields and safe-stop on absence |
| OCR errors | Wrong semantic interpretation during analysis | Use OCR as supporting evidence, not business truth |
| Business-rule ambiguity | Incorrect automated decision | Preserve human decision gate |
| Duplicate submission | Duplicate or incorrect processing | Do not auto-click final decision controls |
| Exceptional cases | Unexpected workflow continuation | Explicit safe-stop states |
| Production permissions/API unavailable | No integration path | Treat API/access as an implementation prerequisite |
| Sensitive payroll/employee information | Privacy/security exposure | Access controls, secure storage, audit logging |
| Segmentation uncertainty on B | Workload grouping errors | Manual sampling, repeat-pattern validation, confidence review |
| Unseen workflow variants | Automation may encounter unsupported states | Parameterized core + validated branches + safe stop |
| No Dataset-B ground truth | Cannot directly measure segmentation accuracy | Report structural evidence and validation limitations |

---

# 17. Reproducibility

## Environment

Recommended Python setup:

```bash
python -m venv .venv
```

Activate the environment and install project dependencies:

```bash
pip install -r requirements.txt
```

The OCR workflow uses a separate environment because of its heavier dependencies.

---

## Raw Data

Place the provided raw dataset under the expected project `raw_data/` structure.

Raw data is ignored by Git:

```text
raw_data/
```

Do not commit raw operational data or screenshots.

---

## Core Processing

The repository contains scripts corresponding to:

### Ground truth and segmentation

```text
ground_truth.py
gt_boundaries.py
boundary_signal_analysis.py
train_segment_boundary_model_repaired.py
rebuild_gt_and_segment_evaluation.py
build_final_dataset_b_segments.py
```

The final B segmentation should result in:

```text
segments.jsonl
```

with **540 records**.

### Step 2 analysis

Key analysis/build scripts include:

```text
analyze_dataset_b_step1_processes.py
analyze_dataset_b_segments.py
analyze_dataset_b_step2_processes.py
discover_step2_layer5_subprocesses_v1.py
revisit_step2_layer5_subprocesses_v2.py
finish_step2_layer6_semantic_and_propagation_v2.py
build_step2_layer8_mechanical_human_v1.py
build_step2_layer9_feasibility_risk_v5.py
build_step2_layer10_automation_decision_table_v1.py
build_step2_layer11_final_recommendation_v2.py
consolidate_step2_to_excel.py
```

The authoritative Step-2 workbook is:

```text
Step2_Final_Evidence_Package.xlsx
```

---

## PI Workflow and Prototype

Key Step-3 scripts include:

```text
step3a1_authoritative_pi_evidence_reconstruction_v2.py
step3a2_pi_chronological_workflow_reconstruction_v3.py
step3a4_pi_workflow_map_v3_patch1.py
step3b_pi_automation_boundary_v3.py
step3c_pi_parameterized_vs_branch_analysis_v3.py
step3d_pi_technical_interface_discovery_v2.xlsx
step3e2_pi_prototype_validation_v1.py
step3e3_pi_final_prototype_assessment_v2.py
step3f_pi_prototype_variant_validation_v2.py
step3g_pi_value_quantification_v2.py
step3h_pi_real_dataset_replay_v6.py
step3i_pi_real_data_extraction_similarity_v4.py
```

For the prototype sandbox:

```bash
cd prototype_pi_sandbox
```

The sandbox contains the reconstructed PI interface and runner.

For the live browser demonstration, the live runner is intended to be:

```bash
python pi_demo_runner_live.py
```

The live runner should be committed into the repository as part of the final submission.

The automation demo is also available at:

https://youtu.be/LL4OfRidPTM

---

# 18. Authoritative Outputs

The final repository should prominently expose the following artifacts:

```text
segments.jsonl

Step2_Final_Evidence_Package.xlsx
STEP2_DECISION_SCORECARD_FINAL.xlsx

step3a4_pi_workflow_map_v3.xlsx
step3b_pi_automation_boundary_v3.xlsx
step3d_pi_technical_interface_discovery_v2.xlsx
step3e2_pi_prototype_validation_v1.xlsx
step3e3_pi_final_prototype_assessment_v1.xlsx
step3f_pi_actual_variant_validation_v1.xlsx
step3g_pi_value_quantification_*.xlsx
step3h_pi_real_dataset_replay_v6.xlsx
step3i_pi_real_data_extraction_similarity_v3.xlsx

FINAL_REPORT_From_Operation_Logs_to_Automation_Proposal.pdf
FINAL_REPORT_From_Operation_Logs_to_Automation_Proposal.docx

Work log / WORK_LOG.md
README.md
```

Intermediate/debug files may exist for reproducibility and historical investigation, but the files above should be treated as the main submission artifacts.

---

# 19. Git History

The development history was kept meaningful rather than collapsing the project into one final commit.

The documented progression includes commits corresponding to:

```text
Initialize project structure
Add initial dataset discovery loader
Improve dataset validation
Build canonical session reconstruction
Add ground truth extraction
Add ground truth boundary extraction
Add baseline segmentation and error analysis
...
Step 2 analysis / semantic evidence / automation decision
Step 3 PI prototype and validation
```

The repository intentionally records methodological corrections and failed experiments because they explain how the final pipeline was reached.

Before submission, verify:

```bash
git status
git log --oneline --decorate -20
```

and confirm that:

- raw data is not tracked,
- final `segments.jsonl` is tracked,
- authoritative reports/artifacts are tracked as intended,
- the live prototype runner is included and committed,
- there are no credentials or secrets in Git history or working files.

---

# 20. Seven-Day Project Allocation

The final submission can map the work into the requested seven-day allocation as:

| Day | Focus |
|---|---|
| 1 | Dataset discovery, schema understanding, loader, integrity checks |
| 2 | Session reconstruction, chunk handling, ground-truth extraction |
| 3 | Boundary signal analysis and heuristic baselines |
| 4 | Supervised boundary model, consolidation and error analysis |
| 5 | Dataset-B segmentation, process-family discovery, OCR/semantic analysis |
| 6 | PI candidate selection, workflow reconstruction, technical discovery and prototype |
| 7 | Prototype validation, real-data replay, impact/risk analysis and final packaging |

The detailed actual work history is maintained separately in the completed work log.

---

# 21. Limitations

The following limitations are intentional and should not be interpreted as missing implementation:

### Dataset B has no ground truth

Segmentation quality on B is not reported as an accuracy/F1 metric.

### Production portal was not accessible

The local prototype is not evidence of production integration.

### Historical logs do not expose all business values

Some case values were only visible in screenshots, not raw interaction payloads.

### OCR coverage is incomplete

Direct OCR evidence does not cover all 540 segments.

### Semantic propagation was rejected

The attempted propagation method achieved only 47.1% precision on accepted held-out cases and was therefore not used to force labels onto uncovered segments.

### Savings are not estimated

Observed workload time is reported, but realized automation savings require a controlled production timing study.

### Human decisions remain outside automation

Approval, hold, rejection, policy interpretation and exception handling are deliberately retained as human-controlled actions.

---

# 22. AI Assistance Disclosure

AI tools were used during development for engineering assistance, including:

- code drafting and refactoring,
- debugging,
- analysis design,
- experiment interpretation,
- documentation drafting,
- validation planning.

The analysis pipeline, evaluation methodology, evidence checks, failure analysis, final process-family interpretation, automation boundaries, and validation decisions were reviewed and tested against the available project data.

The prototype does not delegate the final business decision to an AI model.

---

# 23. Final Submission Checklist

Before submission, verify:

```text
[ ] segments.jsonl contains all 540 Dataset-B inferred work units
[ ] segments.jsonl is not PI-only
[ ] Dataset A segmentation methodology is documented
[ ] Dataset B limitations are documented
[ ] Step-2 family counts and absolute time are documented
[ ] PI / LA candidate rationale is documented
[ ] PI human-decision boundary is explicit
[ ] Prototype is runnable
[ ] Live runner is inside the repository
[ ] Live runner is committed to Git
[ ] 16/16 combined prototype tests are documented
[ ] 6/6 final variant-validation tests are documented
[ ] Real Dataset-B PI replay evidence is documented
[ ] Production integration limitation is explicit
[ ] Manual work remaining is documented
[ ] Realistic impact methodology is documented
[ ] Risks and mitigations are documented
[ ] Seven-day allocation is included
[ ] Detailed work log is included/submitted
[ ] Final report is included
[ ] Raw data is not committed
[ ] No secrets/credentials are committed
[ ] Git history is clean and meaningful
[ ] Final submission artifacts are clearly identified
```

---

# 24. Final Project State

The project progressed from an unstructured event stream to a defensible automation proposal:

```text
183,245 raw events
        |
        v
78 sessions
        |
        v
Dataset-A boundary modelling
        |
        v
Dataset-B inference
        |
        v
540 inferred work units
        |
        v
6 stable workload families
        |
        v
PI selected as first pilot
        |
        v
209 PI segments
        |
        v
workflow + technical evidence
        |
        v
parameterized HITL prototype
        |
        v
16/16 prototype validations passed
        |
        v
real Dataset-B PI replay
        |
        v
production-readiness gaps explicitly identified
```

The resulting proposal is not an autonomous payroll decision system. It is a **human-in-the-loop automation assistant** for the repetitive mechanical portion of the PI workflow, with safe stops and explicit human control at business-decision points.
