# PI Automation Sandbox

This is the presentation-ready local prototype for Step 3E/3F.

## Open the reconstructed web interface

```powershell
python prototype\pi_sandbox_server.py --open
```

URL:

`http://127.0.0.1:8765/payroll-items`

The page shows:
- a visible source case
- the target `/payroll-items` form
- evidence context
- verification state
- explicit human gate
- an auditable automation trace

## Run the automated demonstration

```powershell
python prototype\pi_demo_runner.py
```

The six cases are:
1. normal expense reimbursement
2. payroll-change parameter
3. document/application variant
4. missing required input
5. invalid amount
6. decision/exception

The automation never clicks Register or Hold.

## Important evidence boundary

The application is a controlled reconstruction. The field concepts, observed route,
mechanical workflow and human decision boundary are grounded in the project evidence.
The fixture values are local demo values and must not be presented as production records.

The architectural point being demonstrated is:

`live DOM read -> deterministic validation/transformation -> target DOM population -> verification -> human review`
