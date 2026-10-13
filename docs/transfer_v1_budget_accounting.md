# Transfer v1 operational budget accounting

Date: 2026-10-07.

The registered 600-second wall-time cap is one global active-command budget for
dataset generation plus the later independent audit/replay. Generation records
its consumed time in the run manifest; audit receives only the remaining time
from that manifest and reports the combined total. Controller setup and report
writing are separately timestamped. A generator cap leaves the dataset
incomplete, and audit cannot convert it to a complete suitability result. This
note records the binding Task 4b/4c clarification without changing the
registered threshold or any candidate, audit, or scientific setting.

No corpus run is performed as part of the Task 4b implementation checks.
