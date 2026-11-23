# V3.2 UI action matrix

|UI action|request|observed effect|
|---|---|---|
|Send|POST /api/v3/chat|real tool response visible|
|One practice|POST /api/learning/cycles|Oracle event, tool/oracle source, 0 steps|
|Enable autonomy|POST /api/v3/learning|permission state changes|
|Create safe task|POST /api/v3/tasks|queued task ID visible|
|Status|GET /api/v3/status|offline/model/training state refreshed|

Memory controls and dedicated growth/task navigation remain PARTIAL in the legacy shell and are recorded as such.
