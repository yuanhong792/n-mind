# Planning V1 code map

`nmind.planning.contracts` defines persistent goal/plan/step/result contracts. `validator` rejects cycles, missing dependencies, unsupported tools and budget violations. `planner.RulePlanner` creates finite rule plans with `plan_source=rule`. `tool_adapter` wraps existing KnowledgeSearch and MemoryService plus a deterministic calculator. `executor.PlanExecutor` is single-threaded, bounded, pausable, cancellable and journals every step in SQLite. `reflection.record_experience` writes only auditable outcomes to Memory V2.

Neural planning is not executable in this release: any future proposal must be separately validated and labelled as a candidate. No shell tool, network tool, training tool or arbitrary file tool is registered.
