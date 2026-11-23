# V3.3 当前资产与缺口

## REUSE
- V3Facade: `src/nmind/v3/facade.py`
- EventLog/TaskService: `src/nmind/v3/events.py`, `src/nmind/v3/tasks.py`
- Practice/CapabilityTracker: `src/nmind/bot/practice.py`, `src/nmind/bot/capability_tracker.py`
- WebBotServer and V3.2 workbench: `src/nmind/bot/web_server.py`, `src/nmind/bot/ui/web/index.html`

## EXTEND
- `src/nmind/v3/learning_engine.py`: persistent explainable goal selection, bounded cycles, review and failure ledger.
- V33 routes: `/api/v33/learning/*`.

## GAPS / DEFER
- Neural-only learning remains unverified and is not promoted by this engine.
- Memory correction UI and full D1-D9 browser evidence require a dedicated browser run.
- Historical 276-node manifest is unavailable; current collection is measured from the repository.
