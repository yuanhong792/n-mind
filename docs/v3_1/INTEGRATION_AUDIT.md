# V3.1 集成审计

|能力|路径|状态|处理|
|---|---|---|---|
|聊天|src/nmind/bot/chat_service.py|REUSABLE|V3Facade来源归因适配|
|运行时|src/nmind/bot/runtime.py|REUSABLE|复用用户优先/暂停机制|
|记忆|src/nmind/memory/service.py|REUSABLE|新增保存/纠正/删除API|
|目标与练习|goal_manager.py, practice.py|PARTIAL|一次学习调用真实Practice+Oracle|
|能力台账|capability_tracker.py|REUSABLE|记录tool/oracle，不计入neural|
|任务|src/nmind/v3/tasks.py|PARTIAL|V3持久化状态机|
|事件|src/nmind/v3/events.py|REUSABLE|SQLite append-only审计|
|Web工作台|src/nmind/bot/web_server.py|PARTIAL|新增V3.1 API|
|联网|Explorer/Fetcher|REUSABLE|V3.1默认不启用|

环境：.venv-gate24，PyTorch 2.15.0+cpu，CUDA false；本轮不联网、不训练。
