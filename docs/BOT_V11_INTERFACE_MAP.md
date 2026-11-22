# Bot V1.1 interface map

## Runtime path

`ui/web/index.html` → `WebBotServer` → `BotClient`/`BotStore` → `BotRuntime.process_chat_queue` → `ChatService.reply`.

`ChatService` keeps the source boundary: `tool` first, then `retrieval`, then optional `memory`, then `neural`, otherwise `none`. Memory V2 is created once by `build_web_stack` from `root_dir/memory.db` and passed into ChatService; it is not a second memory implementation.

## APIs

- `POST /api/chat/sessions`, `GET /api/chat/sessions/{id}/messages`, `POST /api/chat/sessions/{id}/messages`, `GET /api/chat/messages/{message_id}`
- `GET /api/knowledge/stats`, `POST /api/knowledge/search`
- `GET /api/memories?query=...` — bounded Memory V2 recall
- `POST /api/memories/{id}/feedback` — explicit user feedback
- `DELETE /api/memories/{id}` — tombstoned forget
- `GET /api/runtime/status`, `GET /api/web/status`, `POST /api/control`

All server binding remains `127.0.0.1`. Real network access remains controlled by `BotPolicy` and `SourceRegistry`; frontend switches cannot bypass either.

## Persistence

Chat messages stay in `bot_state.db`; Memory V2 stays in `memory.db`; Python web knowledge stays in `knowledge.db`. This preserves the training/data separation and supports restart recovery.
