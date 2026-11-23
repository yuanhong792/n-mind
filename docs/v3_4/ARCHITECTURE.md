# V3.4 Architecture

The existing V3 facade owns `KnowledgeGraphService` and emits EventLog entries. The graph is a bounded SQLite service with a read-only query path and explicit mutation endpoints. V3.3 can record verified Practice/Oracle outcomes as scoped experiences. Chat, neural, tool, retrieval and memory sources remain distinct.
