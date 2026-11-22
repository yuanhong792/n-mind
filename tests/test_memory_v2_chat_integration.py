from nmind.bot.chat_service import ChatRequest, ChatService
from nmind.memory import MemoryService, MemoryStore


def test_chat_marks_memory_as_distinct_source(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    ms = MemoryService(store)
    ms.observe(session_id="old", observation="调试失败后应先查看完整错误堆栈", task="debug", confidence=.9)
    out = ChatService(memory_service=ms).reply(ChatRequest.new("new", "错误堆栈怎么处理"))
    assert out.response_source == "memory"
    assert out.memory_hits
    assert "历史记忆" in out.note
