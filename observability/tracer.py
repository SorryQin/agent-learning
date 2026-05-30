import time
import sys


class Tracer:
    """Agent 执行过程跟踪器。调用方在关键节点手动调用对应方法。"""

    def __init__(self):
        self.turn = 0
        self._ident = ""       # 缩进，区分嵌套层级

    def start_turn(self):
        self.turn += 1
        self._info(f"{'═' * 40} Turn #{self.turn} {'═' * 40}")

    def node_enter(self, name: str):
        self._node_start = time.time()
        self._info(f"[NODE] → {name}")

    def node_exit(self, name: str, next_step: str = ""):
        d = time.time() - self._node_start
        flow = f" → {next_step}" if next_step else ""
        self._info(f"[NODE] ← {name} ({d:.1f}s){flow}")

    def llm_send(self, messages: list):
        self._llm_start = time.time()
        char_count = sum(len(m.content or "") for m in messages if hasattr(m, "content"))
        self._info(f"[LLM]  发送 — {len(messages)} 条消息, {char_count} 字符")

    def llm_recv(self, response, has_tools: bool):
        d = time.time() - self._llm_start
        # 尝试取 usage_metadata
        usage = None
        try:
            usage = getattr(response, "usage_metadata", None)
        except Exception:
            pass
        if usage:
            self._info(
                f"[LLM]  收到 — 输入 {usage.get('input_tokens', '?')} "
                f"/ 输出 {usage.get('output_tokens', '?')} tokens ({d:.1f}s)"
            )
        else:
            self._info(f"[LLM]  收到 ({d:.1f}s) 调工具={has_tools}")

    def tool_call(self, name: str, args: dict):
        self._tool_start = time.time()
        self._tool_name = name
        args_str = str(args)[:200]
        self._info(f"[TOOL] → {name} — {args_str}")

    def tool_result(self, output_len: int):
        d = time.time() - self._tool_start
        self._info(f"[TOOL] ← {self._tool_name} ({d:.2f}s, 返回 {output_len} 字符)")

    def _info(self, msg):
        ts = time.strftime("%H:%M:%S")
        print(f"  ⚡ [{ts}] {msg}", file=sys.stderr)
