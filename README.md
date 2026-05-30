# AI Python 编程私教

## 1. 项目概述

基于 LangGraph 构建的 AI 编程教学助手。通过引导式对话帮助学员学习 Python，支持代码沙箱执行、长期记忆画像、流式输出、可观测追踪。

提供两种交互方式：
- **CLI 终端** — `python main.py`，命令行交互
- **Web 界面** — `streamlit run app.py`，浏览器交互

---

## 2. 技术栈

| 层 | 技术 | 用途 |
|---|---|---|
| Agent 框架 | LangGraph | 状态机驱动的 Agent 图编排 |
| LLM SDK | LangChain + langchain-deepseek | LLM 调用链、工具绑定、流式 |
| 语言模型 | DeepSeek (ChatDeepSeek) | 核心对话推理 |
| 代码执行 | Docker (python:3.10-slim) | 安全沙箱运行学员代码 |
| 数据校验 | Pydantic + Field | 工具参数 Schema 定义和校验 |
| 状态序列化 | Python json | 对话历史、长期记忆持久化 |
| 流式 UI | Streamlit | Web 界面 |
| 可观测性 | 自定义 Tracer（stderr） | LLM 调用、节点耗时、Token 追踪 |

---

## 3. 架构

### 3.1 文件结构

```
project/
├── main.py                     # CLI 入口，对话循环 + 记忆管理
├── app.py                      # Streamlit Web 入口
├── agent/
│   └── graph.py                # LangGraph 图定义、节点、状态、分层 Prompt
├── tools/
│   ├── sandbox.py              # Docker 代码沙箱
│   ├── schemas.py              # Pydantic 工具输入 Schema + 危险代码校验
│   └── python_tools.py         # 工具注册（StructuredTool）
├── memory/
│   ├── summarizer.py           # 消息序列化、滑动窗口、对话历史读写
│   └── long_term.py            # 长期记忆（学员画像）：加载/保存/格式化/LLM 提取
├── models/
│   └── llm.py                  # LLM 客户端创建
├── config/
│   └── settings.py             # 全局配置（路径、安全限制、模型参数）
├── observability/
│   └── tracer.py               # 执行过程追踪器
└── storage/
    ├── chat_history.json        # 对话历史持久化
    └── long_term_memory.json    # 长期记忆持久化
```

### 3.2 Agent 图结构

```
         ┌──────────────┐
         │              │
         ▼              │
     ┌───────┐    ┌─────────┐
     │ tutor │───→│  tools  │
     └───────┘    └─────────┘
         │
         └──→ END
```

两个节点：
- **tutor** — 组装分层 System Prompt → 调用 LLM → 判断是否需调工具
- **tools** — 执行 LLM 发起的工具调用 → 结果回注消息列表

条件路由 `should_continue` 决定 tutor 后是去 tools 还是结束。
加入 `MAX_TOOL_CALLS` 限制防止无限工具循环。

### 3.3 数据流

```
main.py:
  启动 → load_ltm() → session_count+1 → to_prompt(ltm)
         ↓
  state["profile_prompt"] = ...
  state["current_task"] = ltm["current_focus"]
         ↓
  app.invoke(state) ──→ 图跑完全程 → 返回最终 state
         ↓
  update_ltm(ltm, messages) → save_ltm(ltm)
         ↓
  循环至下一轮用户输入
```

---

## 4. Agent 状态

```python
class AgentState(TypedDict):
    messages:      list    # 对话消息列表（自动追加）
    next_step:     str     # 下一步路由目标
    tool_call_count: int   # 本轮已调工具次数
    profile_prompt: str    # 学员画像文本（每次 invoke 前刷新）
    current_task:  str     # 当前学习重点（从长期记忆同步）
```

---

## 5. 关键功能

### 5.1 分层 Prompt

System Prompt 按职责拆分为独立模块，拼接时按需组合：

```
IDENTITY        → 你是谁：AI 编程私教
SAFETY_RULES    → 不能直接给答案、防危险代码
TOOL_RULES      → 何时用什么工具
TEACHING_STYLE  → 教学方法论（引导式、苏格拉底式、分步拆解）
STUDENT_CONTEXT → 学员画像（从长期记忆注入）
CURRENT_TASK    → 当前学习重点（可选）
```

每一层可独立修改、条件启用，便于扩展不同教学模式。

### 5.2 工具 Schema 化

使用 Pydantic BaseModel 显式定义工具输入参数：

```python
class RunPythonCodeSchema(BaseModel):
    code: str = Field(description="要执行的 Python 代码字符串")
```

配合 `field_validator` 在 Schema 层拦截危险代码模式（os.system、subprocess、eval、exec、__import__ 等），提前拒绝而非依赖运行时隔离。

Docker 沙箱提供基础设施层隔离（read_only、network_disabled、cap_drop），两层防护叠加。

### 5.3 长期记忆系统

| 函数 | 职责 |
|------|------|
| `load()` | 读取 JSON，不存在返回默认结构 |
| `save()` | 写入 JSON，自动时间戳 |
| `to_prompt()` | 结构化记忆 → 可读中文段落供 LLM 使用 |
| `update()` | 对话完成后，LLM 分析对话提取更新并合并 |

存储结构：

```json
{
  "profile": {
    "skill_level": "beginner",
    "learning_style": "hands-on",
    "strengths": ["变量", "循环"],
    "weaknesses": ["函数参数"],
    "common_mistakes": ["缩进不一致"]
  },
  "topics": ["变量", "循环", "函数"],
  "session_count": 5,
  "current_focus": "函数定义",
  "last_updated": "2026-05-29T10:00:00"
}
```

动态更新机制：每次对话完成后，用独立 LLM 调用分析本轮对话，提取学员水平变化，写回 JSON。
每次 invoke 前重新生成 `profile_prompt` 注入 state，实现同一次会话内的记忆更新生效。

### 5.4 代码执行沙箱

Docker 容器安全策略：
- `read_only=True` — 根文件系统只读
- `network_disabled=True` — 禁用网络
- `cap_drop=["ALL"]` — 剥夺所有系统权限
- `security_opt=["no-new-privileges"]` — 禁止提权
- `pids_limit=64` — 防 fork 炸弹
- `nano_cpus=0.5*1e9` — 限制 CPU
- 超时自动 kill

### 5.5 流式输出

通过 `on_token` 回调函数实现流式逐字输出：

```
main.py 定义 on_token → 传给 graph
                         ↓
tutor_node 循环中每收到一个 chunk 就调用 on_token
                         ↓
CLI: print(text, end="", flush=True)
Streamlit: collected.append(text) → placeholder.markdown(...)
```

### 5.6 可观测追踪

自定义 Tracer，手动插桩到关键节点，输出到 stderr 不影响对话界面：

| 事件 | 捕获信息 |
|------|---------|
| `node_enter` | 进入节点 + 时间戳 |
| `node_exit` | 离开节点 + 耗时 + 下一步去向 |
| `llm_send` | 发送消息数、总字符数 |
| `llm_recv` | Token 消耗（input/output）、耗时、是否调工具 |
| `tool_call` | 工具名、参数（截断 200 字符） |
| `tool_result` | 返回长度、执行耗时 |

### 5.7 对话历史管理

滑动窗口机制，截取最近 N 轮对话，控制 token 消耗：

```python
def sliding_window(messages, max_rounds=5):
    system = messages[:1]         # 保留 System Prompt
    history = messages[1:]
    window = history[-(max_rounds * 2):]   # 保留最近 max_rounds 轮
    return system + window
```

### 5.8 双重入口

CLI 版支持恢复上次对话（检测 `chat_history.json` 是否存在，询问用户是否继续）。
Streamlit 版利用 `st.cache_resource` 缓存 LLM 实例，避免重复创建。

### 5.9 路径安全

文件读取工具 `load_code_file` 的四层校验：
1. `os.path.realpath` 解析符号链接和 `..`
2. `os.path.commonpath` 检查是否在白名单目录内
3. 文件类型白名单（`.py`、`.txt`、`.md`）
4. 文件大小上限（100KB）

---

## 6. 配置

```python
# settings.py
DEEPSEEK_MODEL           # 模型名（从 .env 读取）
DEEPSEEK_API_KEY         # API 密钥（从 .env 读取）
ALLOWED_DIR              # 允许读取的文件目录
ALLOWED_EXT              # 允许的文件扩展名
MAX_SIZE                 # 允许读取的文件大小上限
MAX_TOOL_CALLS           # 每轮最大工具调用次数
HISTORY_FILE             # 对话历史 JSON 路径
LONG_TERM_MEMORY_FILE    # 长期记忆 JSON 路径
```

---

## 7. 运行方式

```bash
cd project

# CLI 模式
python main.py

# Web 模式
streamlit run app.py
```

Tracer 日志输出到 stderr，可独立重定向：

```bash
python main.py 2>trace.log
```
