from typing import TypedDict, List, Any, Annotated
from langgraph.graph import add_messages, END, StateGraph
from langchain_core.messages import SystemMessage
from memory.summarizer import sliding_window
from config.settings import MAX_TOOL_CALLS

# 定义 Agent 状态（add_messages 保证消息追加而非替换）
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    next_step: str
    tool_call_count: int
    profile_prompt: str
    current_task: str

# 使用整段的prompt：已弃用
# TUTOR_SYSTEM_PROMPT = """你是一位耐心且专业的Python编程私教。你的核心任务是引导学员学习，而不是直接给出答案。

# 请严格遵循以下教学原则：
# 1. **引导式提问**：当学员提问时，先反问他们的想法或已知信息，引导他们自己找到答案。
# 2. **苏格拉底式教学**：通过提问暴露他们思路中的漏洞，再给出提示，而不是直接说“你错了”。
# 3. **鼓励动手**：如果问题可以用代码验证，请生成示例代码并用 `run_python_code` 工具运行，展示直观结果。
# 4. **错误分析**：当学员提交的代码报错时，不要只给正确代码。要解释错误原因，并引导他们思考如何修复。
# 5. **避免直接答案**：如果学员要你写作业，你可以说：“我们可以一起分步来想这个问题，你先说说你会怎么开始？”

# 请以友好、鼓励的语气开始对话。可以说：“你好！我是你的Python私教。我们今天学点什么，还是想让我先出个题考考你？” 

# 当前教学状态：欢迎新学员，准备评估水平或直接开始教学。
# """

# 采用分层式PROMPT，更适合选择和拓展
IDENTITY = """你是一位耐心且专业的Python编程私教。"""

SAFETY_RULES = """## 安全规则
1. 绝对不能直接给出作业答案，只能引导学员自己找到答案。
2. 如果学员要求写完整代码替ta完成作业，礼貌拒绝并引导ta分步思考。
3. 不得建议、生成或执行可破坏系统的危险代码。
"""

TOOL_RULES = """## 工具使用规则
1. 当需要运行或测试代码时，使用 `run_python_code` 工具。
2. 当需要查看学员的代码文件时，使用 `load_code_file` 工具。
3. 运行代码前，先让学员预测输出结果，再运行验证，加深理解。
"""

TEACHING_STYLE = """## 教学方法
1. **引导式提问**：学员提问时，先反问ta的想法或已知信息，引导ta自己找到答案。
2. **苏格拉底式教学**：通过提问暴露思路中的漏洞，再给提示，而不是直接说"你错了"。
3. **鼓励动手**：能用代码验证的问题，生成示例代码并运行展示直观结果。
4. **错误分析**：学员代码报错时，解释错误原因，引导ta思考如何修复。
5. **分步拆解**：复杂问题拆成小步骤，引导学员逐一来。
"""

STUDENT_CONTEXT = """## 学员背景
{profile_prompt}
"""

CURRENT_TASK = """## 当前任务
{current_task}
"""

OPENING = """以友好、鼓励的语气开始对话。可以说："你好！我是你的Python私教。我们今天学点什么，还是想让我先出个题考考你？" """

def build_tutor_graph(llm_with_tools, tool_node, on_token=None, tracer=None):
    """接收已绑好 tools 的 llm 和 tool_node，返回编译好的图。
    on_token: 可选回调，每收到一个 token 时调用 on_token(text)
    """

    def tutor_node(state: AgentState) -> dict:
        """教学节点：核心逻辑，处理用户输入并生成教学回复。"""
        tracer and tracer.node_enter("tutor")

        # system_content = TUTOR_SYSTEM_PROMPT
        parts = [
            IDENTITY,
            SAFETY_RULES,
            TOOL_RULES,
            TEACHING_STYLE,
            STUDENT_CONTEXT.format(profile_prompt=state.get("profile_prompt", "")),
        ]
        if state.get("current_task"):
            parts.append(CURRENT_TASK.format(current_task = state["current_task"]))
        
        system_content = "\n\n".join(parts)
        all_messages = [SystemMessage(content=system_content)] + state["messages"]
        all_messages = sliding_window(all_messages)

        # ... 组装 system_content 和 all_messages ...
        tracer and tracer.llm_send(all_messages)

        full_response = None
        for chunk in llm_with_tools.stream(all_messages):
            full_response = chunk if full_response is None else full_response + chunk
            if chunk.content and on_token:
                on_token(chunk.content)
        if full_response and full_response.content and on_token:
            on_token("\n")

        has_tools = bool(full_response and full_response.tool_calls)
        tracer and tracer.llm_recv(full_response, has_tools)
        tracer and tracer.node_exit("tutor", next_step="tools" if has_tools else END)

        return {
            "messages": [full_response],
            "next_step": "tools" if full_response and full_response.tool_calls else END
        }

    def run_tools(state: AgentState) -> dict:
        """执行工具调用并返回更新后的状态。"""
        tracer and tracer.node_enter("tools")
        
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls"):
            for tc in last_msg.tool_calls:
                tracer and tracer.tool_call(tc["name"], tc["args"])

        tool_messages = tool_node.invoke(state["messages"])

        for msg in tool_messages:
            if hasattr(msg, "name"):
                tracer and tracer.tool_result(len(msg.content or ""))

        tracer and tracer.node_exit("tools")

        return {"messages": tool_messages, "tool_call_count": state.get("tool_call_count", 0) + 1}

    def should_continue(state: AgentState) -> str:
        if state.get("tool_call_count", 0) > MAX_TOOL_CALLS:
            return END
        return state.get("next_step", END)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("tutor", tutor_node)
    workflow.add_node("tools", run_tools)
    workflow.set_entry_point("tutor")
    workflow.add_conditional_edges(
        "tutor",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    workflow.add_edge("tools", "tutor")

    return workflow.compile()