import streamlit as st

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode

from config.settings import *
from models.llm import create_llm
from tools.python_tools import tools
from memory.summarizer import save_history, load_history
from agent.graph import build_tutor_graph


# 缓存 LLM / tools / node 等创建一次就够了的大对象
@st.cache_resource
def get_cached_objects():
    llm = create_llm()
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tool_node


llm_with_tools, tool_node = get_cached_objects()

# 初始化 session_state
if "messages" not in st.session_state:
    history = load_history()
    st.session_state.messages = []
    for msg in history:
        st.session_state.messages.append(msg)

if "graph" not in st.session_state:
    st.session_state.graph = None


# 页面标题
st.title("AI Python 编程私教")

# 显示历史消息
for msg in st.session_state.messages:
    role = "ai" if msg.type == "ai" else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# 输入框
if prompt := st.chat_input("输入你的 Python 问题..."):
    # 显示用户消息
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 回复（流式）
    with st.chat_message("ai"):
        placeholder = st.empty()
        collected = []

        def on_token(text):
            collected.append(text)
            placeholder.markdown("".join(collected))

        # 构建并运行图的副本（每次输时需要新一次的 invoke）
        graph = build_tutor_graph(llm_with_tools, tool_node, on_token=on_token)
        result = graph.invoke({
            "messages": st.session_state.messages,
            "next_step": "tutor",
        })

        # 把 AI 回复保存到历史
        ai_msg = result["messages"][-1]
        st.session_state.messages.append(ai_msg)
        save_history(st.session_state.messages)
