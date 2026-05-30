import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from config.settings import *
from models.llm import create_llm
from tools.python_tools import tools
from memory.summarizer import dict_to_messages, messages_to_dict, sliding_window, save_history, load_history
from agent.graph import build_tutor_graph
from memory.long_term import load as load_ltm, save as save_ltm, to_prompt as ltm_to_prompt, update as update_ltm
from observability.tracer import Tracer

tool_node = ToolNode(tools)
llm = create_llm()
llm_with_tools = llm.bind_tools(tools)
trace = Tracer()

def on_token(text):
    print(text, end="", flush=True)

def main():
    print("=" * 50)
    print("welcome to AI编程私教")
    print("=" * 50)

    # 初始化状态
    if os.path.exists(HISTORY_FILE):
        user_input = input("检测到上次的对话记录，是否继续？(y/n): ").strip().lower()
        if user_input == "y":
            messages = load_history()
            print(f"已从 {HISTORY_FILE} 恢复对话")
        else:
            os.remove(HISTORY_FILE)
            messages = []
    else:
        messages = []

    ltm = load_ltm()
    ltm["session_count"] += 1

    state = {
        "messages": messages,
        "next_step": "tutor",
        "tool_call_count": 0,
        "profile_prompt": ltm_to_prompt(ltm),
        "current_task": ltm.get("current_focus", ""),
    }

    app = build_tutor_graph(llm_with_tools, tool_node, on_token=on_token, tracer=trace)
    state = app.invoke(state)
    # ai_message = state["messages"][-1]
    # print(ai_message.content)

    while True:
        try:
            user_input = input("\n 你：")
            if user_input.lower() in ["exit", "quit"]:
                print("see you~")
                break

            trace.start_turn()

            state["messages"].append(HumanMessage(content=user_input))
            state["next_step"] = "tutor"
            state["tool_call_count"] = 0
            state["profile_prompt"] = ltm_to_prompt(ltm)
            state["current_task"] = ltm.get("current_focus", "")

            print("AI私教: ", end="", flush=True)
            state = app.invoke(state)
            # print(state["messages"][-1].content)
            print()
            save_history(state["messages"])

            ltm = update_ltm(ltm, state["messages"])
            save_ltm(ltm)
            # print(f"[记忆] 学员水平：{ltm['profile']['skill_level']}，擅长：{ltm['profile']['strengths']}")
                
        except KeyboardInterrupt:
            print("对话已中断，再见！")
            break
        except Exception as e:
            print(f"出错了: {e}")
            # 可以选择重置状态或继续
            pass

if __name__ == "__main__":
    main()