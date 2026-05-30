from config.settings import HISTORY_FILE
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os

def messages_to_dict(messages):
    data = []
    for message in messages:
        if hasattr(message, "content"):
            data.append({
                "type": type(message).__name__.replace("Message", "").lower(),
                "content": message.content
            })

    return data

TYPE_MAP = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
}

def dict_to_messages(data):
    messages = []
    for item in data:
        cls = TYPE_MAP.get(item["type"])
        if cls:
            messages.append(cls(content = item["content"]))
    
    return messages

def sliding_window(messages, max_rounds=5):
    system = messages[:1]
    history = messages[1:]
    window = history[-(max_rounds * 2):]
    return system + window

def save_history(messages):
    data = messages_to_dict(messages)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return dict_to_messages(data)