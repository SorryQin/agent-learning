import json
import os
from datetime import datetime

from langchain_core.messages import HumanMessage

from config.settings import LONG_TERM_MEMORY_FILE
from models.llm import create_llm

DEFAULT_MEMORY = {
    "profile": {
        "skill_level": "unknown",
        "learning_style": "unknown",
        "strengths": [],
        "weaknesses": [],
        "common_mistakes": [],
    },
    "topics": [],
    "session_count": 0,
    "current_focus": "",
}

def load() -> dict:
    """读取长期记忆文件"""
    if not os.path.exists(LONG_TERM_MEMORY_FILE):
        print(f"LONG_TERM_MEMORY_FILE path not exist: {LONG_TERM_MEMORY_FILE}")
        return dict(DEFAULT_MEMORY)
    with open(LONG_TERM_MEMORY_FILE, "r", encoding="utf-8") as f:
        content = f.read()
        if not content.strip():
            return dict(DEFAULT_MEMORY)
        return json.loads(content)

def save(memory: dict):
    """写入长期记忆文件，自动更新最后修改时间"""
    memory["last_updated"] = datetime.now().isoformat()
    with open(LONG_TERM_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def to_prompt(memory: dict):
    """把长期记忆格式化成一段文本，拼到 system prompt 里交给 LLM"""
    profile = memory["profile"]
    lines = ["##学员档案"]
    lines.append(f"熟练度：{profile.get('skill_level', 'unknown')}")
    if profile.get("strengths"):
        lines.append(f"掌握较好的：{'、'.join(profile['strengths'])}")
    if profile.get("weaknesses"):
        lines.append(f"需要加强的：{'、'.join(profile['weaknesses'])}")
    if profile.get("common_mistakes"):
        lines.append(f"常犯错误：{'、'.join(profile['common_mistakes'])}")
    lines.append(f"已进行 {memory.get('session_count', 0)} 次教学")
    if memory.get("current_focus"):
        lines.append(f"当前学习重点：{memory['current_focus']}")
    return "\n".join(lines)

EXTRACTION_PROMPT = """你是一位教学分析师。请分析上面AI私教与本轮学员的对话，更新学员档案。

要求：只返回JSON，不要包含其他文字。
不要编造信息，如果本轮对话没有足够信息判断，就保持原有值。

{
  "skill_level": "beginner | intermediate | advanced | unknown",
  "learning_style": "theoretical | hands-on | unknown",
  "strengths": ["掌握较好的主题"],
  "weaknesses": ["需要加强的主题"],
  "common_mistakes": ["常犯错误描述"],
  "new_topics_found": ["本轮识别到的新主题"],
  "current_focus": "当前学习重点"
}"""

def update(memory: dict, messages: list) -> dict:
    """用 LLM 分析本轮对话，提取更新并合并到记忆"""
    llm = create_llm()
    dialogue_parts = []
    for msg in messages:
        role = "human" if msg.type == "human" else "ai"
        text = (msg.content or "")[:800]
        if text:
            dialogue_parts.append(f"{role}: {text}")

    dialogue_text = "\n\n".join(dialogue_parts)

    prompt = (
        f"当前学员的档案:\n"
        f"{json.dumps(memory, ensure_ascii=False, indent=2)}\n\n"
        f"本轮对话:\n{dialogue_text}\n\n"
        f"{EXTRACTION_PROMPT}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        updates = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        print(f"长期记忆更新失败：llm解析失败\n")
        return memory
    
    # merge memory
    profile = memory["profile"]
    profile["skill_level"] = updates.get("skill_level", profile["skill_level"])
    profile["learning_style"] = updates.get("learning_style", profile["learning_style"])
    profile["strengths"] = updates.get("strengths", profile["strengths"])
    profile["weaknesses"] = updates.get("weaknesses", profile["weaknesses"])
    profile["common_mistakes"] = updates.get("common_mistakes", profile["common_mistakes"])
    memory["current_focus"] = updates.get("current_focus", memory["current_focus"])

    for topic in updates.get("new_topics_found", []):
        if topic and topic not in memory["topics"]:
            memory["topics"].append(topic)

    return memory