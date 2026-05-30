from langchain_deepseek import ChatDeepSeek
from config.settings import *

def create_llm():
    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        temperature=0.1,
        max_tokens=1000,
        timeout=30,
        max_retries=3,
        api_key=DEEPSEEK_API_KEY,
        extra_body={"thinking": {"type": "disabled"}},
    )