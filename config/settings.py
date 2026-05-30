from dotenv import load_dotenv
import os

load_dotenv()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

ALLOWED_DIR = "/Users/sorryqin/Desktop/agent/qzr/0527/project"
ALLOWED_EXT = {".py", ".txt", ".md"}
MAX_SIZE = 100 * 1024
MAX_TOOL_CALLS = 5

HISTORY_FILE = "/Users/sorryqin/Desktop/agent/qzr/0527/project/storage/chat_history.json"
LONG_TERM_MEMORY_FILE = "/Users/sorryqin/Desktop/agent/qzr/0527/project/storage/long_term_memory.json"