import os

from langchain_core.tools import StructuredTool

from config.settings import ALLOWED_DIR, ALLOWED_EXT, MAX_SIZE
from tools.sandbox import sandbox
from tools.schemas import RunPythonCodeSchema, LoadCodeFileSchema


def _run_python_code(code: str) -> str:
    """在 Docker 沙箱中执行 Python 代码并返回输出。"""
    print("running code in sandbox now...")
    return sandbox.run(code)


def _load_code_file(file_path: str) -> str:
    """从白名单目录内读取允许类型的文件内容。"""
    try:
        real_path = os.path.realpath(file_path)
        real_root = os.path.realpath(ALLOWED_DIR)
        if os.path.commonpath([real_path, real_root]) != real_root:
            return "拒绝访问：路径不在允许目录内"

        if not os.path.isfile(real_path):
            return "文件不存在或不是普通文件"

        if os.path.splitext(real_path)[1].lower() not in ALLOWED_EXT:
            return "拒绝访问：文件类型不允许"

        if os.path.getsize(real_path) > MAX_SIZE:
            return f"文件过大（>{MAX_SIZE} 字节）"

        with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败: {str(e)}"


run_python_code = StructuredTool.from_function(
    func=_run_python_code,
    name="run_python_code",
    description="当需要运行或测试 Python 代码时使用此工具。输入必须是纯代码字符串",
    args_schema=RunPythonCodeSchema,
)

load_code_file = StructuredTool.from_function(
    func=_load_code_file,
    name="load_code_file",
    description="从指定文件读取代码内容。仅支持读取项目目录内的 .py / .txt / .md 文件",
    args_schema=LoadCodeFileSchema,
)

tools = [run_python_code, load_code_file]