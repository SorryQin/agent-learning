import re

from pydantic import BaseModel, Field, field_validator


# 禁止的危险模式（正则列表）
DANGEROUS_PATTERNS: list[re.Pattern] = [
    # 文件删除/破坏
    re.compile(r"\bos\.(system|popen|remove|unlink|rmdir|chmod)\s*\("),
    re.compile(r"\bshutil\.(rmtree|move|chown|chmod)\s*\("),
    re.compile(r"\bsubprocess\."),
    re.compile(r"\bpathlib\.Path\(.*?\)\.(unlink|rmdir)\s*\)"),
    # 动态执行（绕过沙箱）
    re.compile(r"\b__import__\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bcompile\s*\("),
    # 阻塞/耗尽型
    re.compile(r"\bwhile\s+True\b"),  # 无限循环（配合 pids_limit 已防 fork 炸弹）
]


class RunPythonCodeSchema(BaseModel):
    """运行 Python 代码时的输入参数"""
    code: str = Field(
        description="要执行的 Python 代码字符串，必须是完整可运行的纯代码"
    )

    @field_validator("code")
    @classmethod
    def check_dangerous_code(cls, v: str) -> str:
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(v):
                raise ValueError(
                    f"检测到危险代码模式「{pattern.pattern}」，已阻止执行，请修改后重试"
                )
        return v


class LoadCodeFileSchema(BaseModel):
    """读取代码文件时的输入参数"""
    file_path: str = Field(
        description="要读取的文件路径（绝对路径或相对于项目的相对路径）"
    )
