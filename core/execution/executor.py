import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sys
import io
import re
import textwrap
from typing import Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExecutionResult:
    def __init__(self, success: bool, result: Any = None, error: str = None, code: str = ""):
        self.success = success
        self.result = result
        self.error = error
        self.code = code


class CodeExecutor:
    def __init__(self):
        self.global_context = {
            "pd": pd,
            "gpd": gpd,
            "px": px,
            "go": go,
            "print": print
        }

    def _clean_code(self, text: str) -> str:
        """
        从 LLM 回复中精准提取 Python 代码块，并自动去除缩进。
        """
        # 匹配 ```python 后的内容，但非贪婪地处理首行换行
        # (?:[^\n]*\n) 匹配 ```python 后面的任意字符直到换行符（不吞噬下一行的缩进）
        pattern_python = r'```python(?:[^\n]*\n)(.*?)```'

        # 通用匹配
        pattern_generic = r'```(?:[^\n]*\n)(.*?)```'

        code_block = None

        # 优先寻找 ```python
        # re.DOTALL 让 . 匹配换行符
        matches = re.findall(pattern_python, text, re.DOTALL | re.IGNORECASE)
        if matches:
            code_block = matches[-1]

        # 其次寻找 ```
        elif re.findall(pattern_generic, text, re.DOTALL | re.IGNORECASE):
            code_block = re.findall(pattern_generic, text, re.DOTALL | re.IGNORECASE)[-1]

        if code_block:
            # 这里的 code_block 保留了第一行的缩进
            # dedent 会统一去除所有行的公共缩进
            return textwrap.dedent(code_block).strip()

        # 如果没找到 Markdown，尝试处理原文本
        return textwrap.dedent(text).strip()

    def execute(self, code_str: str, data_context: Dict[str, Any]) -> ExecutionResult:
        """
        Args:
            code_str: Python 代码
            data_context: 变量名到 DataFrame 的映射字典
        """
        clean_code = self._clean_code(code_str)

        local_scope = {}
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            logger.info("Executing code...")

            # 1. 尝试执行代码定义
            try:
                exec(clean_code, self.global_context, local_scope)
            except SyntaxError as e:
                sys.stdout = old_stdout
                return ExecutionResult(False, error=f"SyntaxError (Check indentation or non-code text): {str(e)}",
                                       code=clean_code)
            except NameError:
                logger.info("NameError detected during definition phase. Treating as Script Mode.")
            except Exception as e:
                raise e

            # 2. 检查 'plot' 函数 (Scaffold)
            if "plot" in local_scope and callable(local_scope["plot"]):
                logger.info("Detected 'plot' function. Executing in Scaffold mode.")
                try:
                    fig = local_scope["plot"](data_context)
                    sys.stdout = old_stdout
                    return ExecutionResult(success=True, result=fig, code=clean_code)
                except Exception as e:
                    sys.stdout = old_stdout
                    return ExecutionResult(False, error=f"Runtime Error inside plot(): {traceback.format_exc()}",
                                           code=clean_code)

            # 3. Fallback: Script Mode
            else:
                logger.info("Function 'plot' not found. Falling back to Script mode.")
                script_scope = data_context.copy()
                exec(clean_code, self.global_context, script_scope)
                sys.stdout = old_stdout

                if "fig" in script_scope:
                    return ExecutionResult(success=True, result=script_scope["fig"], code=clean_code)
                else:
                    return ExecutionResult(
                        success=False,
                        error="Code executed but neither 'plot(data_context)' function nor 'fig' variable was found.",
                        code=clean_code
                    )

        except Exception:
            sys.stdout = old_stdout
            error_trace = traceback.format_exc()
            return ExecutionResult(success=False, error=error_trace, code=clean_code)


# --- 单元测试 ---
if __name__ == "__main__":
    print("=== Testing Code Executor (Fixed Regex) ===")

    executor = CodeExecutor()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    context = {"df": df}

    # Test A: 缩进的代码块 (这是导致之前报错的关键 case)
    print("\n[Test A] Indented Code Block")
    messy_response = """
    Here is the code:

    ```python
    import plotly.express as px
    def plot(data_context):
        df = data_context['df']
        # 第一行 import 和这一行都有 4 空格缩进
        return px.bar(df, x='a', y='b', title='Cleaned Code Chart')
    ```
    """
    res_a = executor.execute(messy_response, context)
    if res_a.success:
        print("✅ Success! Indentation fixed.")
    else:
        print(f"❌ Failed: {res_a.error}")

    # Test B: 多个代码块
    print("\n[Test B] Multiple Code Blocks")
    multi_block_response = """
    Wrong:
    ```python
    def plot(ctx): return 1/0
    ```

    Correct:
    ```python
    import plotly.express as px
    def plot(data_context):
        df = data_context['df']
        return px.scatter(df, x='a', y='b', title='Last Block')
    ```
    """
    res_b = executor.execute(multi_block_response, context)
    if res_b.success and "Last Block" in res_b.result.layout.title.text:
        print("✅ Success! Used the last code block.")
    else:
        print(f"❌ Failed: {res_b.error}")