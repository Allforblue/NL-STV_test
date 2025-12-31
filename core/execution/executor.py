import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sys
import io
from typing import Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExecutionResult:
    """定义执行结果的标准返回格式"""

    def __init__(self, success: bool, result: Any = None, error: str = None, code: str = ""):
        self.success = success
        self.result = result  # 这里的 result 通常是 Plotly Figure 对象
        self.error = error  # 错误堆栈信息
        self.code = code  # 实际执行的代码


class CodeExecutor:
    def __init__(self):
        # 定义允许在 exec 环境中使用的库 (白名单机制建议在未来加强)
        self.global_context = {
            "pd": pd,
            "gpd": gpd,
            "px": px,
            "go": go,
            "print": print
        }

    def _clean_code(self, code: str) -> str:
        """
        清洗代码字符串，移除 Markdown 标记 (```python ... ```)
        """
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def execute(self, code_str: str, df: Any) -> ExecutionResult:
        """
        执行代码的核心方法。

        Args:
            code_str: Python 代码字符串
            df: 当前的数据集 (DataFrame 或 GeoDataFrame)，将以变量名 'df' 注入环境

        Returns:
            ExecutionResult 对象
        """
        # 1. 清洗代码
        clean_code = self._clean_code(code_str)

        # 2. 准备执行上下文 (Local Scope)
        # 我们强制约定：数据变量名为 'df'
        local_scope = {"df": df}

        # 3. 捕获标准输出 (可选，用于调试 print 语句)
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            logger.info("Executing code snippet...")
            # --- 核心执行 ---
            # exec() 在 global_context (库) 和 local_scope (数据) 中运行
            exec(clean_code, self.global_context, local_scope)
            # --------------

            sys.stdout = old_stdout  # 还原 stdout

            # 4. 提取结果
            # 我们强制约定：AI 生成的代码必须将图表赋值给变量 'fig'
            if "fig" in local_scope:
                fig_obj = local_scope["fig"]
                return ExecutionResult(success=True, result=fig_obj, code=clean_code)
            else:
                return ExecutionResult(
                    success=False,
                    error="Code executed successfully, but no variable named 'fig' was generated.",
                    code=clean_code
                )

        except Exception:
            sys.stdout = old_stdout  # 确保出错也能还原 stdout

            # 5. 捕获详细的错误堆栈
            # 这是实现 Self-Healing 的关键，我们需要把这个报错回传给 AI
            error_trace = traceback.format_exc()
            logger.error(f"Execution failed:\n{error_trace}")

            return ExecutionResult(success=False, error=error_trace, code=clean_code)


# --- 单元测试 ---
if __name__ == "__main__":
    print("=== Testing Code Executor ===")

    # 1. 创建模拟数据
    data = {
        "category": ["A", "B", "C", "A", "B"],
        "value": [10, 20, 15, 25, 30],
        "lat": [40.71, 40.72, 40.73, 40.74, 40.75],
        "lon": [-74.00, -74.01, -74.02, -74.03, -74.04]
    }
    df_mock = pd.DataFrame(data)
    executor = CodeExecutor()

    # 2. 测试用例 A: 成功的绘图代码
    print("\n[Test A] Valid Code Execution")
    valid_code = """
import plotly.express as px
# 绘制柱状图
fig = px.bar(df, x='category', y='value', title='Test Bar Chart')
    """
    res_a = executor.execute(valid_code, df_mock)
    if res_a.success:
        print("✅ Success! Got object type:", type(res_a.result))
        # res_a.result.show() # 如果在本地运行，取消注释可以看到图
    else:
        print("❌ Failed:", res_a.error)

    # 3. 测试用例 B: 错误的代码 (测试报错捕获)
    print("\n[Test B] Broken Code Execution")
    broken_code = """
# 这是一个错误的列名 'values' (应该是 'value')
fig = px.bar(df, x='category', y='values')
    """
    res_b = executor.execute(broken_code, df_mock)
    if not res_b.success:
        print("✅ Correctly caught error!")
        print("Error Snippet:", res_b.error.split('\n')[-2])  # 打印最后一行报错
    else:
        print("❌ Should have failed but succeeded.")

    # 4. 测试用例 C: 忘记赋值给 fig
    print("\n[Test C] Missing 'fig' variable")
    no_fig_code = """
chart = px.bar(df, x='category', y='value')
# 忘记写 fig = chart
    """
    res_c = executor.execute(no_fig_code, df_mock)
    if not res_c.success and "no variable named 'fig'" in res_c.error:
        print("✅ Correctly detected missing output variable.")
    else:
        print("❌ Failed validation.")