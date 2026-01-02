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
        # 定义允许在 exec 环境中使用的库
        self.global_context = {
            "pd": pd,
            "gpd": gpd,
            "px": px,
            "go": go,
            "print": print
        }

    def _clean_code(self, code: str) -> str:
        """
        清洗代码字符串，移除 Markdown 标记
        """
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def execute(self, code_str: str, data_context: Dict[str, Any]) -> ExecutionResult:
        """
        Args:
            code_str: Python 代码
            data_context: 变量名到 DataFrame 的映射字典
                          例如 {'df_trips': df1, 'df_zones': df2}
        """
        clean_code = self._clean_code(code_str)

        # [修改点]: 直接复制传入的字典作为局部作用域
        # 这样 AI 代码里就可以直接引用 data_context 中的 key (变量名)
        local_scope = data_context.copy()

        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            logger.info(f"Executing code snippet with context keys: {list(data_context.keys())}")

            # exec 在 global_context (库) 和 local_scope (数据变量) 中运行
            exec(clean_code, self.global_context, local_scope)

            sys.stdout = old_stdout

            # 检查结果变量 'fig'
            if "fig" in local_scope:
                return ExecutionResult(success=True, result=local_scope["fig"], code=clean_code)
            else:
                return ExecutionResult(
                    success=False,
                    error="Code executed successfully, but no 'fig' variable was assigned.",
                    code=clean_code
                )

        except Exception:
            sys.stdout = old_stdout
            error_trace = traceback.format_exc()
            return ExecutionResult(success=False, error=error_trace, code=clean_code)


# --- 单元测试 ---
if __name__ == "__main__":
    print("=== Testing Code Executor (Multi-Context Support) ===")

    executor = CodeExecutor()

    # 1. 创建模拟数据
    data_a = {
        "category": ["A", "B", "C"],
        "value": [10, 20, 15]
    }
    df_mock = pd.DataFrame(data_a)

    # 模拟多文件场景：创建第二个 DataFrame
    data_b = {
        "category": ["A", "B", "C"],
        "info": ["Info1", "Info2", "Info3"]
    }
    df_info = pd.DataFrame(data_b)

    # 2. 测试用例 A: 单变量兼容性测试
    print("\n[Test A] Single Variable Execution")
    valid_code = """
fig = px.bar(df, x='category', y='value', title='Test Bar Chart')
    """
    # 注意：现在必须传入字典 {'df': df_mock}
    res_a = executor.execute(valid_code, {"df": df_mock})

    if res_a.success:
        print("✅ Success! Got object type:", type(res_a.result))
    else:
        print("❌ Failed:", res_a.error)

    # 3. 测试用例 B: 错误捕获
    print("\n[Test B] Broken Code Execution")
    broken_code = "fig = px.bar(df, x='category', y='wrong_column')"
    res_b = executor.execute(broken_code, {"df": df_mock})

    if not res_b.success:
        print("✅ Correctly caught error!")
    else:
        print("❌ Should have failed.")

    # 4. [新增] 测试用例 D: 多变量协同测试
    print("\n[Test D] Multi-Variable Execution (Merge scenario)")

    # 模拟 AI 生成的代码：使用两个变量 df_main 和 df_desc
    multi_var_code = """
# 模拟关联操作
merged_df = pd.merge(df_main, df_desc, on='category')
fig = px.bar(merged_df, x='category', y='value', hover_data=['info'], title='Merged Data Plot')
    """

    # 构造包含两个 DataFrame 的上下文
    context = {
        "df_main": df_mock,
        "df_desc": df_info
    }

    res_d = executor.execute(multi_var_code, context)

    if res_d.success:
        print("✅ Multi-variable Execution Success!")
        print(f"   Context used: {list(context.keys())}")
    else:
        print("❌ Multi-variable Failed:", res_d.error)