import logging
from typing import Dict, Any, Optional
from core.llm.ollama_client import LocalLlamaClient

logger = logging.getLogger(__name__)


class CodeGenerator:
    def __init__(self, llm_client: LocalLlamaClient):
        self.llm = llm_client

    def _build_system_prompt(self, meta: Dict[str, Any]) -> str:
        """
        构建系统提示词，注入数据上下文和编码约束。
        """
        # 1. 提取列信息
        columns_info = []
        # 兼容 Phase 1 的输出结构
        stats = meta.get("basic_stats", {}).get("column_stats", {})
        semantic_tags = meta.get("semantic_analysis", {}).get("semantic_tags", {})

        for col, info in stats.items():
            # 获取该列的语义标签 (例如 ST_TIME, BIZ_METRIC)
            tag = semantic_tags.get(col, "UNKNOWN")
            dtype = info.get('dtype', 'object')
            columns_info.append(f"- {col} (Type: {dtype}, Semantic: {tag})")

        columns_text = "\n".join(columns_info)

        # 2. 核心 Prompt (Prompt Engineering)
        system_prompt = f"""
        You are an expert Python Data Analyst. Your goal is to write Python code to visualize data based on user queries.

        === DATASET CONTEXT ===
        The data is loaded into a pandas DataFrame named `df`.
        Columns available in `df`:
        {columns_text}

         === CONSTRAINTS (MUST FOLLOW) ===
        1. Library: Use ONLY `plotly.express` (as px).
        2. Input: Use the variable `df` directly.
        3. Output: Assign final figure to `fig`.
        
        4. [IMPORTANT] Big Data Handling:
           - The `df` might contain MILLIONS of rows.
           - FOR AGGREGATION (histogram, bar, line, groupby): Use FULL data. Do NOT sample.
             Example: `fig = px.bar(df.groupby('zone').size()...)` is SAFE.
           - FOR SCATTER/MAPS (scatter, scatter_mapbox): 
             Check data size. If rows > 20000, YOU MUST SAMPLE.
             Example: 
             ```python
             if len(df) > 20000:
                 df_plot = df.sample(20000)
             else:
                 df_plot = df
             fig = px.scatter(df_plot, ...)
             ```
        5. Geo-Data:
           - If the user asks for a map and you have LAT/LON columns, use `px.scatter_mapbox`.
           - Set `mapbox_style="open-street-map"` or "carto-positron".
        6. Formatting: Return ONLY the raw Python code inside a markdown block. NO explanations.

        === EXAMPLE ===
        User: "Show me the distribution of fare amount"
        Response:
        ```python
        import plotly.express as px
        fig = px.histogram(df, x='fare_amount', title='Distribution of Fare Amount')
        ```
        """
        return system_prompt

    def generate_code(self, query: str, context_summary: Dict[str, Any]) -> str:
        """
        根据用户查询和数据摘要生成代码
        """
        system_prompt = self._build_system_prompt(context_summary)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Query: {query}"}
        ]

        logger.info(f"Generating code for query: '{query}'...")

        # 调用 LLM (非 JSON 模式，我们需要代码文本)
        response_text = self.llm.chat(messages, json_mode=False)

        return response_text

    # === [新增] 自愈修复方法 ===
    def fix_code(self, original_code: str, error_trace: str, context_summary: Dict[str, Any]) -> str:
        """
        当代码执行出错时调用此方法。
        将 错误信息 + 原始代码 发回给 AI 进行修复。
        """
        system_prompt = self._build_system_prompt(context_summary)

        # 构造一个“修复请求”的 Prompt
        fix_prompt = f"""
        The code you generated previously failed to execute.

        === BROKEN CODE ===
        {original_code}

        === ERROR TRACEBACK ===
        {error_trace}

        === INSTRUCTION ===
        1. Analyze the error message to understand what went wrong (e.g., wrong column name, syntax error, library misuse).
        2. Fix the code.
        3. Return ONLY the fixed Python code block.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": fix_prompt}
        ]

        logger.warning(f"Attempting to FIX code based on error...")
        return self.llm.chat(messages, json_mode=False)


# --- 单元测试 ---
if __name__ == "__main__":
    # 模拟 Phase 1 的输出结果
    mock_summary = {
        "basic_stats": {
            "column_stats": {
                "tpep_pickup_datetime": {"dtype": "datetime64[ns]"},
                "fare_amount": {"dtype": "float64"},
                "passenger_count": {"dtype": "int64"},
                "PULocationID": {"dtype": "int64"}
            }
        },
        "semantic_analysis": {
            "semantic_tags": {
                "tpep_pickup_datetime": "ST_TIME",
                "fare_amount": "BIZ_PRICE",
                "passenger_count": "BIZ_METRIC",
                "PULocationID": "ST_LOC_ID"
            }
        }
    }

    print("=== Testing Code Generator ===")

    # 1. 初始化
    try:
        client = LocalLlamaClient()
        generator = CodeGenerator(client)

        # 2. 测试简单查询
        query = "Plot a histogram of the fare amount"
        print(f"\nUser Query: {query}")

        code = generator.generate_code(query, mock_summary)

        print("\n[AI Generated Code]:")
        print(code)

        # 3. 简单的校验
        if "px.histogram" in code and "fig =" in code:
            print("\n✅ Valid code structure detected.")
        else:
            print("\n❌ Code structure might be wrong.")

    except Exception as e:
        print(f"Test failed: {e}")