import logging
from typing import Dict, Any, List, Optional
from core.llm.ollama_client import LocalLlamaClient

logger = logging.getLogger(__name__)


class CodeGenerator:
    def __init__(self, llm_client: LocalLlamaClient):
        self.llm = llm_client

    # [修改点 1] 参数改为 summaries 列表
    def _build_system_prompt(self, summaries: List[Dict[str, Any]]) -> str:
        """
        构建系统提示词，支持多文件上下文。
        """

        # 1. 构建每个 DataFrame 的上下文描述
        context_descriptions = []

        for summary in summaries:
            # 获取 app.py 中分配的变量名 (例如 df_trips, df_zones)
            # 如果没有分配，默认为 'df' (兼容单文件模式)
            var_name = summary.get('variable_name', 'df')
            file_name = summary.get('file_info', {}).get('name', 'unknown_file')

            # 提取列信息
            columns_info = []
            stats = summary.get("basic_stats", {}).get("column_stats", {})
            semantic_tags = summary.get("semantic_analysis", {}).get("semantic_tags", {})

            for col, info in stats.items():
                tag = semantic_tags.get(col, "UNKNOWN")
                dtype = info.get('dtype', 'object')
                columns_info.append(f"  - {col} (Type: {dtype}, Semantic: {tag})")

            columns_text = "\n".join(columns_info)

            # 组合单个 DataFrame 的描述
            desc = f"DataFrame `{var_name}` (Source: {file_name}):\n{columns_text}"
            context_descriptions.append(desc)

        full_context_text = "\n\n".join(context_descriptions)

        # 2. 核心 Prompt (Prompt Engineering) - 更新为支持多变量
        system_prompt = f"""
        You are an expert Python Data Analyst. Your goal is to write Python code to visualize data based on user queries.

        === AVAILABLE DATASETS ===
        The following pandas DataFrames are loaded in memory and ready to use:

        {full_context_text}

         === CONSTRAINTS (MUST FOLLOW) ===
        1. Library: Use ONLY `plotly.express` (as px).
        2. Input: Use the variable names provided above directly (e.g., `df_trips`, `df_zones`). 
        3. Multi-File: 
           - You can merge dataframes if needed. Example: `df_merged = pd.merge(df_trips, df_zones, on='LocationID')`.
           - If merging, ensure column names match or use `left_on`/`right_on`.
        4. Output: Assign final figure to `fig`.

        5. [IMPORTANT] Big Data Handling:
           - The DataFrames might contain MILLIONS of rows.
           - FOR AGGREGATION (histogram, bar, line, groupby): Use FULL data. Do NOT sample.
             Example: `fig = px.bar(df.groupby('zone').size()...)` is SAFE.
           - FOR SCATTER/MAPS (scatter, scatter_mapbox): 
             Check data size. If rows > 20000, YOU MUST SAMPLE.
             Example: 
             ```python
             # Assume we are using 'df_main' for plotting
             if len(df_main) > 20000:
                 df_plot = df_main.sample(20000)
             else:
                 df_plot = df_main
             fig = px.scatter(df_plot, ...)
             ```
        6. Geo-Data:
           - If using GeoPandas (DataFrames with 'geometry'), convert to WGS84 before plotting if needed: `df = df.to_crs(epsg=4326)`.
           - For maps, use `px.choropleth_mapbox` or `px.scatter_mapbox`.
           - Set `mapbox_style="open-street-map"` or "carto-positron".
        7. Formatting: Return ONLY the raw Python code inside a markdown block. NO explanations.

        === EXAMPLE (Multi-File) ===
        User: "Join trips with zones and map the count"
        Response:
        ```python
        import plotly.express as px
        import pandas as pd
        # Aggregate trips
        df_counts = df_trips.groupby('PULocationID').size().reset_index(name='count')
        # Merge with geometry
        df_map = pd.merge(df_zones, df_counts, left_on='LocationID', right_on='PULocationID')
        # Plot
        df_map = df_map.to_crs(epsg=4326)
        fig = px.choropleth_mapbox(df_map, geojson=df_map.geometry, locations=df_map.index, color='count', mapbox_style="carto-positron")
        ```
        """
        return system_prompt

    # [修改点 2] 参数类型提示改为 List[Dict]
    def generate_code(self, query: str, summaries: List[Dict[str, Any]]) -> str:
        """
        根据用户查询和数据摘要生成代码
        """
        system_prompt = self._build_system_prompt(summaries)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Query: {query}"}
        ]

        logger.info(f"Generating code for query: '{query}'...")

        # 调用 LLM (非 JSON 模式，我们需要代码文本)
        response_text = self.llm.chat(messages, json_mode=False)

        return response_text

    # [修改点 3] 参数类型提示改为 List[Dict]
    def fix_code(self, original_code: str, error_trace: str, summaries: List[Dict[str, Any]]) -> str:
        """
        当代码执行出错时调用此方法。
        将 错误信息 + 原始代码 发回给 AI 进行修复。
        """
        system_prompt = self._build_system_prompt(summaries)

        # 构造一个“修复请求”的 Prompt
        fix_prompt = f"""
        The code you generated previously failed to execute.

        === BROKEN CODE ===
        {original_code}

        === ERROR TRACEBACK ===
        {error_trace}

        === INSTRUCTION ===
        1. Analyze the error message to understand what went wrong (e.g., wrong column name, syntax error, library misuse, variable name error).
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
    # 模拟 Phase 1 的输出结果 (现在是一个列表)
    mock_summaries = [
        # 文件 1: 订单数据
        {
            "file_info": {"name": "trips.parquet"},
            "variable_name": "df_trips",
            "basic_stats": {
                "column_stats": {
                    "tpep_pickup_datetime": {"dtype": "datetime64[ns]"},
                    "fare_amount": {"dtype": "float64"},
                    "PULocationID": {"dtype": "int64"}
                }
            },
            "semantic_analysis": {
                "semantic_tags": {
                    "tpep_pickup_datetime": "ST_TIME",
                    "fare_amount": "BIZ_PRICE",
                    "PULocationID": "ST_LOC_ID"
                }
            }
        },
        # 文件 2: 区域数据
        {
            "file_info": {"name": "zones.shp"},
            "variable_name": "df_zones",
            "basic_stats": {
                "column_stats": {
                    "LocationID": {"dtype": "int64"},
                    "Zone": {"dtype": "object"},
                    "geometry": {"dtype": "geometry"}
                }
            },
            "semantic_analysis": {
                "semantic_tags": {
                    "LocationID": "ST_LOC_ID",
                    "Zone": "BIZ_CAT",
                    "geometry": "ST_GEO"
                }
            }
        }
    ]

    print("=== Testing Code Generator (Multi-File) ===")

    # 1. 初始化
    try:
        client = LocalLlamaClient()
        generator = CodeGenerator(client)

        # 2. 测试复杂查询 (涉及关联)
        query = "Join the trips and zones on LocationID and plot the average fare per zone"
        print(f"\nUser Query: {query}")

        code = generator.generate_code(query, mock_summaries)

        print("\n[AI Generated Code]:")
        print(code)

        # 3. 简单的校验
        if "pd.merge" in code and "fig =" in code:
            print("\n✅ Valid merge logic detected.")
        else:
            print("\n⚠️ Code might be missing merge logic (Check output).")

    except Exception as e:
        print(f"Test failed: {e}")