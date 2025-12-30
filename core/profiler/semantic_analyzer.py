import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

# 引入我们之前写好的模块
# 注意：如果运行时提示 ModuleNotFoundError，请确保在项目根目录下运行，或设置 PYTHONPATH
try:
    from core.ingestion.loader_factory import LoaderFactory
    from core.llm.ollama_client import LocalLlamaClient
    from core.profiler.basic_stats import get_dataset_fingerprint
except ImportError:
    # 兼容直接运行此脚本时的路径问题
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from core.ingestion.loader_factory import LoaderFactory
    from core.llm.ollama_client import LocalLlamaClient
    from core.profiler.basic_stats import get_dataset_fingerprint

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    def __init__(self, llm_client: LocalLlamaClient):
        self.llm = llm_client

    def _build_prompt(self, filename: str, fingerprint: Dict[str, Any]) -> str:
        """
        构建发送给 Llama 3.1 的 Prompt (经过幻觉抑制优化)。
        """
        # 1. 构建列的详细描述清单
        columns_summary = []
        column_names_list = []  # 记录原始列名，用于后续校验

        for col, info in fingerprint["column_stats"].items():
            column_names_list.append(col)
            desc = f"Column: '{col}' | Type: {info['dtype']} | Samples: {info['samples']}"
            if "min" in info:
                desc += f" | Range: {info['min']:.2f} to {info['max']:.2f}"
            if "geom_type" in info:
                desc += f" | Geometry Type: {info['geom_type']}"
            columns_summary.append(desc)

        columns_text = "\n".join(columns_summary)

        # 2. 定义更加精准的语义标签集
        allowed_tags = """
        - ST_TIME: Timestamp, date, datetime (e.g., pickup_time)
        - ST_LOC_ID: Location IDs, Zone IDs (e.g., PULocationID)
        - ST_GEO: Geometry column (WKT, geometry objects)
        - ST_LAT: Latitude coordinates
        - ST_LON: Longitude coordinates
        - BIZ_METRIC: Numerical measures (distance, passenger_count, speed)
        - BIZ_PRICE: Monetary values (fare, total_amount)
        - BIZ_CAT: Categorical data (VendorID, payment_type, Zone Names)
        - ID_KEY: Primary keys or foreign keys (row_id)
        - OTHER: Anything else
        """

        # 3. 强约束 Prompt
        prompt = f"""
        You are a Spatial Data Expert. I need you to analyze the schema of a dataset.

        === DATASET METADATA ===
        File Name: "{filename}"
        Rows: {fingerprint['rows']}
        CRS: {fingerprint.get('crs', 'N/A')}

        === ACTUAL COLUMNS (Use ONLY these names as keys) ===
        {columns_text}

        === INSTRUCTIONS ===
        1. Analyze the 'ACTUAL COLUMNS' list above.
        2. Map EACH column name to exactly one semantic tag from the list below:
        {allowed_tags}

        3. Determine the 'dataset_type':
           - TRAJECTORY: Has time + space (points/lines) + metrics.
           - GEO_ZONE: Has polygons/multipolygons (reference map).
           - LOOKUP_TABLE: Has IDs and Names but no geometry/coordinates.

        4. Strict JSON Output Rules:
           - The keys in "semantic_tags" MUST be the exact column names from the input.
           - DO NOT invent new columns.
           - DO NOT use the tags as keys.

        === RESPONSE FORMAT (JSON ONLY) ===
        {{
            "dataset_type": "TRAJECTORY/GEO_ZONE/LOOKUP_TABLE",
            "description": "Short summary",
            "semantic_tags": {{
                "{column_names_list[0]}": "TAG",
                "{column_names_list[1]}": "TAG"
            }}
        }}
        """
        return prompt

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        主入口：分析文件并返回增强后的元数据
        """
        logger.info(f"Starting semantic analysis for: {file_path}")

        try:
            loader = LoaderFactory.get_loader(file_path)

            # Action 1: 获取真实的行数 (全量扫描/元数据读取)
            real_row_count = loader.count_rows(file_path)

            # Action 2: 获取样本数据 (只取前 10 行给 AI 看)
            df_preview = loader.peek(file_path, n=10)

        except Exception as e:
            logger.error(f"Loader failed: {e}")
            return {"error": f"Failed to load file: {str(e)}"}

        # 3. 计算基础统计指纹
        try:
            fingerprint = get_dataset_fingerprint(df_preview)

            # [关键修正]：用真实行数覆盖样本行数！
            fingerprint['rows'] = real_row_count

        except Exception as e:
            logger.error(f"Fingerprinting failed: {e}")
            return {"error": f"Failed to generate stats: {str(e)}"}

        # 4. 构建 Prompt (后续逻辑不变...)
        filename = Path(file_path).name
        prompt = self._build_prompt(filename, fingerprint)

        # 5. 调用 LLM 获取语义标签
        try:
            print(f"   >>> Sending metadata of [{filename}] to Llama 3.1...")
            ai_result = self.llm.query_json(
                prompt=prompt,
                system_prompt="You are a data analysis assistant that outputs only valid JSON."
            )
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            ai_result = {
                "dataset_type": "UNKNOWN",
                "description": "AI analysis failed.",
                "semantic_tags": {}
            }

        # 6. 合并结果
        final_summary = {
            "file_info": {
                "path": str(file_path),
                "name": filename
            },
            "basic_stats": fingerprint,
            "semantic_analysis": ai_result
        }

        return final_summary


# --- 实战测试部分 ---
if __name__ == "__main__":
    # 设置日志级别以便观察过程
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. 自动定位 data 目录
    # 假设结构是 NL-STV/core/profiler/semantic_analyzer.py
    # data 目录在 NL-STV/data
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"

    print(f"🔍 Searching for data files in: {data_dir}")

    # 2. 扫描目录下常见格式的文件
    target_extensions = ['*.csv', '*.parquet', '*.shp']
    found_files = []
    for ext in target_extensions:
        found_files.extend(list(data_dir.glob(ext)))

    if not found_files:
        print("❌ No files found in data directory! Please check your path.")
        exit()

    # 3. 初始化 AI 客户端
    print("🔌 Connecting to Local Ollama...")
    try:
        client = LocalLlamaClient(model_name="llama3.1:latest")
        analyzer = SemanticAnalyzer(client)
    except Exception as e:
        print(f"❌ Failed to init Ollama: {e}")
        exit()

    # 4. 遍历文件进行分析
    print(f"✅ Found {len(found_files)} files. Starting Batch Analysis...\n")

    for file_path in found_files:
        print(f"--------------------------------------------------")
        print(f"📂 Processing: {file_path.name}")

        # 执行分析
        result = analyzer.analyze(str(file_path))

        # 检查是否出错
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue

        # 提取关键信息进行展示
        ai_output = result.get("semantic_analysis", {})
        dataset_type = ai_output.get("dataset_type", "UNKNOWN")
        desc = ai_output.get("description", "No description")
        tags = ai_output.get("semantic_tags", {})

        print(f"🤖 AI Assessment:")
        print(f"   - Type: \033[92m{dataset_type}\033[0m")  # 绿色高亮
        print(f"   - Summary: {desc}")
        print(f"   - Column Mapping:")
        for col, role in tags.items():
            print(f"     * {col:<25} -> {role}")

        print(f"\n✅ Analysis Complete for {file_path.name}")

    print("\n--------------------------------------------------")
    print("🎉 All files processed.")