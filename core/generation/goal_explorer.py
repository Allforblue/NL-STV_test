import logging
import json
from typing import Dict, Any, List
from core.llm.AI_client import AIClient

logger = logging.getLogger(__name__)


class GoalExplorer:
    def __init__(self, llm_client: AIClient):
        self.llm = llm_client

    def generate_goals(self, summary: Dict[str, Any], n: int = 4) -> List[str]:
        """
        根据数据摘要生成推荐的分析目标。
        """
        # 提取关键信息
        tags = summary.get("semantic_analysis", {}).get("semantic_tags", {})
        file_name = summary.get("file_info", {}).get("name", "data")

        # 简单的启发式规则：提取可用于分析的列
        metrics = [col for col, tag in tags.items() if tag in ["BIZ_METRIC", "BIZ_PRICE"]]
        cats = [col for col, tag in tags.items() if tag in ["BIZ_CAT", "ST_LOC_ID"]]
        times = [col for col, tag in tags.items() if tag == "ST_TIME"]

        # 构建 Prompt
        prompt = f"""
        You are a Data Strategy Consultant. 
        I have a dataset named "{file_name}" with the following key columns:
        - Metrics (Numbers): {metrics}
        - Categories (Groups): {cats}
        - Time: {times}

        Your task is to suggest {n} insightful visualization goals for a user.

        Rules:
        1. Goals must be questions or instructions (e.g., "Show distribution of...", "Plot X vs Y").
        2. Focus on relationships between Metrics and Categories, or Time trends.
        3. Keep them concise (under 10 words).
        4. Return ONLY a valid JSON list of strings.

        Example Output:
        ["Plot histogram of fare_amount", "Show monthly trend of trips", "Scatter plot distance vs price"]
        """

        try:
            logger.info("Generating analysis goals...")
            # 调用 LLM (JSON 模式)
            response = self.llm.query_json(prompt, system_prompt="You output strictly JSON lists.")

            # 兼容性处理：如果返回的是字典 {"goals": [...] }
            if isinstance(response, dict):
                # 尝试寻找列表类型的 value
                for val in response.values():
                    if isinstance(val, list):
                        return val
                return []
            elif isinstance(response, list):
                return response
            else:
                return []

        except Exception as e:
            logger.error(f"Goal exploration failed: {e}")
            # 兜底策略：如果没有 LLM 或报错，返回一些硬编码的建议
            fallback = []
            if metrics: fallback.append(f"Show distribution of {metrics[0]}")
            if len(metrics) >= 2: fallback.append(f"Scatter plot of {metrics[0]} vs {metrics[1]}")
            if times: fallback.append("Show trends over time")
            return fallback


# --- 单元测试 ---
if __name__ == "__main__":
    # 模拟数据
    mock_summary = {
        "file_info": {"name": "taxi_data.csv"},
        "semantic_analysis": {
            "semantic_tags": {
                "tpep_pickup_datetime": "ST_TIME",
                "fare_amount": "BIZ_PRICE",
                "trip_distance": "BIZ_METRIC",
                "payment_type": "BIZ_CAT"
            }
        }
    }

    client = AIClient()
    explorer = GoalExplorer(client)
    goals = explorer.generate_goals(mock_summary)
    print("Suggested Goals:", goals)