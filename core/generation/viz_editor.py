import logging
from typing import Dict, Any, List
from core.llm.AI_client import AIClient
from core.generation.scaffold import STChartScaffold

logger = logging.getLogger(__name__)


class VizEditor:
    """
    Visualization Editor: 负责基于用户指令修改现有的可视化代码。
    参考 LIDA 的设计，专注于 Code-to-Code 的转换。
    """

    def __init__(self, aIClient: AIClient):
        self.llm = aIClient
        self.scaffold = STChartScaffold()

    def edit_code(self, original_code: str, query: str, summaries: List[Dict[str, Any]]) -> str:
        """
        Args:
            original_code: 上一次生成的 Python 代码
            query: 用户的修改指令 (例如 "把颜色改成红色")
            summaries: 数据摘要 (用于上下文理解)
        """

        # 1. 获取基础上下文 (复用 Scaffold 以保持 GIS 规则一致性)
        # 我们只需要其中的 GIS 规则部分，或者完整的 system prompt
        base_system_prompt = self.scaffold.get_system_prompt(summaries)

        # 2. 构建专门针对 "编辑" 的 System Prompt
        editor_system_prompt = f"""
        {base_system_prompt}

        === ROLE ===
        You are a Senior Visualization Engineer. 
        Your task is to MODIFY the provided Python code based on the user's instructions.

        === RULES ===
        1. **Keep the Structure**: You MUST preserve the `def plot(data_context):` function signature.
        2. **Incremental Change**: Only modify the parts relevant to the user's request (e.g., changing colors, titles, filtering data). Do not rewrite the whole logic unless necessary.
        3. **Data Context**: Continue to use `data_context['var_name']` to access data. DO NOT introduce mock data.
        4. **GIS Constraints**: If the user asks to filter/change map data, REMEMBER the GIS rules (WGS84 CRS, Mapbox style) defined above.
        5. **Output**: Return the FULLY functional modified code block.
        """

        # 3. 构建用户输入
        user_prompt = f"""
        === ORIGINAL CODE ===
        {original_code}

        === USER INSTRUCTION ===
        "{query}"

        === TASK ===
        Apply the instruction to the original code. 
        Return the complete modified Python code block (including imports).
        """

        messages = [
            {"role": "system", "content": editor_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        logger.info(f"Editing code based on query: '{query}'...")

        # 4. 调用 LLM
        response_text = self.llm.chat(messages, json_mode=False)
        return response_text