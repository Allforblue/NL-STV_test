import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union

# 配置简单的日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalLlamaClient:
    """
    与本地部署的 Ollama 服务进行交互的客户端。
    主要用于处理 Llama 3.1 的 API 请求，特别是 JSON 格式的结构化输出。
    """

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model_name: str = "llama3.1:latest",
                 timeout: int = 120):
        """
        初始化客户端。

        Args:
            base_url: Ollama API 地址，默认为 http://localhost:11434
            model_name: 模型名称，需与 'ollama list' 中的名称一致
            timeout: 请求超时时间（秒），处理大文件摘要时建议设置较长
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.api_chat_endpoint = f"{self.base_url}/api/chat"

        # 初始化检查
        if not self.is_alive():
            logger.warning(f"无法连接到 Ollama 服务 ({self.base_url})。请确保 Ollama 正在运行。")
        else:
            logger.info(f"Ollama 服务连接成功，使用模型: {self.model_name}")

    def is_alive(self) -> bool:
        """检查 Ollama 服务是否健康"""
        try:
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """
        发送聊天请求并获取回复内容。

        Args:
            messages: 消息列表，格式如 [{"role": "user", "content": "..."}]
            json_mode: 是否强制模型输出 JSON 格式 (Ollama 原生支持)

        Returns:
            str: 模型的回复文本
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False  # 关闭流式输出，便于一次性获取完整结果
        }

        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                self.api_chat_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get('message', {}).get('content', '')

        except requests.RequestException as e:
            logger.error(f"LLM 请求失败: {e}")
            raise ConnectionError(f"Failed to communicate with Ollama: {e}")

    def query_json(self, prompt: str, system_prompt: str = "You are a helpful data assistant.") -> Dict[str, Any]:
        """
        专门用于获取 JSON 结构化数据的的高级方法。
        包含自动清洗 Markdown 标记的逻辑。

        Args:
            prompt: 用户的指令
            system_prompt: 系统设定

        Returns:
            dict: 解析后的 Python 字典
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        raw_response = self.chat(messages, json_mode=True)

        # 数据清洗：有时候模型即使在 JSON 模式下也会包裹 ```json ... ```
        clean_response = self._clean_markdown(raw_response)

        try:
            return json.loads(clean_response)
        except json.JSONDecodeError:
            logger.error(f"JSON 解析失败。原始返回: {raw_response}")
            # 这里可以添加重试逻辑 (Phase 2 实现)
            raise ValueError("LLM 未返回有效的 JSON 格式")

    def _clean_markdown(self, text: str) -> str:
        """去除可能存在的 Markdown 代码块标记"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()


# --- 单元测试/使用示例 ---
if __name__ == "__main__":
    # 简单的测试脚本，直接运行此文件可验证连接
    print("正在测试本地 Ollama 连接...")

    client = LocalLlamaClient(model_name="llama3.1:latest")

    # 1. 测试普通对话
    try:
        reply = client.chat([{"role": "user", "content": "Say hello to NL-STV project!"}])
        print(f"\n[普通对话测试]\nAI: {reply}")
    except Exception as e:
        print(f"普通对话测试失败: {e}")

    # 2. 测试 JSON 提取 (这是后续功能的核心)
    print("\n[JSON 提取测试]")
    test_prompt = """
    Analyze this file metadata:
    Filename: nyc_taxi_2025.csv
    Columns: pickup_lat, pickup_lon, fare_amount

    Return a JSON with "file_type" and "columns_detected".
    """

    try:
        json_result = client.query_json(test_prompt)
        print("解析结果:", json.dumps(json_result, indent=2))

        if "file_type" in json_result:
            print("✅ JSON 测试通过")
        else:
            print("❌ JSON 结构不符合预期")

    except Exception as e:
        print(f"JSON 测试失败: {e}")