import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import sys
from pathlib import Path
import logging

# --- 环境设置 ---
# 将项目根目录加入路径，确保能导入 core 模块
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from core.ingestion.loader_factory import LoaderFactory
from core.llm.ollama_client import LocalLlamaClient
from core.profiler.semantic_analyzer import SemanticAnalyzer

# 配置页面
st.set_page_config(
    page_title="NL-STV Platform",
    page_icon="🗺️",
    layout="wide"
)


# --- 工具函数 ---
def save_uploaded_file(uploaded_file, save_dir="data_sandbox"):
    """将上传的内存文件保存到磁盘，以便 Loader 读取"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


@st.cache_resource
def get_analyzer():
    """初始化 AI 分析器 (单例模式)"""
    try:
        client = LocalLlamaClient(model_name="llama3.1:latest")
        return SemanticAnalyzer(client)
    except Exception as e:
        st.error(f"无法连接到 Ollama: {e}")
        return None


# --- 主界面逻辑 ---

def main():
    st.title("🗺️ NL-STV: AI 驱动的时空数据分析平台")
    st.markdown("---")

    # 1. 侧边栏：控制与状态
    with st.sidebar:
        st.header("1. 数据接入")
        uploaded_file = st.file_uploader(
            "上传数据文件",
            type=["csv", "parquet", "xlsx", "zip"],  # zip 用于 shapefile
            help="支持 CSV, Parquet, Excel。Shapefile 请压缩为 zip 上传。"
        )

        st.info("💡 当前使用模型: Llama 3.1 (Local)")

    # 2. 主区域：分析结果
    if uploaded_file is not None:
        # 保存文件
        file_path = save_uploaded_file(uploaded_file)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.success(f"文件已上传: {uploaded_file.name}")
            st.caption(f"大小: {uploaded_file.size / 1024:.1f} KB")

        # 触发分析
        analyzer = get_analyzer()
        if analyzer:
            with st.spinner("🤖 AI 正在阅读数据并提取语义..."):
                # 调用我们在后端写的核心逻辑
                analysis_result = analyzer.analyze(file_path)

            if "error" in analysis_result:
                st.error(analysis_result["error"])
            else:
                render_analysis_report(analysis_result)
    else:
        render_landing_page()


def render_landing_page():
    st.markdown("""
    ### 欢迎使用自然语言时空可视化器

    这是一个**零代码**数据分析工具。您只需要上传文件，AI 将会自动：
    1.  **识别数据类型** (轨迹、区域、OD流向)
    2.  **理解字段含义** (哪个是时间，哪个是经纬度)
    3.  *(即将上线)* **生成可视化图表**

    👈 请在左侧上传您的 `csv` 或 `parquet` 文件开始。
    """)


def render_analysis_report(result):
    """渲染分析报告"""
    ai_data = result.get("semantic_analysis", {})
    stats = result.get("basic_stats", {})

    # --- 顶部：AI 摘要 ---
    st.header("2. AI 数据摘要")

    # 使用 Metric 展示关键指标
    m1, m2, m3 = st.columns(3)
    m1.metric("数据行数", f"{stats['rows']:,}")
    m2.metric("字段数", stats['cols'])

    # 动态颜色标签
    d_type = ai_data.get('dataset_type', 'UNKNOWN')
    color = "green" if d_type == "TRAJECTORY" else "blue" if d_type == "GEO_ZONE" else "orange"
    m3.markdown(f"**数据类型**: :{color}[{d_type}]")

    st.info(f"📝 **AI 解读**: {ai_data.get('description', '无描述')}")

    # --- 中部：字段语义映射 ---
    st.subheader("3. 语义映射表")
    st.caption("AI 已自动识别以下列的业务含义：")

    # 构造一个展示用的 DataFrame
    tags = ai_data.get("semantic_tags", {})
    col_stats = stats.get("column_stats", {})

    table_data = []
    for col, role in tags.items():
        # 获取该列的示例数据
        meta = col_stats.get(col, {})
        samples = str(meta.get("samples", []))
        dtype = meta.get("dtype", "unknown")

        table_data.append({
            "列名 (Column)": col,
            "数据类型": dtype,
            "AI 语义标签": role,
            "示例值": samples
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # --- 底部：数据预览 ---
    with st.expander("查看原始数据预览 (Top 5 Rows)", expanded=False):
        # 重新快速读取一下用于展示 (利用 LoaderFactory)
        loader = LoaderFactory.get_loader(result['file_info']['path'])
        df_preview = loader.peek(result['file_info']['path'], n=5)
        st.dataframe(df_preview)


if __name__ == "__main__":
    main()