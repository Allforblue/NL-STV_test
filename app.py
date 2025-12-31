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

# 引入核心模块
from core.ingestion.loader_factory import LoaderFactory
from core.llm.ollama_client import LocalLlamaClient
from core.profiler.semantic_analyzer import SemanticAnalyzer
from core.generation.code_generator import CodeGenerator
from core.execution.executor import CodeExecutor
from core.generation.goal_explorer import GoalExplorer  # [新增] 目标探索器

# 配置页面
st.set_page_config(
    page_title="NL-STV Platform",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 缓存资源 (单例模式) ---
@st.cache_resource
def get_core_modules():
    """初始化所有核心组件，避免重复加载 LLM"""
    try:
        # 统一使用同一个 LLM 客户端
        client = LocalLlamaClient(model_name="llama3.1:latest")

        analyzer = SemanticAnalyzer(client)
        generator = CodeGenerator(client)
        executor = CodeExecutor()
        explorer = GoalExplorer(client)

        return analyzer, generator, executor, explorer
    except Exception as e:
        st.error(f"核心组件初始化失败: {e}")
        return None, None, None, None


@st.cache_data
def load_data_for_analysis(file_path, use_full_data=False):
    """
    加载用于绘图的数据。
    根据开关决定是全量加载还是采样加载。
    """
    loader = LoaderFactory.get_loader(file_path)

    # Shapefile/GeoJSON 通常体积可控，总是全量加载以保证地图完整性
    if file_path.endswith('.shp') or file_path.endswith('.geojson'):
        return loader.load(file_path)

    if use_full_data:
        # 全量模式：适合统计分析，但前端渲染散点图可能会卡
        return loader.load(file_path)
    else:
        # 极速模式：采样 50,000 行，适合快速探索和散点图预览
        return loader.peek(file_path, n=50000)


# --- 辅助函数 ---
def save_uploaded_file(uploaded_file):
    save_dir = "data_sandbox"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# --- 主逻辑 ---
def main():
    st.title("🗺️ NL-STV: AI 驱动的时空数据分析平台")

    # 1. 初始化组件
    analyzer, generator, executor, explorer = get_core_modules()
    if not analyzer:
        st.stop()

    # 2. 初始化 Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_summary" not in st.session_state:
        st.session_state.analysis_summary = None
    if "suggested_goals" not in st.session_state:
        st.session_state.suggested_goals = []
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "last_use_full" not in st.session_state:
        st.session_state.last_use_full = False
    # 用于处理按钮点击触发聊天
    if "prompt_trigger" not in st.session_state:
        st.session_state.prompt_trigger = None

    # 3. 侧边栏：控制面板
    with st.sidebar:
        st.header("📂 数据接入")
        uploaded_file = st.file_uploader(
            "上传数据文件 (CSV/Parquet/Shapefile)",
            type=["csv", "parquet", "zip", "shp"]
        )

        st.markdown("---")
        st.header("⚙️ 设置")
        use_full_data = st.toggle(
            "🚀 启用全量数据模式",
            value=False,
            help="开启后加载所有数据。统计更准，但大量点的绘图可能变慢。"
        )

        st.info(f"💡 AI 模型: Llama 3.1 (Local)")

        # 重置逻辑：如果换了文件 OR 切换了模式，清空状态
        file_changed = uploaded_file and uploaded_file.name != st.session_state.current_file
        mode_changed = use_full_data != st.session_state.last_use_full

        if file_changed or mode_changed:
            st.session_state.current_file = uploaded_file.name if uploaded_file else None
            st.session_state.last_use_full = use_full_data
            st.session_state.analysis_summary = None
            st.session_state.suggested_goals = []
            st.session_state.messages = []
            st.cache_data.clear()  # 清除旧的数据缓存

    # 4. 核心流程
    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)

        # --- Phase 1: 自动分析与目标生成 ---
        if not st.session_state.analysis_summary:
            with st.status("🔍 AI 正在阅读数据...", expanded=True) as status:
                st.write("正在提取数据指纹与语义...")
                summary = analyzer.analyze(file_path)

                if "error" in summary:
                    status.update(label="❌ 分析失败", state="error")
                    st.error(summary["error"])
                    st.stop()

                st.session_state.analysis_summary = summary

                st.write("💡 正在构思分析方向 (Goal Exploration)...")
                goals = explorer.generate_goals(summary)
                st.session_state.suggested_goals = goals

                status.update(label="✅ 数据感知完成", state="complete", expanded=False)

        # 展示数据摘要
        summary = st.session_state.analysis_summary
        with st.expander("📊 查看数据摘要与语义映射", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**数据类型**: {summary['semantic_analysis'].get('dataset_type', 'N/A')}")
                st.markdown(f"**总行数**: {summary['basic_stats']['rows']:,}")
            with col2:
                st.markdown(f"**AI 描述**: {summary['semantic_analysis'].get('description', 'N/A')}")

            st.table(
                pd.DataFrame(list(summary['semantic_analysis']['semantic_tags'].items()), columns=['列名', '语义标签']))

        # --- Phase 2: 对话式绘图 ---
        st.divider()
        st.subheader("💬 AI 可视化助手")

        # 4.1 推荐目标按钮 (Goal Explorer)
        if st.session_state.suggested_goals:
            st.caption("✨ 猜你想问：")
            # 动态创建列
            cols = st.columns(len(st.session_state.suggested_goals))
            for i, goal in enumerate(st.session_state.suggested_goals):
                if cols[i].button(goal, key=f"goal_btn_{i}", use_container_width=True):
                    st.session_state.prompt_trigger = goal

        # 4.2 聊天历史展示
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["type"] == "text":
                    st.markdown(msg["content"])
                elif msg["type"] == "plot":
                    st.plotly_chart(msg["content"], use_container_width=True)
                elif msg["type"] == "code":
                    with st.expander("查看生成的代码"):
                        st.code(msg["content"], language="python")

        # 4.3 输入处理逻辑
        # 优先处理按钮点击，否则处理输入框
        user_input = None
        chat_input_val = st.chat_input("请输入指令，例如：'画出车费的分布' 或 '展示OD流向'")

        if st.session_state.prompt_trigger:
            user_input = st.session_state.prompt_trigger
            st.session_state.prompt_trigger = None  # 消费掉触发器
        elif chat_input_val:
            user_input = chat_input_val

        # 4.4 执行逻辑
        if user_input:
            # 显示用户消息
            st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # AI 处理
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🤔 正在思考绘图代码...")

                # A. 加载数据上下文
                try:
                    df_context = load_data_for_analysis(file_path, use_full_data=use_full_data)
                except Exception as e:
                    st.error(f"数据加载失败: {e}")
                    st.stop()

                # B. 代码生成与执行循环 (含自愈机制)
                try:
                    # 初次生成
                    generated_code = generator.generate_code(user_input, summary)

                    # 执行
                    message_placeholder.markdown("⚡ 正在执行代码...")
                    exec_result = executor.execute(generated_code, df_context)

                    # === 自愈机制 (Self-Healing Loop) ===
                    max_retries = 2
                    retry_count = 0

                    while not exec_result.success and retry_count < max_retries:
                        retry_count += 1
                        message_placeholder.warning(f"⚠️ 代码报错，正在进行第 {retry_count} 次自动修复...")

                        # 调用修复
                        fixed_code = generator.fix_code(
                            original_code=generated_code,
                            error_trace=exec_result.error,
                            context_summary=summary
                        )

                        # 重试执行
                        generated_code = fixed_code
                        exec_result = executor.execute(generated_code, df_context)
                    # ===================================

                    # 结果处理
                    if exec_result.success:
                        # 成功
                        st.plotly_chart(exec_result.result, use_container_width=True)

                        # 构建成功消息
                        row_count = len(df_context)
                        success_msg = f"✅ 图表生成成功！(基于 {row_count:,} 条数据)"
                        if retry_count > 0:
                            success_msg += f" | ✨ 自动修复了 {retry_count} 个错误。"

                        message_placeholder.markdown(success_msg)

                        # 保存历史
                        st.session_state.messages.append(
                            {"role": "assistant", "type": "plot", "content": exec_result.result})
                        st.session_state.messages.append(
                            {"role": "assistant", "type": "code", "content": generated_code})

                        with st.expander("查看最终代码"):
                            st.code(generated_code, language="python")
                    else:
                        # 失败 (重试后依然失败)
                        error_msg = f"❌ 抱歉，我尝试了 {retry_count} 次修复但依然失败。\n\n**错误信息**: \n```\n{exec_result.error}\n```"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": error_msg})
                        with st.expander("查看最后生成的代码"):
                            st.code(generated_code, language="python")

                except Exception as e:
                    st.error(f"系统内部错误: {e}")

    else:
        # Landing Page
        st.markdown("""
        ### 👋 欢迎使用 NL-STV 时空分析平台

        这是一个基于 LLM 的智能数据可视化工具。它能够理解您的数据语义，并根据自然语言指令自动编写 Python 代码绘图。

        **功能亮点：**
        - 🧠 **语义感知**: 自动识别时间、坐标、业务指标。
        - 🗣️ **对话绘图**: 说出您的需求，自动生成 Plotly 交互式图表。
        - 🛠️ **自动自愈**: 代码报错？AI 会自己 Debug 并重试。
        - 💡 **目标推荐**: 不知道问什么？AI 会主动给您推荐分析方向。

        **请在左侧上传数据文件开始体验 (支持 CSV, Parquet, Shapefile)。**
        """)


if __name__ == "__main__":
    main()