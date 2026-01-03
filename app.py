import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import sys
import re
from pathlib import Path
import logging

# --- 环境设置 ---
# 将项目根目录加入路径，确保能导入 core 模块
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from core.ingestion.loader_factory import LoaderFactory
from core.llm.AI_client import AIClient  # 确保使用的是支持 DeepSeek 的 Client
from core.profiler.semantic_analyzer import SemanticAnalyzer
from core.generation.code_generator import CodeGenerator
from core.execution.executor import CodeExecutor
from core.generation.goal_explorer import GoalExplorer
from core.generation.viz_editor import VizEditor  # [新增] 导入编辑器

# 配置页面
st.set_page_config(
    page_title="NL-STV Platform",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 辅助工具：变量名清洗 ---
def sanitize_var_name(filename):
    """
    将文件名转换为合法的 Python 变量名。
    例如: 'taxi_zones.shp' -> 'df_taxi_zones'
    """
    # 移除扩展名
    name = os.path.splitext(filename)[0]
    # 替换非字母数字为下划线
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # 避免数字开头, 且统一加 df_ 前缀
    if clean_name[0].isdigit():
        clean_name = "df_" + clean_name
    elif not clean_name.startswith("df_"):
        clean_name = "df_" + clean_name
    return clean_name.lower()


# --- 缓存资源 ---
@st.cache_resource
def get_core_modules():
    try:
        # 这里使用 DeepSeek 模型
        client = AIClient(
            model_name="deepseek-chat"
        )
        return (
            SemanticAnalyzer(client),
            CodeGenerator(client),
            CodeExecutor(),
            GoalExplorer(client),
            VizEditor(client)  # [新增] 返回编辑器
        )
    except Exception as e:
        st.error(f"核心组件初始化失败: {e}")
        return None, None, None, None, None


@st.cache_data
def load_data_snapshot(file_path, use_full_data=False):
    """加载单个文件的数据"""
    loader = LoaderFactory.get_loader(file_path)

    # 地理数据通常全量加载
    if file_path.endswith('.shp') or file_path.endswith('.geojson'):
        return loader.load(file_path)

    if use_full_data:
        return loader.load(file_path)
    else:
        return loader.peek(file_path, n=50000)


def save_uploaded_files(uploaded_files):
    """保存所有上传的文件到 data_sandbox"""
    save_dir = "data_sandbox"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_paths = []
    for up_file in uploaded_files:
        file_path = os.path.join(save_dir, up_file.name)
        with open(file_path, "wb") as f:
            f.write(up_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def get_analyzable_files(file_paths):
    """
    筛选出主数据文件（排除 .dbf, .shx 等伴生文件）。
    """
    valid_exts = ['.csv', '.parquet', '.shp', '.geojson', '.xlsx']
    return [f for f in file_paths if os.path.splitext(f)[1].lower() in valid_exts]


# --- 核心查询处理 (统一入口) ---
def handle_query(query_text, summaries, modules, data_context, force_new=False):
    """
    处理查询逻辑：区分 生成新图(Generate) 和 修改旧图(Edit)
    modules: (generator, executor, editor)
    """
    # 解包模块
    generator, executor, editor = modules

    # 1. 记录用户消息
    st.session_state.messages.append({"role": "user", "type": "text", "content": query_text})
    with st.chat_message("user"):
        st.markdown(query_text)

    # 2. AI 处理
    with st.chat_message("assistant"):
        msg_holder = st.empty()

        try:
            # === 逻辑分流: 编辑 vs 生成 ===
            # 如果存在上下文代码，且用户没有强制开启“新图表模式”，则进入编辑模式
            if st.session_state.last_generated_code and not force_new:
                msg_holder.markdown("🎨 正在基于现有图表进行修改 (Editing)...")
                code = editor.edit_code(
                    original_code=st.session_state.last_generated_code,
                    query=query_text,
                    summaries=summaries
                )
            else:
                msg_holder.markdown("🤔 正在构思新图表 (Generating)...")
                code = generator.generate_code(query_text, summaries)

            # === 执行代码 ===
            msg_holder.markdown("⚡ 正在执行代码...")
            res = executor.execute(code, data_context)

            # === 自愈机制 (Self-Healing) ===
            # 这里复用 generator 的 fix_code，因为它包含最全的 GIS 规则库
            retries = 3
            count = 0
            while not res.success and count < retries:
                count += 1
                msg_holder.warning(f"⚠️ 代码报错，正在进行第 {count} 次自动修复...")
                code = generator.fix_code(code, res.error, summaries)
                res = executor.execute(code, data_context)

            # === 结果展示 ===
            if res.success:
                # [关键] 更新上下文代码
                st.session_state.last_generated_code = code

                # 保存到历史记录
                st.session_state.messages.append({"role": "assistant", "type": "plot", "content": res.result})
                st.session_state.messages.append({"role": "assistant", "type": "code", "content": code})

                # [核心修复] 强制页面重绘，确保按钮区能立刻检测到 last_generated_code 并显示出来
                st.rerun()

            else:
                err_msg = f"❌ 执行失败: \n```\n{res.error}\n```"
                msg_holder.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": err_msg})
                with st.expander("查看最后代码"):
                    st.code(code, language="python")

        except Exception as e:
            st.error(f"System Error: {e}")


# --- 主程序 ---
def main():
    st.title("🗺️ NL-STV: 交互式时空分析平台")

    # 1. 初始化模块
    analyzer, generator, executor, explorer, editor = get_core_modules()
    if not analyzer: st.stop()

    # 打包 modules 方便传递
    modules_pack = (generator, executor, editor)

    # Session State 初始化
    if "messages" not in st.session_state: st.session_state.messages = []
    if "data_summaries" not in st.session_state: st.session_state.data_summaries = []
    if "uploaded_filenames" not in st.session_state: st.session_state.uploaded_filenames = []
    if "suggested_goals" not in st.session_state: st.session_state.suggested_goals = []
    if "prompt_trigger" not in st.session_state: st.session_state.prompt_trigger = None
    if "last_use_full" not in st.session_state: st.session_state.last_use_full = False

    # 上下文状态
    if "last_generated_code" not in st.session_state: st.session_state.last_generated_code = None
    if "last_query" not in st.session_state: st.session_state.last_query = None

    # 侧边栏
    with st.sidebar:
        st.header("📂 多文件接入")
        uploaded_files = st.file_uploader(
            "上传所有相关文件 (支持 .csv, .parquet, .shp 及伴生文件)",
            type=["csv", "parquet", "zip", "shp", "dbf", "shx", "prj", "sbn", "sbx", "xml", "cpg"],
            accept_multiple_files=True
        )

        st.markdown("---")
        st.header("⚙️ 设置")
        use_full = st.toggle("🚀 全量模式", value=False)
        st.info(f"💡 AI 模型: DeepSeek-V3")

        # 状态重置检测
        current_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []
        file_changed = current_names != st.session_state.uploaded_filenames
        mode_changed = use_full != st.session_state.last_use_full

        if file_changed or mode_changed:
            st.session_state.uploaded_filenames = current_names
            st.session_state.last_use_full = use_full
            st.session_state.data_summaries = []
            st.session_state.messages = []
            st.session_state.suggested_goals = []
            st.session_state.last_generated_code = None
            st.session_state.last_query = None
            st.cache_data.clear()

    if uploaded_files:
        # 1. 保存与筛选
        all_paths = save_uploaded_files(uploaded_files)
        analyzable_paths = get_analyzable_files(all_paths)

        if not analyzable_paths:
            st.warning("已上传文件，但未检测到支持的主数据格式 (.csv, .parquet, .shp)。")
        else:
            # 2. 分析语义
            if not st.session_state.data_summaries:
                summaries = []
                with st.status("🔍 正在解析多源数据...", expanded=True) as status:
                    for path in analyzable_paths:
                        fname = os.path.basename(path)
                        st.write(f"正在分析: {fname} ...")

                        summary = analyzer.analyze(path)
                        if "error" not in summary:
                            var_name = sanitize_var_name(fname)
                            summary['variable_name'] = var_name
                            summaries.append(summary)
                            st.write(f"✅ 已加载为变量: `{var_name}`")
                        else:
                            st.error(f"{fname} 分析失败: {summary['error']}")

                    st.session_state.data_summaries = summaries

                    if summaries:
                        st.write("💡 生成分析建议...")
                        st.session_state.suggested_goals = explorer.generate_goals(summaries[0])

                    status.update(label="✅ 所有文件加载完成", state="complete", expanded=False)

            # 3. 准备数据上下文
            data_context = {}
            for summary in st.session_state.data_summaries:
                path = summary['file_info']['path']
                var_name = summary['variable_name']
                try:
                    df = load_data_snapshot(path, use_full_data=use_full)
                    data_context[var_name] = df
                except Exception as e:
                    st.error(f"加载变量 {var_name} 失败: {e}")

            # 4. UI 展示
            if st.session_state.data_summaries:
                with st.expander("📊 已加载的数据集变量 (可在对话中直接使用)", expanded=True):
                    for summary in st.session_state.data_summaries:
                        st.markdown(f"**`{summary['variable_name']}`** ({summary['file_info']['name']})")
                        st.caption(f"包含列: {', '.join(list(summary['basic_stats']['column_stats'].keys())[:5])}...")

            # 5. 交互区域
            st.divider()
            st.subheader("💬 AI 可视化助手")

            # A. 推荐按钮
            if st.session_state.suggested_goals:
                cols = st.columns(len(st.session_state.suggested_goals))
                for i, goal in enumerate(st.session_state.suggested_goals):
                    if cols[i].button(goal, key=f"btn_{i}"):
                        st.session_state.prompt_trigger = goal

            # B. 历史记录 (这里会显示成功后的图表)
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["type"] == "text":
                        st.markdown(msg["content"])
                    elif msg["type"] == "plot":
                        st.plotly_chart(msg["content"], use_container_width=True)
                    elif msg["type"] == "code":
                        with st.expander("查看代码"):
                            st.code(msg["content"], language="python")

            # --- 输入与控制区 ---
            col_tools = st.columns([1, 1.5, 5])
            trigger_query = None
            force_new_toggle = False

            # C. 重新生成按钮
            if st.session_state.last_query:
                if col_tools[0].button("🔄 重新生成", help="重试上一次指令"):
                    trigger_query = st.session_state.last_query

            # D. 新图表模式开关 (仅当有上下文时显示)
            if st.session_state.last_generated_code:
                with col_tools[1]:
                    force_new_toggle = st.toggle(
                        "🆕 新图表模式",
                        value=False,
                        help="开启后将忽略当前图表，根据指令重新生成新图。"
                    )

            # E. 输入框
            chat_input_val = st.chat_input("输入指令 (例如 '把颜色改成红色' 或 '关联表A和表B')")

            # 优先级判断
            if st.session_state.prompt_trigger:
                trigger_query = st.session_state.prompt_trigger
                st.session_state.prompt_trigger = None
                force_new_toggle = True  # 点击推荐问题通常意味着想要新图
            elif chat_input_val:
                trigger_query = chat_input_val

            # 执行处理
            if trigger_query:
                st.session_state.last_query = trigger_query
                # 传递所有模块包
                handle_query(trigger_query, st.session_state.data_summaries, modules_pack, data_context,
                             force_new=force_new_toggle)

    else:
        st.markdown("""
        ### 👋 欢迎使用 NL-STV 多源数据分析平台

        请在左侧上传您的数据文件。

        **💡 提示：**
        - **多文件**: 支持同时上传多个文件（如业务数据 + 区域 Shapefile）。
        - **Shapefile**: 请务必上传 `.shp` 及其依赖文件 (`.dbf`, `.shx`)。
        - **交互**: 支持基于当前图表进行多轮对话修改（如 "把图例去掉"）。
        """)


if __name__ == "__main__":
    main()