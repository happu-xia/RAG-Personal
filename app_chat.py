"""
本文件是项目的「统一入口与集成层」，仅负责界面编排与能力串联，
    config_data.py           → 配置常量（向量库路径 / 模型名 / 分块参数）
    vector_stores.py         → 向量数据库（Chroma，./chroma.db）
    knowledge_base.py        → 参考资料上传 / 向量化入库（KnowledgeBaseService）
    rag.py                   → 基础 RAG 问答服务（RagServices）
    conversation_memory.py   → 历史会话本地持久化（ConversationMemoryStore）
    rag_with_memory.py       → RAG + 历史记忆引擎（RagWithMemory，对外 ask）

本界面集成三大能力：
    1) 参考资料上传  → KnowledgeBaseService.upload_by_str（./chroma.db）
    2) 历史记忆持久化 → ConversationMemoryStore（./conversation_history/*.json）
    3) 针对性作答     → RagWithMemory.ask（检索资料 + 注入历史上下文）

界面特性（简洁清爽 / 现代柔和 / 功能分明）：
    - 浅灰底 + 白色卡片，柔和护眼、干净整洁
    - 蓝色（#1677ff）作主色调：标题徽标、用户气泡、发送按钮、选中/激活态
    - 分区用小标题（蓝条）+ 柔和分隔线，功能区域清晰易懂
    - 侧栏「参考资料」「会话管理」「历史会话」三块明确分区
    - 用户气泡蓝色 / 助手气泡白底浅边，角色对比分明且舒适
    - 聊天输入框圆角浅边，聚焦时蓝色描边 + 浅蓝光晕
    - 历史搜索框：长扁椭圆黑框、无填充，深色文字确保清晰可读
    - 提示框统一浅色描边，主色点缀，不刺眼

启动方式（项目目录下，使用已安装 langchain + streamlit 的 Python）：
    streamlit run app_chat.py
"""

import html
import streamlit as st
from dotenv import load_dotenv

from knowledge_base import KnowledgeBaseService
from conversation_memory import ConversationMemoryStore
from rag_with_memory import RagWithMemory

load_dotenv()

# -------------------------- 简洁清爽（浅底 + 蓝主色）样式 --------------------------
PRIMARY = "#1677ff"        # 主色蓝（标题徽标 / 用户气泡 / 发送按钮 / 选中态）
PRIMARY_SOFT = "#e8f1ff"   # 主色浅蓝（选中/激活背景）
INK = "#1f2329"            # 主文字（柔和近黑，避免纯黑刺眼）
SUB = "#6b7280"            # 次要文字（说明/时间）
LINE = "#eaecef"          # 分隔线 / 浅边框
CARD = "#ffffff"          # 卡片白

st.markdown(
    f"""
    <style>
    /* ===== 全局：浅灰底 + 柔和近黑字 ===== */
    html, body, .stApp {{ background: #f7f9fc !important; color: {INK}; }}
    .main .block-container {{ max-width: 960px; padding-top: 1.5rem; }}
    .stCaption, [data-testid="stCaption"],
    .stMarkdown, .stText, .stAlert {{ color: {INK} !important; }}

    /* ===== 左侧功能区：白卡片 + 柔和右边线 ===== */
    [data-testid="stSidebar"] {{
        background: {CARD};
        border-right: 1px solid {LINE};
        box-shadow: 1px 0 0 rgba(16,24,40,0.02);
    }}
    [data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
    /* 分区小标题：蓝条 + 加粗，功能清晰 */
    .panel-title {{
        color: {INK}; font-size: 14px; font-weight: 700;
        margin-bottom: 8px; padding-left: 10px;
        border-left: 3px solid {PRIMARY};
    }}
    .panel-hint {{ color: {SUB}; font-size: 12px; margin-bottom: 8px; }}

    /* ===== 柔和分隔线 ===== */
    [data-testid="stDivider"] {{ border-color: {LINE} !important; margin: 14px 0; }}

    /* ===== 功能按键：白底 + 蓝边 + 蓝字（清晰可点） ===== */
    [data-testid="stSidebar"] button,
    [data-testid="stFileUploader"] button,
    [data-testid="stChatInput"] button {{
        background: {CARD} !important;
        color: {PRIMARY} !important;
        border: 1px solid {PRIMARY} !important;
        border-radius: 10px !important;
        font-weight: 600;
    }}
    [data-testid="stSidebar"] button:hover,
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stChatInput"] button:hover {{
        background: {PRIMARY} !important;
        color: #ffffff !important;
        border-color: {PRIMARY} !important;
    }}
    /* 禁用态：浅灰底 + 浅边，明确不可点 */
    [data-testid="stSidebar"] button:disabled {{
        background: #f3f4f6 !important;
        color: #b6bcc6 !important;
        border: 1px solid #e3e6ea !important;
        opacity: 1;
    }}
    /* 侧栏按钮：左对齐、紧凑、省略号 */
    [data-testid="stSidebar"] button {{
        text-align: left; font-size: 13px; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis;
    }}
    /* 文件上传区：虚线浅边，干净 */
    [data-testid="stFileUploader"] section {{
        border: 1px dashed #c9d3df !important;
        border-radius: 10px;
    }}

    /* ===== 右侧聊天标题：深色文字 + 浅色底栏 ===== */
    .title-bar {{
        color: {INK};
        font-size: 22px; font-weight: 700;
        padding: 8px 2px 14px 2px;
        border-bottom: 1px solid {LINE};
        margin-bottom: 16px;
    }}
    .title-bar .badge {{
        font-size: 12px; font-weight: 500; color: {PRIMARY};
        background: {PRIMARY_SOFT}; border: none;
        border-radius: 6px; padding: 3px 10px; margin-left: 10px;
    }}

    /* ===== 聊天消息气泡：用户蓝 / 助手白，角色分明且舒适 ===== */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        justify-content: flex-end;
    }}
    /* 用户（右侧，主色蓝底白字） */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
        [data-testid="stChatMessageContent"] {{
        background: {PRIMARY}; color: #ffffff;
        border-radius: 16px 16px 4px 16px;
        box-shadow: 0 1px 2px rgba(22,119,255,0.25);
    }}
    /* 助手（左侧，白底浅边） */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
        [data-testid="stChatMessageContent"] {{
        background: {CARD}; color: {INK};
        border: 1px solid {LINE};
        border-radius: 16px 16px 16px 4px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    }}

    /* ===== 聊天输入框：圆角浅边，聚焦蓝色描边 + 浅蓝光晕 ===== */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stChatInputContainer"],
    .stChatFloatingInputContainer {{
        background: #f7f9fc !important;
    }}
    [data-testid="stChatInput"] {{
        background: {CARD} !important;
        border: 1.5px solid #d9dee5 !important;
        border-radius: 14px !important;
        padding: 4px 6px;
        box-shadow: none !important;
        transition: border-color .15s, box-shadow .15s;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px rgba(22,119,255,0.12) !important;
    }}
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {{
        background: {CARD} !important;
        border: none !important;
        border-radius: 14px !important;
        color: {INK} !important;
        caret-color: {PRIMARY} !important;
    }}
    /* 发送按钮：蓝底圆角 + 白图标 */
    [data-testid="stChatInput"] button {{
        background: {PRIMARY} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        background: #0e5fd8 !important;
        color: #ffffff !important;
    }}
    [data-testid="stChatInput"] button svg {{
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }}

    /* ===== 历史会话「选中」块：浅蓝底 + 蓝左条（激活态清晰） ===== */
    .hist-selected {{
        background: {PRIMARY_SOFT}; color: {PRIMARY};
        border-left: 3px solid {PRIMARY};
        border-radius: 8px; padding: 8px 10px;
        font-size: 13px; margin: 4px 0; font-weight: 600;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .hist-meta {{ color: {SUB}; font-size: 12px; }}

    /* ===== 历史搜索框：长扁椭圆黑框 · 无填充 · 深色文字清晰可读 ===== */
    /* 1) 外层容器：清掉 Streamlit 自带的边框与白底，避免与胶囊黑框叠加 */
    [data-testid="stTextInput"]:has(input[placeholder="搜索会话 ID 或内容…"]) {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    [data-testid="stTextInput"]:has(input[placeholder="搜索会话 ID 或内容…"]) > div,
    [data-testid="stTextInput"]:has(input[placeholder="搜索会话 ID 或内容…"]) [data-baseweb="input"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        min-height: 0 !important;
    }}
    /* 2) 输入框本体：长扁胶囊形 + 黑色描边 + 无填充 */
    [data-testid="stTextInput"] input[placeholder="搜索会话 ID 或内容…"] {{
        background-color: transparent !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%231a1a1a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: 16px center;
        background-size: 15px 15px;
        border: 1.5px solid #1a1a1a !important;
        border-radius: 9999px !important;
        height: auto !important;
        padding: 9px 18px 9px 40px !important;
        line-height: 1.25 !important;
        color: {INK} !important;          /* 深色文字，确保与浅底对比清晰 */
        caret-color: {PRIMARY} !important;
        transition: border-color .15s, box-shadow .15s;
    }}
    /* 3) 占位符：中灰，浅底上依然清晰（不与背景同色） */
    [data-testid="stTextInput"] input[placeholder="搜索会话 ID 或内容…"]::placeholder {{
        color: #8a94a6 !important;
        opacity: 1 !important;
    }}
    /* 4) 聚焦态：保持黑框，仅加淡黑光晕，风格统一 */
    [data-testid="stTextInput"] input[placeholder="搜索会话 ID 或内容…"]:focus {{
        border-color: #000000 !important;
        box-shadow: 0 0 0 3px rgba(0,0,0,0.08) !important;
        outline: none !important;
    }}
    /* 搜索结果计数 / 空状态 */
    .hist-count {{ color: {PRIMARY}; font-size: 12px; font-weight: 600; margin: 6px 2px 8px; }}
    .hist-empty {{
        color: {SUB}; font-size: 13px; line-height: 1.6;
        padding: 16px 10px; text-align: center;
        background: #f3f5f8; border: 1px dashed #d9dee5; border-radius: 10px;
    }}

    /* ===== 提示框：浅色描边 + 主色左条，柔和 ===== */
    .stAlert {{
        background: {CARD} !important;
        border: 1px solid {LINE} !important;
        border-left: 3px solid {PRIMARY} !important;
        color: {INK} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================ 服务初始化 ============================
# 缓存于 session_state，避免每次 rerun 重建向量库 / 大模型。
if "memory_store" not in st.session_state:
    st.session_state.memory_store = ConversationMemoryStore()
if "rag_service" not in st.session_state:
    st.session_state.rag_service = RagWithMemory(st.session_state.memory_store)
if "active_session" not in st.session_state:
    st.session_state.active_session = "default-session"
if "session_input" not in st.session_state:
    st.session_state.session_input = st.session_state.active_session
if "messages" not in st.session_state:
    st.session_state.messages = []


def _load_messages(store: ConversationMemoryStore, session_id: str):
    """从本地记忆加载某会话的历史消息，供对话流渲染。"""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in store.get_history(session_id)
    ]


store: ConversationMemoryStore = st.session_state.memory_store


def _switch_session(session_id: str):
    """切换到指定会话：更新状态、重载消息；按钮点击会自动触发脚本 rerun 刷新界面。"""
    store.create_session(session_id)  # 已存在则复用，避免重复初始化
    st.session_state.active_session = session_id
    # 注意：session_input 已绑定 st.text_input 控件，不能直接赋值；
    # 保留文本框当前内容即可，避免触发 Streamlit 的控件键不可改写限制。
    st.session_state.messages = _load_messages(store, session_id)
    # 不调用 st.rerun()：按钮交互本身已自动触发一次脚本 rerun，避免重复刷新。


# ========================== 左侧栏：参考资料上传区 ==========================
with st.sidebar:
    st.markdown('<div class="panel-title">📚 参考资料</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-hint">上传 txt / md，作为问答依据</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "选择文件",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("📥 上传并入库", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("正在向量化并写入知识库…"):
            kb = KnowledgeBaseService()
            for uf in uploaded_files:
                text = uf.getvalue().decode("utf-8")
                result = kb.upload_by_str(text, uf.name)
                st.toast(f"{uf.name}：{result}")
            # 上传后刷新 RAG 检索器，使其纳入新资料（读取最新落盘的向量库）
            st.session_state.rag_service = RagWithMemory(store)
        st.success("参考资料已入库，可开始提问 ✅")

    # ----------------------------- 会话管理 -----------------------------
    st.divider()
    st.markdown('<div class="panel-title">💬 会话管理</div>', unsafe_allow_html=True)
    st.markdown(f"**当前会话：** `{st.session_state.active_session}`")
    st.markdown(
        f'<div class="hist-meta">历史消息：{store.get_message_count(st.session_state.active_session)} 条</div>',
        unsafe_allow_html=True,
    )

    st.text_input(
        "新建 / 切换会话ID",
        key="session_input",
        placeholder="输入ID后点击下方按钮",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ 创建/切换", use_container_width=True):
            sid = (st.session_state.session_input or "").strip() or st.session_state.active_session
            _switch_session(sid)
    with c2:
        if st.button("🗑️ 清空", use_container_width=True):
            store.clear_session(st.session_state.active_session)
            st.session_state.messages = []
            # 按钮点击会自动触发脚本 rerun，无需显式调用 st.rerun()

    # ----------------------------- 历史会话浏览器 -----------------------------
    st.divider()
    st.markdown('<div class="panel-title">🕘 历史会话</div>', unsafe_allow_html=True)

    # 搜索框：内嵌放大镜、圆角浅边、聚焦蓝色（主流搜索框设计）
    search = st.text_input(
        "搜索历史会话",
        key="history_search",
        placeholder="搜索会话 ID 或内容…",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="panel-hint">支持按会话 ID 或对话内容检索</div>',
        unsafe_allow_html=True,
    )

    def _session_matches(sid: str, q: str) -> bool:
        """会话 ID 或任意历史消息内容命中即匹配。"""
        if q in sid.lower():
            return True
        for m in store.get_history(sid):
            if q in (m.get("content") or "").lower():
                return True
        return False

    all_sessions = store.list_sessions()
    q = (search or "").strip().lower()
    matched = (
        [s for s in all_sessions if _session_matches(s, q)]
        if q else all_sessions
    )

    # 结果计数（仅在搜索时展示）
    if q:
        st.markdown(
            f'<div class="hist-count">找到 {len(matched)} 个会话</div>',
            unsafe_allow_html=True,
        )

    if not matched:
        if q:
            st.markdown(
                f'<div class="hist-empty">未找到匹配「{html.escape(search)}」的会话</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="hist-meta">暂无历史会话记录</div>', unsafe_allow_html=True)
    else:
        # 最近更新的排在前
        matched.sort(reverse=True)
        active = st.session_state.active_session
        # 选中状态：浅蓝块 + 蓝左条（激活态清晰）
        if active in matched:
            m = store.get_session_meta(active) or {}
            st.markdown(
                f'<div class="hist-selected">▶ {active}　·　'
                f'{m.get("message_count", 0)}条　·　当前</div>',
                unsafe_allow_html=True,
            )
        for sid in matched:
            if sid == active:
                continue
            meta = store.get_session_meta(sid) or {}
            updated = meta.get("updated_at") or "—"
            cnt = meta.get("message_count", 0)
            label = f"{sid}　·　{cnt}条　·　{updated}"
            if st.button(label, key=f"hist_{sid}", use_container_width=True):
                _switch_session(sid)

# ========================== 主区域：聊天对话区 ==========================
st.markdown(
    '<div class="title-bar">AI 智能助手'
    '<span class="badge">RAG</span></div>',
    unsafe_allow_html=True,
)
st.caption("我会基于你上传的参考资料作答，并结合当前会话的历史对话上下文。")

# ---------------- 渲染历史消息（对话流） ----------------
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# ---------------- 用户输入与问答逻辑 ----------------
if prompt := st.chat_input("输入你的问题，我会基于参考资料回答…"):
    sid = st.session_state.active_session

    # 1) 回显用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)

    # 2) 助手基于「参考资料 + 历史对话」作答
    #    （rag_with_memory 内部完成：向量检索 → 注入历史上下文 → 落盘持久化）
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("检索资料并生成回答…"):
            answer = st.session_state.rag_service.ask(sid, prompt)
        st.write(answer)

    # 3) 落盘展示（ask 已写入本地记忆，此处仅更新对话流）
    st.session_state.messages.append({"role": "assistant", "content": answer})
