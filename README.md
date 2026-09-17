# RAG-Personal 📚

基于 **Streamlit + LangChain + 通义千问** 的检索增强生成（RAG）问答助手，支持上传参考资料、基于资料作答，并把历史会话在本地持久化，实现带上下文的连续追问。

## ✨ 功能特性

- **参考资料上传**：上传 txt / md 文件，自动分块、向量化并写入本地向量库
- **基于资料的问答**：从向量库检索相关片段，交由通义千问生成简洁专业的回答
- **历史会话记忆**：会话记录本地持久化（JSON），支持多会话创建 / 切换 / 清空
- **历史会话检索**：可按**会话 ID 或对话内容**搜索历史会话
- **上下文注入**：提问时自动带上当前会话的历史上下文，支持连续追问

## 🧩 技术栈

| 层面 | 技术 |
| --- | --- |
| 界面 | Streamlit |
| 编排 | LangChain（`langchain-community` 的 `ChatTongyi` + `DashScopeEmbeddings`） |
| 向量库 | Chroma（本地持久化 `./chroma.db`） |
| 模型 | 通义千问 `qwen3-max`（对话）、`text-embedding-v4`（向量化） |
| 记忆 | 本地 JSON（`./conversation_history/*.json`） |

## 📦 环境准备

### 1. 配置 `.env`

在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=你的通义千问API Key
```

> Key 可在阿里云百炼（DashScope）控制台申请。`.env` 已被 `.gitignore` 忽略，不会被提交。

### 2. 安装依赖

```bash
pip install streamlit langchain-community langchain-chroma langchain-text-splitters dashscope python-dotenv
```

## 🚀 运行

```bash
streamlit run app_chat.py
```

浏览器会自动打开界面（默认 `http://localhost:8501`）。

## 🖱️ 使用流程

1. 左侧「📚 参考资料」上传 txt / md 文件，点击「📥 上传并入库」完成向量化
2. 左侧「💬 会话管理」新建或切换到指定会话 ID
3. 主区输入问题，助手基于**参考资料 + 当前会话历史**作答
4. 左侧「🕘 历史会话」可搜索、切换、清空历史会话

## 📁 目录结构

```
RAG_Project01/
├── app_chat.py             # 界面入口（Streamlit，统一集成层）
├── config_data.py          # 配置（模型名 / 向量库路径 / 分块参数）
├── vector_stores.py        # 向量库封装（Chroma）
├── knowledge_base.py       # 参考资料上传与向量化入库
├── rag.py                  # 基础 RAG 问答服务（检索 + 提示词 + 模型）
├── conversation_memory.py  # 历史会话本地持久化
├── rag_with_memory.py      # RAG + 历史记忆引擎（对外 ask 接口）
├── data/                   # 示例知识库资料（尺码推荐 / 洗涤养护 / 颜色选择）
└── .env                    # 密钥配置（需自行创建，不提交）
```

## ⚠️ 说明

以下为**运行时生成**的内容，已在 `.gitignore` 中忽略，首次运行会自动重建：

| 路径 | 说明 |
| --- | --- |
| `chroma.db/` | 向量库（可由上传的资料重新入库重建） |
| `conversation_history/` | 历史会话记录 |
| `md5.txt` | 已入库资料的指纹记录（用于避免重复入库） |

首次使用请自行上传资料；也可以直接用 `data/` 目录里的三个示例 txt 体验效果（在界面上传它们即可入库）。
