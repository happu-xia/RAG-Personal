"""
RAG + 历史会话记忆 · 后台集成模块
================================
- VectorStoresServices（向量数据库，./chroma.db）
- get_retriever()（向量检索器）
- ChatTongyi（后台大模型，qwen3-max）
将「历史会话上下文」注入到 Prompt 的 {history} 占位符，
使回答在参考上传文件（RAG 检索）的同时，结合过往多轮对话。

实现方式：继承 RagServices，仅在本实例内用带 history 的 prompt 重建执行链，
原 rag.py 文件中的逻辑、prompt、链完全不被改动。
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from rag import RagServices
from conversation_memory import (
    ConversationMemoryStore,
    ROLE_USER,
    ROLE_ASSISTANT,
)


class RagWithMemory(RagServices):
    """
    在 RagServices 基础上注入历史会话上下文的问答服务。

    对外暴露 ask(session_id, question)，内部完成：
        1) 确保会话已初始化（已存在则复用，避免重复初始化）
        2) 加载历史上下文
        3) 调用「带 history 的 RAG 链」生成回答
        4) 将本轮问答追加进会话记忆（持久化）
    """

    def __init__(self, memory_store: ConversationMemoryStore, max_history_turns: int = 5):
        """
        :param memory_store: 历史会话记忆存储器（ConversationMemoryStore 实例）。
        :param max_history_turns: 注入 prompt 的最近历史轮数（控制上下文长度）。
        """
        self.memory_store = memory_store
        self.max_history_turns = max_history_turns

        # 复用父类的初始化：向量库、检索器、chat_model 全部沿用既有实现
        super().__init__()

        # 用「带 history 的 prompt」在本实例内重建链；
        # 不修改 rag.py，仅替换本实例的 prompt_template 与 chain
        self.prompt_template = self._build_memory_prompt()
        self.chain = self.get_chain()

    # ----------------------- 构建带历史的 Prompt -----------------------
    def _build_memory_prompt(self) -> ChatPromptTemplate:
        """
        在原有 prompt 基础上增加 {history} 变量。
        原 prompt（rag.py）：
            system: 以我提供的已知资料为主，简洁专业的回答用户问题。参考资料{context}。
            user:   请回答用户提问{input}
        新增 history 注入，使模型感知连续对话。
        """
        return ChatPromptTemplate([
            ("system",
             "以我提供的已知资料为主，简洁专业的回答用户问题。"
             "参考资料{context}。"
             "以下是与用户的历史对话记录，请结合上下文理解用户的连续意图：\n{history}"),
            ("user", "请回答用户提问{input}")
        ])

    # ----------------------- 重建执行链（仅本实例） -----------------------
    def get_chain(self):
        """
        重建执行链，注入 history 变量。仅作用于本实例，原 rag.py 的 chain 不受影响。

        输入为一个字典 {"input": 问题, "history": 历史上下文字符串}：
            - input     -> 直接进入 prompt 的 {input}
            - history   -> 直接进入 prompt 的 {history}
            - context   -> 由 input 经 retriever | format_document 得到
        """
        retriever = self.vector_stores.get_retriever()

        def format_document(docs):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"
            return formatted_str

        chain = (
            RunnablePassthrough.assign(
                # 用 input 检索向量库，得到 context
                context=lambda x: format_document(retriever.invoke(x["input"]))
            )
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )
        return chain

    # ----------------------------- 对外问答 -----------------------------
    def ask(self, session_id: str, question: str) -> str:
        """
        带历史记忆的问答。

        :param session_id: 会话ID（用于隔离/加载不同历史）。
        :param question: 用户本轮提问。
        :return: AI 回答文本。
        """
        # 确保会话已初始化：已存在则复用，避免重复初始化
        if not self.memory_store.session_exists(session_id):
            self.memory_store.create_session(session_id)

        # 加载历史上下文（多轮对话）
        history_ctx = self.memory_store.get_history_context(
            session_id, max_turns=self.max_history_turns
        )

        # 调用带 history 的 RAG 链
        answer = self.chain.invoke({
            "input": question,
            "history": history_ctx if history_ctx else "（无历史对话）",
        })

        # 将本轮问答持久化，供后续多轮对话参考
        self.memory_store.append_turn(
            session_id,
            user_content=question,
            assistant_content=answer,
        )
        return answer


# ------------------------------- 自检用例 -------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    store = ConversationMemoryStore()
    rag = RagWithMemory(store, max_history_turns=5)

    sid = "demo-session"
    print(rag.ask(sid, "我体重110斤，请给我提供尺码推荐"))
    print("-" * 30)
    print(rag.ask(sid, "那如果我现在想买外套呢，买什么颜色？"))  # 第二轮，会带上第一轮历史
    print("-" * 30)
    print("历史条数:", store.get_message_count(sid))
