"""
历史会话记忆系统 (Conversation Memory Store)
============================================

与现有 RAG 项目集成，提供多轮对话的持久化与会话上下文管理。

设计要点：
- 会话存储：按「会话ID」持久化保存用户与 AI 的问答对，支持多轮对话上下文管理。
- 会话检索：根据会话ID加载历史消息，并可格式化为上下文字符串，注入到 RAG 链 / 后台集成模块。
- 会话判断：提供接口判断某会话ID是否已有历史记录，避免重复初始化。
- 数据存储：采用与现有向量数据库（Chroma，本地目录 ./chroma.db）一致的「本地文件存储」方式，
           在同级目录下使用 ./conversation_history/ 目录，每个会话一个 JSON 文件，文件结构清晰。
（rag.py / vector_stores.py / knowledge_base.py / rag_with_memory.py 等逻辑保持不变）。
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

# ----------------------------- 消息角色常量 -----------------------------
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"


class ConversationMemoryStore:
    """
    历史会话记忆存储器。

    每个会话ID对应一个本地 JSON 文件（./conversation_history/<session_id>.json），
    文件内保存：会话元信息 + 有序消息列表（role / content / timestamp / metadata）。
    """

    def __init__(self, store_dir: str = "./conversation_history"):
        """
        :param store_dir: 历史记忆存储目录（与 chroma.db 同目录下的本地存储）。
                          默认 './conversation_history'，与现有向量库本地存储风格保持一致。
        """
        self.store_dir = store_dir
        # 确保存储目录存在（与 knowledge_base.py 中 os.makedirs(..., exist_ok=True) 风格一致）
        os.makedirs(self.store_dir, exist_ok=True)

    # ------------------------------ 内部工具 ------------------------------
    @staticmethod
    def _safe_id(session_id: str) -> str:
        """将会话ID清洗为合法文件名，避免路径穿越与非法字符。"""
        safe = "".join(
            c for c in str(session_id) if c.isalnum() or c in ("-", "_", ".")
        )
        return safe or "default"

    def _session_path(self, session_id: str) -> str:
        """返回某会话ID对应的本地 JSON 文件路径。"""
        return os.path.join(self.store_dir, f"{self._safe_id(session_id)}.json")

    def _load(self, session_id: str) -> Dict[str, Any]:
        """从磁盘加载会话数据；文件不存在则返回空结构。"""
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return {"session_id": session_id, "created_at": None, "messages": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, session_id: str, data: Dict[str, Any]) -> None:
        """将会话数据写回磁盘（原子性：先写临时文件再替换，避免半写损坏）。"""
        path = self._session_path(session_id)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    # -------------------- 1. 会话判断（存在性） --------------------
    def session_exists(self, session_id: str) -> bool:
        """
        判断某会话ID是否已有历史记录。
        用于在初始化前检查，避免重复初始化同一会话。
        """
        return os.path.exists(self._session_path(session_id))

    def get_message_count(self, session_id: str) -> int:
        """返回某会话已保存的消息条数；会话不存在返回 0。"""
        if not self.session_exists(session_id):
            return 0
        return len(self._load(session_id).get("messages", []))

    # ----------------------- 2. 会话创建（初始化） -----------------------
    def create_session(self, session_id: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       force: bool = False) -> Dict[str, Any]:
        """
        创建一个新的会话（会话初始化）。

        :param session_id: 会话唯一标识。
        :param metadata: 可选的会话级元信息（如用户来源、主题等）。
        :param force: 若为 True，会话已存在时强制重建（清空历史）；
                      默认 False —— 已存在则直接复用，避免重复初始化。
        :return: 新建或复用的会话数据结构。
        """
        if self.session_exists(session_id) and not force:
            # 已存在：直接返回，避免重复初始化
            return self._load(session_id)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
            "messages": [],
        }
        self._save(session_id, data)
        return data

    # ----------------------- 3. 消息追加（多轮） -----------------------
    def append_message(self, session_id: str, role: str, content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        向指定会话追加一条消息（多轮对话的核心写入接口）。

        - 若会话不存在，则自动「惰性初始化」（lazy init），保证追加始终成功，
          同时不会在用户显式 create_session 之外产生重复初始化。
        """
        if not self.session_exists(session_id):
            self.create_session(session_id)

        data = self._load(session_id)
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {},
        }
        data["messages"].append(message)
        data["updated_at"] = message["timestamp"]
        self._save(session_id, data)
        return message

    def append_turn(self, session_id: str,
                    user_content: str, assistant_content: str,
                    user_metadata: Optional[Dict[str, Any]] = None,
                    assistant_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        一次性追加「一轮完整问答」（用户提问 + AI 回答）。
        适用于 RAG 问答结束后，把整轮对话落盘的场景。
        """
        u = self.append_message(session_id, ROLE_USER, user_content, user_metadata)
        a = self.append_message(session_id, ROLE_ASSISTANT, assistant_content, assistant_metadata)
        return [u, a]

    # ----------------------- 4. 历史加载（检索） -----------------------
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        加载某会话的全部历史消息（按时间顺序）。
        会话不存在时返回空列表，便于调用方直接判断与注入。
        """
        if not self.session_exists(session_id):
            return []
        return self._load(session_id).get("messages", [])

    def get_history_context(self, session_id: str,
                            max_turns: Optional[int] = None,
                            include_roles: tuple = (ROLE_USER, ROLE_ASSISTANT)) -> str:
        """
        将历史消息格式化为「可注入 RAG / 后台集成模块」的上下文字符串。

        这是「会话检索 → 注入历史上下文」的关键接口：
        返回形如：
            "用户：...\nAI：...\n用户：...\nAI：..."
        供 prompt 中的 {history} 占位符使用，使回答在参考上传文件的同时结合过往对话。

        :param session_id: 会话ID。
        :param max_turns: 仅取最近 N 轮（每轮=一问一答）；None 表示全部历史。
        :param include_roles: 纳入上下文的角色集合（默认仅 user/assistant）。
        :return: 上下文字符串；无历史时返回空串（调用方应以「（无历史对话）」兜底）。
        """
        messages = [m for m in self.get_history(session_id) if m["role"] in include_roles]
        if max_turns is not None:
            # 每轮包含 user+assistant 两条，取最近 2*max_turns 条
            messages = messages[-(max_turns * 2):]

        if not messages:
            return ""

        lines = []
        for m in messages:
            if m["role"] == ROLE_USER:
                label = "用户"
            elif m["role"] == ROLE_ASSISTANT:
                label = "AI"
            else:
                label = "系统"
            lines.append(f"{label}：{m['content']}")
        return "\n".join(lines)

    # ----------------------------- 辅助接口 -----------------------------
    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        返回某会话的元信息，供历史面板展示（更新时间 / 消息数等）。

        :return: 包含 session_id / created_at / updated_at / message_count / metadata 的字典；
                 会话不存在时返回 None。
        """
        if not self.session_exists(session_id):
            return None
        data = self._load(session_id)
        return {
            "session_id": data.get("session_id"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": len(data.get("messages", [])),
            "metadata": data.get("metadata", {}),
        }

    def list_sessions(self) -> List[str]:
        """列出所有已存在的会话ID。"""
        if not os.path.isdir(self.store_dir):
            return []
        return [f[:-5] for f in os.listdir(self.store_dir) if f.endswith(".json")]

    def clear_session(self, session_id: str) -> bool:
        """删除某会话的历史记录文件（谨慎使用，不可恢复）。"""
        path = self._session_path(session_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


# ------------------------------- 自检用例 -------------------------------
if __name__ == "__main__":
    import shutil

    demo_dir = "./_memory_demo"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    store = ConversationMemoryStore(store_dir=demo_dir)
    sid = "test-session-001"

    # 1. 会话判断：初始不存在
    print("初始是否存在:", store.session_exists(sid))          # False

    # 2. 会话创建
    store.create_session(sid, metadata={"topic": "尺码推荐"})
    print("创建后是否存在:", store.session_exists(sid))        # True

    # 3. 消息追加（多轮）
    store.append_turn(sid, "我体重110斤，身高170，推荐什么尺码？", "建议选择 L 码。")
    store.append_turn(sid, "那如果是120斤呢？", "建议选择 XL 码。")
    print("消息条数:", store.get_message_count(sid))           # 4

    # 4. 历史加载
    history = store.get_history(sid)
    print("历史加载条数:", len(history))

    # 5. 历史上下文（注入用）
    ctx = store.get_history_context(sid)
    print("注入上下文:\n", ctx)

    # 6. 重复初始化保护
    store.create_session(sid)  # 已存在，应复用而非清空
    print("重复创建后消息条数(应仍为4):", store.get_message_count(sid))

    # 清理演示目录
    shutil.rmtree(demo_dir)
    print("自检完成 ✓")
