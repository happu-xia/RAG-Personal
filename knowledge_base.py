"""
知识库
"""
import os
from datetime import datetime
import config_data as config
from dotenv import load_dotenv
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def check_md5(md5_str: str):
    """检查传入的md5字符串是否处理过"""
    if not os.path.exists(config.md5_path):
        # 文件不存在时创建空文件并返回False
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False

    with open(config.md5_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if md5_str == line:
                return True  # 找到了匹配的MD5
        else:
            return False

def save_md5(md5_str:str):
    "将传入的md5文件记录并保存"
    with open(config.md5_path, 'a',encoding='utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str:str,encoding='utf-8'):
    "将传入的字符串转化为md5字符串"

    "将字符串转化为byte数组"
    str_bytes = input_str.encode(encoding)

    "创建md5对象"
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes) #上传更新数据，格式为零和一
    return md5_obj.hexdigest() #将数据格式转化为16进制


class KnowledgeBaseService(object):
    def __init__(self):
        #如果文件夹不存在则创建，存在跳过
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name, #表名
            embedding_function= DashScopeEmbeddings(model= 'text-embedding-v4'),
            persist_directory=config.persist_directory, #数据库本地存储文件夹
        ) #向量存储实例化，Chroma是向量库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size= config.chunk_size, #分割后的文本段最大长度
            chunk_overlap= config.chunk_overlap, #连续文本之间的重复字符重叠数量
            separators= config.separators, #划分符号
            length_function= len,
        )#文本分割器的对象

    def upload_by_str(self, data:str,filename):
        "将传入的字符串向量化存入数据库中"
        "先得到传入的md5值"
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]数据库已经存储"
        if len(data) > config.max_spliter_char_number:
            knowledge_chunks:list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]
        metadata = {
            "source":filename,
            "create_time":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "operator:":"小王",
        }
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas =[metadata for _ in knowledge_chunks],
        )
        save_md5(md5_hex)
        return "[成功]内容已经加载到数据库"

"""
Chroma 是一个开源的、专为 AI 应用设计的轻量级向量数据库（Vector Database），在 LangChain 生态中主要充当"知识库的存储和检索引擎"，是构建 RAG（检索增强生成）系统的核心组件之一。
Chroma 是什么？
简单来说，Chroma 能把文本（如文档、PDF、网页内容等）通过嵌入模型（Embedding Model）转换成一串高维数字（即"向量"），然后存储起来。当用户提问时，它会通过计算向量之间的语义相似度，找到与问题最相关的文档片段，而不是简单的关键词匹配。
举个例子：你搜索"AI"，Chroma 能同时找到包含"人工智能""机器学习"等语义相关内容，而不仅仅是字面上有"AI"这个词。
在 LangChain 中有什么用？
Chroma 在 LangChain 中主要承担以下角色：
文档向量存储：将加载好的文档（PDF、网页、文本等）自动分块、嵌入并存入本地数据库，支持内存模式（重启后数据丢失）和持久化模式（保存到磁盘）两种方式。
语义检索（Retriever）：通过 as_retriever() 方法将 Chroma 向量库转换为 LangChain 的检索器，在 RAG 流程中负责"根据用户问题，从知识库中找出最相关的文档片段"。
元数据过滤：支持给文档附加元数据（如来源、分类、时间戳等），查询时可以结合向量相似度和元数据条件进行混合检索。
快速原型开发：Chroma 零配置、开箱即用，非常适合学习、测试和小规模项目，不需要额外部署数据库服务。
"""