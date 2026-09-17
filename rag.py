from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from vector_stores import VectorStoresServices
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
import config_data as config

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagServices(object):
    def __init__(self):
        self.vector_stores = VectorStoresServices(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate(
            [
                ("system","以我提供的已知资料为主,"
                 "简洁专业的回答用户问题。参考资料{context}。"),
                ("user","请回答用户提问{input}")
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.chain = self.get_chain()

    def get_chain(self):
        "获取最终的执行链"
        retriever = self.vector_stores.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            #格式化检索到的文档
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"

            return formatted_str

        #创建执行链
        chain = (
            {
                "input": RunnablePassthrough(),
                "context": retriever | format_document
            } | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        return chain

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    res = RagServices().chain.invoke("我体重110斤，请给我提供尺码推荐")
    print(res)
