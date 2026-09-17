md5_path = "./md5.txt"

#chroma
collection_name = 'rag'
persist_directory = './chroma.db'

#spliter
chunk_size = 1000
chunk_overlap = 100
separators=["\n\n", "\n", ".", " ", ""]
max_spliter_char_number = 1000 #文本分割阈值

#检索返回匹配的文档数量
similarity_threshold = 1

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"
