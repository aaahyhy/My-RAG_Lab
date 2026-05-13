from langchain_openai import OpenAIEmbeddings


#  阿里
from langchain_community.embeddings import DashScopeEmbeddings

def get_embedding_model():
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key="your-access-key",
    )
    return embeddings





#   百度千帆 (Qianfan)
# from langchain_community.embeddings import QianfanEmbeddingsEndpoint
#
# def get_embedding_model():
#     embeddings = QianfanEmbeddingsEndpoint(
#         model="Embedding-V1",                    # 具体模型名称
#         qianfan_ak="your-access-key",            # 千帆 API Key
#         qianfan_sk="your-secret-key",            # 千帆 Secret Key
#     )
#     return embeddings
#
# # 智谱ai
# from langchain_community.embeddings import ZhipuAIEmbeddings
#
# def get_embedding_model():
#     embeddings = ZhipuAIEmbeddings(
#         model="embedding-2",                     # 或 "text-embedding-3" 等
#         zhipuai_api_key="your-api-key",          # 从智谱AI控制台获取
#     )
#     return embeddings

