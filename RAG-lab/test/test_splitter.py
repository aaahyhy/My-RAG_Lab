from src.ingestion.document_loader import load_documents
from src.ingestion.text_splitter import split_documents

docs = load_documents("data/documents")

chunks = split_documents(docs)

print("原始文档数:", len(docs))
print("chunk数量:", len(chunks))

# 统计 chunk 长度
lengths = [len(c.page_content) for c in chunks]

print("平均chunk长度:", sum(lengths) / len(lengths))
print("最大chunk长度:", max(lengths))
print("最小chunk长度:", min(lengths))

print("\n示例 chunk:\n")
print(chunks[0].page_content[:500])

print("\nmetadata:")
print(chunks[0].metadata)
import numpy as np

print("\nchunk长度分布：")
print("p50:", np.percentile(lengths, 50))
print("p75:", np.percentile(lengths, 75))
print("p90:", np.percentile(lengths, 90))
print("p95:", np.percentile(lengths, 95))

print("\nchunk header:")
print(chunks[0].page_content.split("\n")[0])