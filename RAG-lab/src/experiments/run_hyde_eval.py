import os
import time
import logging
import pandas as pd
from src.vectorstore.chroma_store import load_vectorstore
from src.embeddings.embedding_model import get_embedding_model
from src.retrieval.base_retriever import BaseRetriever
from src.generation.llm_generator import LLMGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.generation.llm_generator import get_llm
from src.evaluation.run_ragas import run_ragas
from src.query_transform.hyde import HyDE

# =========================
# 加载 embedding
# =========================
embedding_model = get_embedding_model()

# =========================
# 获取项目路径
# =========================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

db_path = os.path.join(project_root, "vector_db2")
print(f"正在尝试从该路径加载向量库: {db_path}")

# =========================
# 创建结果保存目录
# =========================
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# 配置日志模块（同时输出到文件和终端）
# =========================
log_file_path = os.path.join(RESULTS_DIR, "experiment_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# =========================
# 加载向量库
# =========================
vectordb = load_vectorstore(embedding_model, db_path)
logging.info(f"向量库加载完成，实际存储条目数: {vectordb._collection.count()}")

# =========================
# 构建 Retriever
# =========================
retriever = BaseRetriever(vectordb)

# =========================
# 构建 Generator
# =========================
generator = LLMGenerator()

# =========================
# HyDE Query Transform
# =========================
hyde = HyDE(generator)

# =========================
# 构建 RAG Pipeline (HyDE)
# =========================
pipeline = RAGPipeline(
    retriever=retriever,
    generator=generator,
    query_transform=hyde
)

# =========================
# Evaluation LLM
# =========================
llm = get_llm()

# =========================
# 数据集路径
# =========================
dataset_path = os.path.join(project_root, "data/evaluation/ragas_testset.json")

# =========================
# 运行 RAGAS 评测并保存结果
# =========================
# 为当前实验创建子目录（模式名：hyde）
mode = "hyde"
save_dir = os.path.join(RESULTS_DIR, mode)
os.makedirs(save_dir, exist_ok=True)
predictions_path = os.path.join(save_dir, "predictions.json")

logging.info(f"开始运行评测，模式: {mode}")
start_time = time.time()

# 可选：添加 Token 监控（如果希望记录 token 消耗，需取消注释并导入 get_openai_callback）
from langchain_community.callbacks import get_openai_callback
with get_openai_callback() as cb:
    scores, df = run_ragas(
        pipeline=pipeline,
        dataset_path=dataset_path,
        llm=llm,
        embeddings=embedding_model,
        save_path=predictions_path
    )
    tokens_used = cb.total_tokens
    cost = cb.total_cost
# # 如果不使用 Token 监控，直接调用
# scores, df = run_ragas(
#     pipeline=pipeline,
#     dataset_path=dataset_path,
#     llm=llm,
#     embeddings=embedding_model,
#     save_path=predictions_path
# )
end_time = time.time()
elapsed_time = end_time - start_time

# 保存 predictions.json（如果 run_ragas 内部已经保存，可跳过，但为确保完整性再次保存）
if df is not None and not df.empty:
    # 使用 orient="records" 保存为 JSON 对象数组
    df.to_json(predictions_path, orient="records", force_ascii=False, indent=4)
    logging.info(f"预测详情已成功保存至: {predictions_path}")
else:
    logging.warning("未返回有效的 DataFrame，无法保存 predictions.json")

# 记录评测结果（scores 字典包含各项指标）
logging.info(f"⏱️ 耗时: {elapsed_time:.2f} 秒")
if 'tokens_used' in locals():  # 如果开启了 token 监控
    logging.info(f"🪙 Token消耗: {tokens_used} (预估成本: ${cost:.4f})")
    scores["time_seconds"] = round(elapsed_time, 2)
    scores["total_tokens"] = tokens_used
else:
    scores["time_seconds"] = round(elapsed_time, 2)

# 保存对比文件（虽然只有一个模式，但可以统一格式）
comparison_path = os.path.join(RESULTS_DIR, "hyde_eval_last.csv")
df_results = pd.DataFrame([scores])
df_results.insert(0, "method", mode)  # 将 mode 作为第一列
df_results.to_csv(comparison_path, index=False)

logging.info(f"对比结果已保存至: {comparison_path}")
logging.info(f"完整日志已保存至: {log_file_path}")

print("\n===== 评测完成 =====")
print(f"评测指标: {scores}")
print(f"预测详情保存至: {predictions_path}")
print(f"对比文件保存至: {comparison_path}")