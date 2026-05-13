import os
import pandas as pd
import time
import logging
from src.pipeline.pipeline_factory import build_pipeline
from src.evaluation.run_ragas import run_ragas
from src.embeddings.embedding_model import get_embedding_model
from src.generation.llm_generator import get_llm
from src.vectorstore.chroma_store import load_vectorstore

# =========================
# 新增：精准耗时拦截器 (Timing Wrapper)
# 作用：只记录 pipeline.invoke() 的执行时间，排除 Ragas 评估打分的耗时
# =========================
class TimingPipelineWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.total_inference_time = 0.0
        self.call_count = 0

    def invoke(self, inputs):
        start = time.time()
        # 兼容性处理：如果内部 pipeline 只有 run 没有 invoke
        if hasattr(self.pipeline, "invoke"):
            result = self.pipeline.invoke(inputs)
        else:
            result = self.pipeline.run(inputs)

        self.total_inference_time += (time.time() - start)
        self.call_count += 1
        return result

    def run(self, inputs):
        """核心修复：显式提供 run 方法供 Ragas 调用"""
        return self.invoke(inputs)

    def __call__(self, inputs):
        """兜底：支持直接对象调用"""
        return self.invoke(inputs)

# =========================
# 路径与日志配置
# =========================
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

DATASET_PATH = os.path.join(project_root, "data/evaluation/ragas_testset.json")
VECTOR_DB_PATH = os.path.join(project_root, "vector_db2")
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

log_file_path = os.path.join(RESULTS_DIR, "experiment_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def run_experiment(mode, vectordb, llm, embeddings):
    logging.info(f"\n===== Running experiment: {mode} =====")

    # 构建并包装 Pipeline
    raw_pipeline = build_pipeline(mode, vectordb)
    timed_pipeline = TimingPipelineWrapper(raw_pipeline)

    save_dir = os.path.join(RESULTS_DIR, mode)
    os.makedirs(save_dir, exist_ok=True)
    predictions_path = os.path.join(save_dir, "predictions.json")

    # 执行评估
    scores, df = run_ragas(
        pipeline=timed_pipeline,  # 传入带有计时器的 wrapper
        dataset_path=DATASET_PATH,
        llm=llm,
        embeddings=embeddings,
        save_path=predictions_path
    )

    if df is not None and not df.empty:
        df.to_json(predictions_path, orient="records", force_ascii=False, indent=4)
        logging.info(f"✅ [{mode}] 预测详情已成功保存至: {predictions_path}")
    else:
        logging.warning(f"⚠️ [{mode}] 未返回有效的 DataFrame，无法保存 predictions.json")

    # 提取真实检索与生成耗时
    total_time = timed_pipeline.total_inference_time
    avg_time = total_time / timed_pipeline.call_count if timed_pipeline.call_count > 0 else 0

    logging.info(f"⏱️ [{mode}] 真实推理总耗时: {total_time:.2f} 秒 (平均单次: {avg_time:.2f} 秒)")

    # 存入 scores，替换掉原来的虚假耗时
    scores["inference_time_seconds"] = round(total_time, 2)
    scores["avg_latency_seconds"] = round(avg_time, 2)

    return scores


def main():
    modes = ["base", "rerank", "hyde"]

    logging.info("Loading embedding model...")
    embeddings = get_embedding_model()

    logging.info("Loading vector database...")
    vectordb = load_vectorstore(embeddings, VECTOR_DB_PATH)

    logging.info("Loading evaluation LLM...")
    llm = get_llm()

    all_results = []
    for mode in modes:
        try:
            scores = run_experiment(mode, vectordb, llm, embeddings)
            scores["method"] = mode
            all_results.append(scores)
        except Exception as e:
            logging.error(f"❌ Experiment {mode} failed: {e}", exc_info=True)

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ['method'] + [c for c in df.columns if c != 'method']
        df = df[cols]
        comparison_path = os.path.join(RESULTS_DIR, "comparison.csv")
        df.to_csv(comparison_path, index=False)

        print("\n===== Experiment Summary =====")
        print(df)
        print(f"\nSaved comparison to: {comparison_path}")
    else:
        print("No successful experiments.")


if __name__ == "__main__":
    main()