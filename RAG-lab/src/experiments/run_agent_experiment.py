import os
import pandas as pd
import time
import logging

from src.pipeline.pipeline_factory import build_pipeline
from src.evaluation.run_ragas import run_ragas
from src.embeddings.embedding_model import get_embedding_model
from src.generation.llm_generator import get_llm
from src.generation.llm_generator import get_chat_model
from src.vectorstore.chroma_store import load_vectorstore
from src.agent.router_agent import RouterAgent


# 引入我们在跑全局脚本时用到的拦截器，确保耗时对比在一个维度上
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
# 路径配置
# =========================
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

DATASET_PATH = os.path.join(project_root, "data/evaluation/ragas_testset.json")
VECTOR_DB_PATH = os.path.join(project_root, "vector_db2")
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

log_file_path = os.path.join(RESULTS_DIR, "agent_experiment_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, encoding='utf-8'), logging.StreamHandler()]
)


def main():
    logging.info("===== Starting Agent-Improved RAG Experiment =====")

    # 1. 评估用的 LLM (Ragas 专用)
    eval_llm = get_llm()

    # 2. Agent 决策用的 LLM (原生 LangChain)
    agent_decision_llm = get_chat_model()

    embeddings = get_embedding_model()
    vectordb = load_vectorstore(embeddings, VECTOR_DB_PATH)

    # 3. 实例化子 Pipeline (这里如果你之前的 pipeline 内部也用了 get_llm，记得也要换成 get_chat_model)
    pipelines_dict = {
        "base": build_pipeline("base", vectordb),
        "rerank": build_pipeline("rerank", vectordb),
        "hyde": build_pipeline("hyde", vectordb)
    }

    # 4. 实例化 Agent：传入那个“干净”的模型
    raw_agent_pipeline = RouterAgent(llm=agent_decision_llm, pipelines_dict=pipelines_dict)

    # 5. 包装计时器
    timed_agent_pipeline = TimingPipelineWrapper(raw_agent_pipeline)
    mode = "agent_improved"
    save_dir = os.path.join(RESULTS_DIR, mode)
    os.makedirs(save_dir, exist_ok=True)
    predictions_path = os.path.join(save_dir, "predictions.json")

    # 6. 运行 Ragas 评估
    scores, df = run_ragas(
        pipeline=timed_agent_pipeline,
        dataset_path=DATASET_PATH,
        llm=eval_llm,  # 评估依然用带马甲的
        embeddings=embeddings,
        save_path=predictions_path
    )

    if df is not None and not df.empty:
        df.to_json(predictions_path, orient="records", force_ascii=False, indent=4)
        if 'agent_strategy_used' in df.columns:
            logging.info(f"📊 Agent 路由分布: \n{df['agent_strategy_used'].value_counts()}")

    # 获取真实耗时
    total_time = timed_agent_pipeline.total_inference_time
    avg_time = total_time / timed_agent_pipeline.call_count if timed_agent_pipeline.call_count > 0 else 0

    scores["inference_time_seconds"] = round(total_time, 2)
    scores["avg_latency_seconds"] = round(avg_time, 2)
    scores["method"] = mode

    logging.info(f"⏱️ 真实推理总耗时: {total_time:.2f} 秒 (平均单次: {avg_time:.2f} 秒)")

    # 写入表格
    comparison_path = os.path.join(RESULTS_DIR, "comparison.csv")
    if os.path.exists(comparison_path):
        existing_df = pd.read_csv(comparison_path)
        existing_df = existing_df[existing_df['method'] != mode]
        # 删除旧版没有的 token 字段（如果有的话）避免报错
        if 'total_tokens' in existing_df.columns:
            existing_df = existing_df.drop(columns=['total_tokens', 'time_seconds'], errors='ignore')
        updated_df = pd.concat([existing_df, pd.DataFrame([scores])], ignore_index=True)
    else:
        updated_df = pd.DataFrame([scores])

    cols = ['method'] + [c for c in updated_df.columns if c != 'method']
    updated_df = updated_df[cols]
    updated_df.to_csv(comparison_path, index=False)
    print("\n===== Final Experiment Summary =====")
    print(updated_df)


if __name__ == "__main__":
    main()