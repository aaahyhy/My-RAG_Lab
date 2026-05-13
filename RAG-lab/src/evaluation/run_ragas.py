import json
import time
from tqdm import tqdm
from datasets import Dataset
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


def run_ragas(pipeline, dataset_path, llm=None, embeddings=None, save_path=None):

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in tqdm(data, desc="Running RAG pipeline"):

        q = item["question"]
        gt = item["ground_truth"]
        try:
            rag_result = pipeline.run(q)
            time.sleep(1)
        except Exception as e:
            print(f"Pipeline error on question: {q}")
            print(e)
            rag_result = {"answer": "", "contexts": []}

        questions.append(q)
        answers.append(rag_result.get("answer", ""))
        contexts.append(rag_result.get("contexts", []))
        ground_truths.append(gt)


    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    for m in metrics:
        if hasattr(m, "n"):
            m.n = 1
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        # 强制单线程(max_workers=1)，把每次请求的超时时间拉长到 120 秒，允许重试 10 次
        run_config=RunConfig(
            max_workers=1,
            timeout=120,
            max_retries=10
        )
    )

    df = result.to_pandas()

    scores = {m.name: df[m.name].mean() for m in metrics}

    print("Average scores:", scores)

    return scores, df