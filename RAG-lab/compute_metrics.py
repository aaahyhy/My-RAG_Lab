import json
import os
from pathlib import Path

def compute_and_save_metrics(json_path):
    """计算 predictions.json 中所有样本的指标平均值并保存到 _metrics.json"""
    # 读取 predictions.json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 定义需要计算的指标
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

    # 收集每个指标的数值列表
    scores = {metric: [] for metric in metrics}

    for item in data:
        for metric in metrics:
            val = item.get(metric)
            if val is not None:
                scores[metric].append(val)

    # 计算平均值
    avg_scores = {}
    for metric in metrics:
        if scores[metric]:
            avg_scores[metric] = sum(scores[metric]) / len(scores[metric])
        else:
            avg_scores[metric] = None

    # 确定输出路径（与输入同目录，文件名为 predictions_metrics.json）
    output_path = json_path.parent / "predictions_metrics.json"

    # 保存为 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(avg_scores, f, indent=2, ensure_ascii=False)

    print(f"✅ 已计算 {json_path.parent.name} 的指标平均值，保存至 {output_path}")
    print(avg_scores)

def main():

    results_dir = Path(r"/results0")
    if not results_dir.exists():
        print("未找到 results 目录，请确认路径是否正确。")
        return

    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            json_path = subdir / "predictions.json"
            if json_path.exists():
                compute_and_save_metrics(json_path)
            else:
                print(f"⚠️ 跳过 {subdir.name}，未找到 predictions.json")

if __name__ == "__main__":
    main()