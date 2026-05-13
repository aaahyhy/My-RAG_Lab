# My-RAG_Lab

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-orange.svg)

## 📌 项目简介
A modular RAG experiment &amp; evaluation framework. Supports Base, HyDE, and Rerank strategies with automated RAGAS metrics. Designed for academic research and engineering practice.
**RAG-Lab** 是一个专为学术研究与工程实践设计的模块化 RAG (Retrieval-Augmented Generation) 实验框架。本项目通过 **Pipeline Factory** 设计模式实现了各组件的解耦，支持 Base、HyDE、Rerank 等多种策略的快速切换与量化对比。

## ✨ 核心特性

- **🧩 模块化设计**：基于可插拔架构，将 Retriever、Query Expansion、Reranker、Generator 等模块解耦。
- **📄 深度文档治理**：集成 **Docling** 解析引擎，针对双栏学术 PDF 优化解析逻辑，并通过正则策略精准剔除 References 等噪声。
- **🧪 实验驱动**：内置多种检索增强策略（如 **HyDE**、**Cross-Encoder Rerank**），支持一键对比消融实验。
- **📊 自动化评测**：集成 **RAGAS** 框架，提供 Faithfulness、Context Recall、Answer Relevancy 等维度的端到端量化评估。
- **⚡ 资源优化**：针对显存溢出 (OOM) 与 API 延迟进行专项优化，支持受限环境下的高效评测。

## 🏗️ 系统架构

系统采用模块化 Pipeline 设计，核心流程如下：

1.  **数据处理层**：Docling 解析 PDF -> Markdown 清洗 -> Hierarchical Chunking (1200 chars / 150 overlap)。
2.  **索引层**：SentenceTransformers 向量化 -> ChromaDB 矢量存储。
3.  **检索与增强层**：Query 转化 (HyDE) -> 相似度检索 -> 重排序 (Cross-Encoder)。
4.  **生成与评测层**：DeepSeek/GPT 推理生成 -> RAGAS 多指标自动化评估。

## 🚀 快速开始

### 环境配置
```bash
git clone https://github.com/your-username/RAG-Lab.git
cd RAG-Lab
pip install -r requirements.txt
