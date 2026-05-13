from src.retrieval.retriever_factory import build_retriever
from src.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.query_transform.hyde import HyDE
from src.pipeline.rag_pipeline import RAGPipeline
from src.generation.llm_generator import LLMGenerator

def build_pipeline(mode, vectordb):
    """
    Build RAG pipeline according to experiment mode.

    Modes:
    - base
    - rerank
    - hyde
    - hybrid
    """
    print(f"Building pipeline mode: {mode}")
    generator = LLMGenerator()

    # 统一通过 factory 构建 retriever
    retriever = build_retriever(mode, vectordb)

    # =========================
    # [修改点] 动态设置粗排召回数量
    # =========================
    # 如果是 rerank 模式，粗排池扩大到 15 条；否则只召回最终需要的 3 条
    retrieve_k = 15 if mode == "rerank" else 3
    if hasattr(retriever, 'k'):
        retriever.k = retrieve_k

    # =========================
    # Base RAG
    # =========================
    if mode == "base":
        return RAGPipeline(
            retriever=retriever,
            generator=generator
        )

    # =========================
    # Rerank RAG
    # =========================
    elif mode == "rerank":
        reranker = CrossEncoderReranker() # 默认内部已经是 top_k=3
        return RAGPipeline(
            retriever=retriever,
            generator=generator,
            reranker=reranker
        )

    # =========================
    # HyDE RAG
    # =========================
    elif mode == "hyde":
        hyde = HyDE(generator)
        return RAGPipeline(
            retriever=retriever,
            generator=generator,
            query_transform=hyde
        )

    # =========================
    # Hybrid RAG
    # =========================
    elif mode == "hybrid":
        return RAGPipeline(
            retriever=retriever,
            generator=generator
        )

    # =========================
    # Unknown
    # =========================
    else:
        raise ValueError(f"Unknown mode: {mode}")
