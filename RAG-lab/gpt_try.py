import os
import asyncio
import json
import re
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.persona import Persona
from ragas.run_config import RunConfig

# ==========================================
# 核心修复：自定义 LLM 包装器，清理 Markdown 标签
# ==========================================
class DeepSeekWrapper(ChatOpenAI):
    def _clean_content(self, text: str) -> str:
        if not text:
            return text
        # 正则提取：只保留 ```json ... ``` 内部或去掉标签后的内容
        cleaned = re.sub(r"```json\s?|```", "", text).strip()
        return cleaned

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            gen.message.content = self._clean_content(gen.message.content)
            if hasattr(gen, 'text'):
                gen.text = self._clean_content(gen.text)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = await super()._agenerate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            gen.message.content = self._clean_content(gen.message.content)
            if hasattr(gen, 'text'):
                gen.text = self._clean_content(gen.text)
        return result

# ---------------------------
# 过滤逻辑
# ---------------------------
def is_trivial_question(question: str) -> bool:
    q = str(question).lower()
    trivial_patterns = [r"who is", r"who wrote", r"what is the title"]
    return any(re.search(p, q) for p in trivial_patterns)

def is_context_too_short(contexts: list) -> bool:
    if not isinstance(contexts, list): return True
    return sum(len(str(c)) for c in contexts) < 20

# ---------------------------
# 主逻辑
# ---------------------------
async def main():
    PDF_DIR = "data/documents"
    OUTPUT_FILE = "data/evaluation/ragas_testset.json"
    TESTSET_SIZE = 50

    # 1. LLM 初始化
    import httpx
    http_client = httpx.Client(
        timeout=httpx.Timeout(connect=60.0, read=300.0, write=60.0, pool=60.0)
    )
    generator_llm = DeepSeekWrapper(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0,
        max_retries=5,
        http_client=http_client
    )

    # 2. Embedding 初始化 (注意 model_name)
    emb = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. 加载文档
    from src.ingestion.document_loader import load_documents
    from src.ingestion.text_splitter import split_documents
    print("正在加载文档并切分...")
    docs = load_documents(PDF_DIR)
    chunks = split_documents(docs, strategy="hierarchical")
    chunks = [c for c in chunks if len(c.page_content.strip()) > 30]

    # 4. 生成器初始化
    default_persona = Persona(
        name="AI Researcher",
        role_description="An expert in AI security and multimodal LLM safety."
    )
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=emb,
        persona_list=[default_persona]
    )

    # 5. 定义分布
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
    ]
    run_config = RunConfig(max_workers=1, timeout=240)

    # --- 开始生成 ---
    print(f"🚀 开始全量生成测试集 (目标: {TESTSET_SIZE} 条)...")
    try:
        dataset = generator.generate_with_langchain_docs(
            chunks,
            testset_size=TESTSET_SIZE,
            query_distribution=query_distribution,
            run_config=run_config,
            raise_exceptions=False
        )
        df = dataset.to_pandas()
    except Exception as e:
        print(f"❌ 运行核心报错: {e}")
        return

    # --- 6. 后处理与过滤 (在 try 块外运行) ---
    results = []
    if df is None or df.empty:
        print("⚠️ 未能生成有效数据。")
        return

    for _, row in df.iterrows():
        question = row.get("user_input")
        answer = row.get("reference")
        contexts = row.get("reference_contexts")

        # 核心过滤逻辑：防止 NaN 导致的列表歧义
        if not isinstance(question, str) or len(question.strip()) == 0:
            continue
        if not isinstance(contexts, list):
            continue
        if is_trivial_question(question) or is_context_too_short(contexts):
            continue

        results.append({
            "id": len(results),
            "question": str(question),
            "ground_truth": str(answer),
            "contexts": [{"content": str(c)} for c in contexts],
            "metadata": {"synthesizer": row.get("synthesizer_name", "unknown")}
        })

    # 保存文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功！最终保存了 {len(results)} 条有效数据到 {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())