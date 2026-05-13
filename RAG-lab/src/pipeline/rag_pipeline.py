from src.generation.prompt_builder import build_prompt

class RAGPipeline:
    def __init__(
        self,
        retriever,
        generator,
        reranker=None,
        query_transform=None
    ):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.query_transform = query_transform

    def run(self, query):
        original_query = query

        # 1. Query transform (如 HyDE)
        search_query = query
        if self.query_transform:
            search_query = self.query_transform.transform(query)
            print("DEBUG: 1. Query 转换完成 (HyDE)")

        # 2. Retrieve (粗排)
        docs = self.retriever.retrieve(search_query)
        print(f"DEBUG: 2. 向量检索完成，[粗排] 找回了 {len(docs)} 条文档")

        # 3. Rerank (精排)
        if self.reranker:
            # [修改点] 重排序应该始终基于用户的 original_query，而不是基于 transform 后的 query
            docs = self.reranker.rerank(original_query, docs)
            print(f"DEBUG: 3. Cross-Encoder 排序完成，[精排] 截断保留 {len(docs)} 条文档")

        contexts = [d.page_content for d in docs]

        # 4. Build prompt
        prompt = build_prompt(original_query, docs)

        # 5. Generation
        answer = self.generator.generate(prompt)
        print("DEBUG: 4. LLM 答案生成完毕")

        return {
            "answer": answer,
            "contexts": contexts
        }

    def invoke(self, inputs):
        return self.run(inputs)
