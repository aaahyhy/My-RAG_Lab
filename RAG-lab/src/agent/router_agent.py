import logging
from langchain_core.prompts import PromptTemplate


class RouterAgent:
    def __init__(self, llm, pipelines_dict):
        """
        初始化路由 Agent
        :param llm: 你的判别大模型
        :param pipelines_dict: {"base": pipe, "rerank": pipe, "hyde": pipe}
        """
        # 修复：解开 Ragas 对 LLM 的包装 (InstructorLLM -> 原生 LLM)
        if hasattr(llm, "llm"):
            self.llm = llm.llm
        elif hasattr(llm, "langchain_llm"):
            self.llm = llm.langchain_llm
        else:
            self.llm = llm

        self.pipelines = pipelines_dict

        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are a routing agent for an academic RAG system.
Choose the best retrieval strategy.
Strategies:
base
Use for short factual lookup questions.
rerank
Use for complex technical questions, comparisons, or detailed explanations.
hyde
Use when the query is vague, conceptual, or lacks keywords.

Examples:
Question: What is RLHF?
Strategy: base
Question: Compare PPO and DPO.
Strategy: rerank
Question: Why do jailbreak prompts bypass safety alignment?
Strategy: hyde

Respond with ONLY one word:
base
rerank
hyde

Question: {query}
Strategy:"""
        )

    def invoke(self, inputs):
        """核心路由逻辑"""
        # 1. 提取 Query
        if isinstance(inputs, str):
            query = inputs
        else:
            # 兼容 Ragas 传入的多种字典格式
            query = inputs.get("query") or inputs.get("question") or ""

        # 2. Agent 决策
        try:
            formatted_prompt = self.prompt.format(query=query)

            # 兼容不同大模型调用方式
            if hasattr(self.llm, "invoke"):
                decision = self.llm.invoke(formatted_prompt)
                output = decision.content if hasattr(decision, "content") else str(decision)
            elif hasattr(self.llm, "predict"):
                output = self.llm.predict(formatted_prompt)
            else:
                output = str(self.llm(formatted_prompt))

            # 3. 结果解析与清理
            strategy = output.strip().lower()
            # 模糊匹配，防止输出包含标点或多余文字
            if "base" in strategy:
                strategy = "base"
            elif "hyde" in strategy:
                strategy = "hyde"
            else:
                strategy = "rerank"  # 默认 fallback

        except Exception as e:
            logging.warning(f"Agent routing failed: {e}. Defaulting to rerank.")
            strategy = "rerank"

        logging.info(f"🕵️ Agent decision: [{strategy.upper()}] for query: {query[:50]}...")

        # 4. 执行对应的 Pipeline
        chosen_pipeline = self.pipelines[strategy]

        # 再次确认 pipeline 的调用接口
        if hasattr(chosen_pipeline, "invoke"):
            response = chosen_pipeline.invoke(inputs)
        else:
            response = chosen_pipeline.run(inputs)

        # 5. 将选择记录到结果字典，方便后续分析
        if isinstance(response, dict):
            response["agent_strategy_used"] = strategy

        return response

    def run(self, inputs):
        """核心修复：兼容 Ragas/LangChain 的 .run() 调用方式"""
        return self.invoke(inputs)

    def __call__(self, inputs):
        """核心修复：兼容对象直接调用方式"""
        return self.invoke(inputs)