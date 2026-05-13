import os
from openai import OpenAI
from ragas.llms import llm_factory
from langchain_openai import ChatOpenAI
class LLMGenerator:
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY", "sk-45e1dce70a884ed0834660f4e48e6505")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=5,  # 增加底层自动重试
            timeout=120  # 增加超时时间
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def get_llm(model: str = "deepseek-chat"):

    api_key = os.getenv("OPENAI_API_KEY", "your-access-key")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return llm_factory(model, client=client)


def get_chat_model(model: str = "deepseek-chat"):
    api_key = os.getenv("OPENAI_API_KEY", "your-access-key")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        max_retries=5
    )