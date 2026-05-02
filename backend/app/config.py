from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"  # override with https://api.deepseek.com for DeepSeek

    qdrant_local_path: str = "data/qdrant_db"
    qdrant_collection: str = "tax_docs"

    embed_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "deepseek-chat"  # or gpt-4o-mini for OpenAI

    max_retries: int = 2
    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
