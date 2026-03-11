from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    persist_directory: str = "./rag_storage"
    collection_name: str = "documents"

    # Vector store config
    vector_store_type: str = "chroma"

    # Embedding config
    embedding_type: str = "ollama"
    embedding_model: str = "nomic-embed-text"

    # LLM config
    llm_type: str = "groq"  # Or for ollama use "ollama"
    llm_model: str = "llama-3.3-70b-versatile"  # Or for ollama use llama3.2:latest
    llm_temperature: float = 0.1
    llm_num_predict: int = 2048

    # Vision config
    vision_type: str = "groq"  # Or for ollama use "ollama"
    vision_model: str = (
        "meta-llama/llama-4-scout-17b-16e-instruct"  # Or for ollama use qwen3.5:0.8b
    )
    vision_temperature: float = 0.1

    # External APIs
    ollama_base_url: str = "http://localhost:11434"

    class Config:
        env_prefix = "RAG_"


config = RAGConfig()
