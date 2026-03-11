from src.core.config import RAGConfig
from src.vector_stores.base import BaseVectorStore
from src.vector_stores.chroma_store import ChromaVectorStore

class VectorStoreFactory:
    @staticmethod
    def get_vector_store(config: RAGConfig) -> BaseVectorStore:
        if config.vector_store_type.lower() == "chroma":
            return ChromaVectorStore(config)
        raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")
