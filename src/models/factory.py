from src.core.config import RAGConfig
from src.models.groq import GroqLLM, GroqVisionLLM
from src.models.ollama import OllamaEmbedding, OllamaLLM, OllamaVisionLLM


class ModelFactory:
    @staticmethod
    def get_embeddings(config: RAGConfig):
        if config.embedding_type.lower() == "ollama":
            return OllamaEmbedding(config)
        raise ValueError(f"Unsupported embedding type: {config.embedding_type}")

    @staticmethod
    def get_llm(config: RAGConfig):
        if config.llm_type.lower() == "ollama":
            return OllamaLLM(config)
        elif config.llm_type.lower() == "groq":
            return GroqLLM(config)
        raise ValueError(f"Unsupported LLM type: {config.llm_type}")

    @staticmethod
    def get_vision_model(config: RAGConfig):
        if config.vision_type.lower() == "ollama":
            return OllamaVisionLLM(config)
        elif config.vision_type.lower() == "groq":
            return GroqVisionLLM(config)
        raise ValueError(f"Unsupported Vision model type: {config.vision_type}")
