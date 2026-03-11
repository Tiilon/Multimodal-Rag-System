from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.core.config import RAGConfig
from src.models.base import BaseEmbeddingModel, BaseLLMModel, BaseVisionModel

class OllamaEmbedding(BaseEmbeddingModel):
    def __init__(self, config: RAGConfig):
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url,
        )
        
    def get_embeddings(self):
        return self.embeddings

class OllamaLLM(BaseLLMModel):
    def __init__(self, config: RAGConfig):
        self.llm = ChatOllama(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            temperature=config.llm_temperature,
            num_predict=config.llm_num_predict,
        )
        
    def get_llm(self):
        return self.llm

class OllamaVisionLLM(BaseVisionModel):
    def __init__(self, config: RAGConfig):
        self.llm = ChatOllama(
            model=config.vision_model,
            base_url=config.ollama_base_url,
            temperature=config.vision_temperature,
        )
        
    def get_vision_model(self):
        return self.llm
