from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        pass

class BaseLLMModel(ABC):
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        pass

class BaseVisionModel(ABC):
    @abstractmethod
    def get_vision_model(self) -> BaseChatModel:
        pass
