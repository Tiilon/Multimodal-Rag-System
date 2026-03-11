from langchain_groq import ChatGroq

from src.core.config import RAGConfig
from src.models.base import BaseLLMModel, BaseVisionModel


class GroqLLM(BaseLLMModel):
    def __init__(self, config: RAGConfig):
        self.llm = ChatGroq(model=config.llm_model, temperature=config.llm_temperature)

    def get_llm(self):
        return self.llm


class GroqVisionLLM(BaseVisionModel):
    def __init__(self, config: RAGConfig):
        self.llm = ChatGroq(
            model=config.vision_model,
            temperature=config.vision_temperature,
        )

    def get_vision_model(self):
        return self.llm
