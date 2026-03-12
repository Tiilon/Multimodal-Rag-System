from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    @abstractmethod
    def init_store(self, embeddings: Any):
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]):
        pass

    @abstractmethod
    async def add_documents_async(self, documents: List[Document]):
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Document]:
        pass

    @abstractmethod
    def search_by_page(self, query: str, page_num: int, k: int = 5) -> List[Document]:
        pass

    @abstractmethod
    def search_by_type(
        self, query: str, content_type: str, k: int = 5
    ) -> List[Document]:
        pass

    @abstractmethod
    def search_with_filter(
        self, query: str, k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        pass

    @abstractmethod
    def get_retriever(self, k: int = 5):
        pass

    @abstractmethod
    def count_documents(self) -> int:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        pass
