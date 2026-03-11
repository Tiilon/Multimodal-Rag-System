import logging
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.core.config import RAGConfig
from src.vector_stores.base import BaseVectorStore

_log = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store: Optional[Chroma] = None

    def init_store(self, embeddings: Any):
        _log.info(f"Initializing vector store at {self.config.persist_directory}")
        self.vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=self.config.persist_directory,
        )
        _log.info(
            f"✅ Vector store initialized with collection: {self.config.collection_name}"
        )

    def add_documents(self, documents: List[Document]):
        if not self.vector_store:
            _log.error("Vector store not initialized")
            return
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vector_store.add_documents(batch)
            _log.info(
                f"Added batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}"
            )
        _log.info("✅ Documents added to store")

    def search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vector_store:
            _log.error("Vector store not initialized")
            return []
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            docs_with_scores = []
            for doc, score in results:
                doc.metadata["relevance_score"] = float(score)
                docs_with_scores.append(doc)
            return docs_with_scores
        except Exception as e:
            _log.error(f"Search failed: {e}")
            return []

    def search_by_type(
        self, query: str, content_type: str, k: int = 5
    ) -> List[Document]:
        if not self.vector_store:
            _log.error("Vector store not initialized")
            return []
        try:
            return self.vector_store.similarity_search(
                query, k=k, filter={"content_type": content_type}
            )
        except Exception as e:
            _log.error(f"Typed search failed: {e}")
            return []

    def search_with_filter(
        self, query: str, k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        if not self.vector_store:
            _log.error("Vector store not initialized")
            return []
        try:
            return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
        except Exception as e:
            _log.error(f"Filtered search failed: {e}")
            return []

    def get_retriever(self, k: int = 5):
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def count_documents(self) -> int:
        if not self.vector_store:
            return 0
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def delete_collection(self, collection_name: str) -> bool:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.config.persist_directory)
            client.delete_collection(name=collection_name)
            _log.info(f"Deleted Chroma collection: {collection_name}")
            return True
        except Exception as e:
            _log.error(f"Failed to delete Chroma collection '{collection_name}': {e}")
            return False
