import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore as LangchainQdrantVectorStore
from qdrant_client import QdrantClient

from src.core.config import RAGConfig
from src.vector_stores.base import BaseVectorStore

_log = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store: Optional[LangchainQdrantVectorStore] = None

    def init_store(self, embeddings: Any):
        if self.config.qdrant_url:
            _log.info(f"Initializing Qdrant store at URL: {self.config.qdrant_url}")
            client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
            )
        else:
            _log.info(f"Initializing Qdrant store at path: {self.config.qdrant_path}")
            client = QdrantClient(path=self.config.qdrant_path)

        try:
            if not client.collection_exists(self.config.collection_name):
                _log.info(f"Collection {self.config.collection_name} does not exist. Creating it now...")
                # Determine embedding dimension
                dummy_vector = embeddings.embed_query("init_qdrant")
                from qdrant_client.http.models import VectorParams, Distance
                client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(size=len(dummy_vector), distance=Distance.COSINE),
                )
        except Exception as e:
            _log.warning(f"Could not check or create collection proactively: {e}")

        self.vector_store = LangchainQdrantVectorStore(
            client=client,
            collection_name=self.config.collection_name,
            embedding=embeddings,
        )

        _log.info(
            f"✅ Qdrant store initialized with collection: {self.config.collection_name}"
        )

    def add_documents(self, documents: List[Document]):
        if not self.vector_store:
            # If from_existing_collection failed or wasn't called properly, we might need to initialize it from texts first
            _log.error(
                "Vector store not initialized. Will attempt to initialize from documents to create collection."
            )
            return

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            uuids = [str(uuid4()) for _ in range(len(batch))]
            self.vector_store.add_documents(documents=batch, ids=uuids)
            _log.info(
                f"Added batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}"
            )
        _log.info("✅ Documents added to Qdrant store")

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
            # Qdrant uses a payload filter. langchain-qdrant supports passing a kwargs filter.
            # Using the simpler kwargs filter if supported, or constructing a Qdrant filter
            from qdrant_client.http import models as rest

            payload_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.content_type",
                        match=rest.MatchValue(value=content_type),
                    )
                ]
            )
            return self.vector_store.similarity_search(
                query, k=k, filter=payload_filter
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
            from qdrant_client.http import models as rest

            must_conditions = []
            if filter_dict:
                for key, value in filter_dict.items():
                    must_conditions.append(
                        rest.FieldCondition(
                            key=f"metadata.{key}",
                            match=rest.MatchValue(value=value),
                        )
                    )

            payload_filter = (
                rest.Filter(must=must_conditions) if must_conditions else None
            )

            return self.vector_store.similarity_search(
                query, k=k, filter=payload_filter
            )
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
            return self.vector_store.client.count(
                collection_name=self.config.collection_name
            ).count
        except Exception:
            return 0

    def delete_collection(self, collection_name: str) -> bool:
        try:
            if self.config.qdrant_url:
                client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                )
            else:
                client = QdrantClient(path=self.config.qdrant_path)
            
            client.delete_collection(collection_name=collection_name)
            _log.info(f"Deleted Qdrant collection: {collection_name}")
            return True
        except Exception as e:
            _log.error(f"Failed to delete Qdrant collection '{collection_name}': {e}")
            return False
