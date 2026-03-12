import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from docling_core.types.doc import PictureItem, TableItem
from langchain_core.documents import Document

from src.core.config import config
from src.document_processing.chunker import DocumentChunker
from src.document_processing.parser import DocumentParser
from src.models.factory import ModelFactory
from src.vector_stores.factory import VectorStoreFactory

_log = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        # Initialize models
        self.embeddings_model = ModelFactory.get_embeddings(config)
        self.llm_model = ModelFactory.get_llm(config)
        self.vision_model = ModelFactory.get_vision_model(config)

        # Initialize vector store
        self.vector_store = VectorStoreFactory.get_vector_store(config)
        self.vector_store.init_store(self.embeddings_model.get_embeddings())

        # Initialize document processing
        self.chunker = DocumentChunker()
        self.parser = DocumentParser(vision_model=self.vision_model.get_vision_model())

        # Store for document metadata
        self.document_metadata: Dict[str, Any] = {}

    async def process_documents(self, doc_paths: List[Path]) -> Dict[str, Any]:
        if not doc_paths:
            _log.warning("No documents to process")
            return {}

        start_time = time.time()
        _log.info(f"Converting {len(doc_paths)} documents...")

        # Non-blocking: Docling runs in a thread so the event loop stays free
        conv_results = await self.parser.convert_all_async(doc_paths)

        valid_pairs = []
        for idx, result in enumerate(conv_results):
            if result.document is None:
                _log.error(f"Failed to convert {doc_paths[idx]}: {result.error}")
            else:
                valid_pairs.append((result, doc_paths[idx]))

        if not valid_pairs:
            _log.warning("No documents were successfully processed")
            return {}

        # Process all documents concurrently (each runs async image captioning inside)
        tasks = [
            self._process_single_document(result, doc_path)
            for result, doc_path in valid_pairs
        ]
        doc_docs_list = await asyncio.gather(*tasks)

        all_langchain_docs = [doc for docs in doc_docs_list for doc in docs]

        if all_langchain_docs:
            _log.info(
                f"Adding {len(all_langchain_docs)} total items to vector store..."
            )
            await self.vector_store.add_documents_async(all_langchain_docs)
        else:
            _log.warning("No documents were successfully processed")

        elapsed = time.time() - start_time
        _log.info(f"Pipeline completed in {elapsed:.2f} seconds")
        return self.document_metadata

    async def _process_single_document(self, result, doc_path: Path) -> List[Document]:
        doc_path_str = str(doc_path)
        doc_name = Path(doc_path_str).name
        _log.info(f"Processing {doc_name}...")

        # Async visual extraction: all images in this doc are captioned concurrently
        visual_elements = await self.parser.extract_visual_elements_async(
            result.document, doc_name
        )
        table_documents = visual_elements.get("tables", [])
        image_documents = visual_elements.get("images", [])

        # Chunking is CPU-light; run in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(None, self.chunker.chunk, result.document)
        _log.info(f"Created {len(chunks)} chunks from {doc_name}")

        langchain_docs = []
        for chunk_idx, chunk in enumerate(chunks):
            page_numbers = (
                list(chunk.meta.page_numbers)
                if hasattr(chunk.meta, "page_numbers") and chunk.meta.page_numbers
                else []
            )
            tables = (
                list(chunk.meta.tables)
                if hasattr(chunk.meta, "tables") and chunk.meta.tables
                else []
            )
            pictures = (
                list(chunk.meta.pictures)
                if hasattr(chunk.meta, "pictures") and chunk.meta.pictures
                else []
            )

            metadata = {
                "document": doc_name,
                "chunk_index": chunk_idx,
                "source_path": doc_path_str,
                "file_type": Path(doc_path_str).suffix,
                "page_numbers": json.dumps(page_numbers),
                "page_number": page_numbers[0] if page_numbers else None,
                "has_tables": str(len(tables) > 0),
                "table_count": len(tables),
                "has_images": str(len(pictures) > 0),
                "image_count": len(pictures),
                "content_type": "text",
            }
            langchain_docs.append(Document(page_content=chunk.text, metadata=metadata))

        self.document_metadata[doc_name] = {
            "path": doc_path_str,
            "pages": len(result.document.pages),
            "tables": len(
                [
                    e
                    for e, _ in result.document.iterate_items()
                    if isinstance(e, TableItem)
                ]
            ),
            "images": len(
                [
                    e
                    for e, _ in result.document.iterate_items()
                    if isinstance(e, PictureItem)
                ]
            ),
            "chunks": len(chunks),
        }

        _log.info(f"Generated {len(table_documents)} table documents")
        _log.info(f"Generated {len(image_documents)} image documents")

        langchain_docs.extend(table_documents)
        langchain_docs.extend(image_documents)
        return langchain_docs

    def search(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.search(query, k=k)

    @staticmethod
    def _extract_page_from_query(query: str) -> Optional[int]:
        match = re.search(r"\bpage\s+(\d+)\b", query, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def search_by_page(self, query: str, page_num: int, k: int = 5) -> List[Document]:
        return self.vector_store.search_by_page(query, page_num, k=k)

    def search_by_type(
        self, query: str, content_type: str, k: int = 5
    ) -> List[Document]:
        return self.vector_store.search_by_type(query, content_type, k=k)

    def search_with_filter(
        self, query: str, k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        return self.vector_store.search_with_filter(query, k=k, filter_dict=filter_dict)

    def get_retriever(self, k: int = 5):
        return self.vector_store.get_retriever(k=k)

    def get_rag_context(self, query: str, k: int = 5) -> str:
        page_num = self._extract_page_from_query(query)
        if page_num:
            docs = self.search_by_page(query, page_num, k=k)
            if not docs:
                # Fallback to semantic search if no page-filtered results (e.g. not yet re-indexed)
                docs = self.search(query, k=k)
        else:
            docs = self.search(query, k=k)

        if not docs:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("document", "Unknown")
            content_type = doc.metadata.get("content_type", "text")

            page_info = ""
            if (
                doc.metadata.get("page_numbers")
                and doc.metadata["page_numbers"] != "[]"
            ):
                page_info = f" (page {doc.metadata['page_numbers']})"

            type_indicator = f"[{content_type.upper()}]"

            context_parts.append(
                f"[{i + 1}] {type_indicator} From {source}{page_info}:\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)

    async def answer_query(self, query: str, k: int = 5) -> str:
        context = self.get_rag_context(query, k=k)
        print(f"\n🔍 Retrieved context for query:\n{context}\n")

        if context == "No relevant documents found.":
            return "I couldn't find any relevant information in the documents to answer your question."

        prompt = f"""Answer the question based on the following context. If the context doesn't contain relevant information, say so.

                    Context:
                    {context}

                    Question: {query}

                    Answer:
                """

        try:
            llm = self.llm_model.get_llm()
            response = await llm.ainvoke(prompt)

            docs = self.search(query, k=k)
            sources = list(
                set([doc.metadata.get("document", "Unknown") for doc in docs])
            )
            content_types = list(
                set([doc.metadata.get("content_type", "text") for doc in docs])
            )

            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

            if sources:
                response_content = (
                    f"{response_content}\n\n📚 Sources: {', '.join(sources)}"
                )
                response_content = (
                    f"{response_content}\n📄 Content types: {', '.join(content_types)}"
                )

            return response_content
        except Exception as e:
            _log.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {e}"

    def query_tables(self, query: str, k: int = 3) -> List[Document]:
        return self.search_by_type(query, "table", k)

    def query_images(self, query: str, k: int = 3) -> List[Document]:
        return self.search_by_type(query, "image", k)

    def get_document_summary(self) -> Dict[str, Any]:
        return self.document_metadata

    def count_documents(self) -> int:
        return self.vector_store.count_documents()
