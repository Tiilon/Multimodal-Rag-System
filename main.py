import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Docling imports
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem
from langchain_chroma import Chroma
from langchain_core.documents import Document

# LangChain imports
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str = "./rag_storage",
        collection_name: str = "documents",
        ollama_embedding_model: str = "nomic-embed-text",
        ollama_llm_model: str = "llama3.2:latest",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """Initialize the RAG pipeline with Docling and LangChain Ollama"""

        # Store configuration
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.ollama_embedding_model = ollama_embedding_model
        self.ollama_llm_model = ollama_llm_model
        self.ollama_base_url = ollama_base_url

        # Initialize LangChain Ollama components
        self.embeddings = OllamaEmbeddings(
            model=ollama_embedding_model,
            base_url=ollama_base_url,
        )

        self.llm = OllamaLLM(
            model=ollama_llm_model,
            base_url=ollama_base_url,
            temperature=0.1,
            num_predict=2048,
        )

        # Initialize Docling converter
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=DoclingParseDocumentBackend,
                ),
            }
        )

        # Initialize chunker
        self.tokenizer_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer_model.tokenizer,
            merge_peers=True,
        )

        # Initialize or load vector store
        self._init_vector_store()

        # Store for document metadata (not the vector store itself)
        self.document_metadata: Dict[str, Any] = {}

    def _init_vector_store(self):
        """Initialize or load existing vector store"""
        _log.info(f"Initializing vector store at {self.persist_directory}")

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        _log.info(
            f"✅ Vector store initialized with collection: {self.collection_name}"
        )

    def process_documents(self, doc_paths: List[Path]) -> Dict[str, Any]:
        """Process multiple documents through the complete pipeline"""

        if not doc_paths:
            _log.warning("No documents to process")
            return {}

        start_time = time.time()

        # Step 1: Convert all documents
        _log.info(f"Converting {len(doc_paths)} documents...")
        conv_results = self.doc_converter.convert_all(doc_paths, raises_on_error=False)

        # Step 2: Process each result into LangChain Documents
        all_langchain_docs = []

        for idx, result in enumerate(conv_results):
            if result.document is None:
                _log.error(f"Failed to convert {doc_paths[idx]}: {result.error}")
                continue

            doc_path = str(doc_paths[idx])
            doc_name = Path(doc_path).name
            _log.info(f"Processing {doc_name}...")

            # Step 3: Extract and save tables/images for reference
            self._extract_visual_elements(result.document, doc_name)

            # Step 4: Chunk the document
            chunks = list(self.chunker.chunk(result.document))
            _log.info(f"Created {len(chunks)} chunks from {doc_name}")

            # Step 5: Convert to LangChain Documents with rich metadata
            for chunk_idx, chunk in enumerate(chunks):
                # Safely extract metadata
                page_numbers = []
                if hasattr(chunk.meta, "page_numbers"):
                    page_numbers = (
                        list(chunk.meta.page_numbers) if chunk.meta.page_numbers else []
                    )

                tables = []
                if hasattr(chunk.meta, "tables"):
                    tables = list(chunk.meta.tables) if chunk.meta.tables else []

                pictures = []
                if hasattr(chunk.meta, "pictures"):
                    pictures = list(chunk.meta.pictures) if chunk.meta.pictures else []

                # Create metadata
                metadata = {
                    "document": doc_name,
                    "chunk_index": chunk_idx,
                    "source_path": doc_path,
                    "file_type": Path(doc_path).suffix,
                    "page_numbers": json.dumps(page_numbers),
                    "has_tables": str(len(tables) > 0),
                    "table_count": len(tables),
                    "has_images": str(len(pictures) > 0),
                    "image_count": len(pictures),
                }

                # Create LangChain Document
                langchain_doc = Document(page_content=chunk.text, metadata=metadata)
                all_langchain_docs.append(langchain_doc)

            # Store document-level info
            self.document_metadata[doc_name] = {
                "path": doc_path,
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

        # Step 6: Add all documents to vector store
        if all_langchain_docs:
            _log.info(f"Adding {len(all_langchain_docs)} chunks to vector store...")

            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(all_langchain_docs), batch_size):
                batch = all_langchain_docs[i : i + batch_size]
                self.vector_store.add_documents(batch)
                _log.info(
                    f"Added batch {i // batch_size + 1}/{(len(all_langchain_docs) - 1) // batch_size + 1}"
                )
            _log.info("✅ Documents added to store")
        else:
            _log.warning("No documents were successfully processed")

        elapsed = time.time() - start_time
        _log.info(f"Pipeline completed in {elapsed:.2f} seconds")

        return self.document_metadata

    def _table_to_text(self, table_df):
        """Convert table dataframe into semantic text for embedding"""

        columns = ", ".join(table_df.columns)

        rows = []
        for _, row in table_df.iterrows():
            row_text = ", ".join(f"{col} = {row[col]}" for col in table_df.columns)
            rows.append(row_text)

        table_text = f"Table with columns: {columns}.\nRows:\n" + "\n".join(rows)
        return table_text

    def _extract_visual_elements(self, document, doc_name: str):
        """Extract and save tables and images for later reference"""
        output_dir = Path("./extracted_elements") / doc_name
        output_dir.mkdir(parents=True, exist_ok=True)

        tables_data = []

        for element, _level in document.iterate_items():
            if isinstance(element, TableItem):
                # Save table as image
                try:
                    table_img = element.get_image(document)
                    if table_img:
                        table_img.save(output_dir / f"table_{element.label}.png")
                except Exception as e:
                    _log.debug(f"Could not save table image: {e}")

                # Export table data to structured format
                try:
                    table_df = element.export_to_dataframe()
                    if table_df is not None and not table_df.empty:
                        table_df.to_csv(output_dir / f"table_{element.label}.csv")

                        tables_data.append(
                            {
                                "label": element.label,
                                "prov": element.prov[0].to_dict()
                                if element.prov
                                else {},
                                "preview": table_df.head(5).to_dict(orient="records"),
                            }
                        )
                except Exception as e:
                    _log.debug(f"Could not export table data: {e}")

            elif isinstance(element, PictureItem):
                # Save image
                try:
                    img = element.get_image(document)
                    if img:
                        img.save(output_dir / f"image_{element.label}.png")
                except Exception as e:
                    _log.debug(f"Could not save image: {e}")

        if tables_data:
            with open(output_dir / "tables_metadata.json", "w") as f:
                json.dump(tables_data, f, indent=2)

            _log.info(f"Extracted {len(tables_data)} tables to {output_dir}")

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents using similarity search"""

        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)

            # Add scores to metadata and return
            docs_with_scores = []
            for doc, score in results:
                doc.metadata["relevance_score"] = float(score)
                docs_with_scores.append(doc)

            return docs_with_scores

        except Exception as e:
            _log.error(f"Search failed: {e}")
            return []

    def search_with_filter(
        self, query: str, k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """Search with metadata filters"""

        try:
            results = self.vector_store.similarity_search(
                query, k=k, filter=filter_dict
            )
            return results
        except Exception as e:
            _log.error(f"Filtered search failed: {e}")
            return []

    def get_retriever(self, k: int = 5):
        """Get a retriever for use in chains"""
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def get_rag_context(self, query: str, k: int = 5) -> str:
        """Get formatted context for RAG"""

        docs = self.search(query, k=k)

        if not docs:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("document", "Unknown")
            page_info = ""
            if (
                doc.metadata.get("page_numbers")
                and doc.metadata["page_numbers"] != "[]"
            ):
                page_info = f" (page {doc.metadata['page_numbers']})"

            context_parts.append(
                f"[{i + 1}] From {source}{page_info}:\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)

    def answer_query(self, query: str, temperature: float = 0.1, k: int = 5) -> str:
        """Generate answer using RAG with Ollama LLM"""

        # Get relevant context
        context = self.get_rag_context(query, k=k)

        if context == "No relevant documents found.":
            return "I couldn't find any relevant information in the documents to answer your question."

        # Create prompt for Ollama
        prompt = f"""Answer the question based on the following context. If the context doesn't contain relevant information, say so.
                    Context:
                    {context}

                    Question: {query}

                    Answer:
                """

        try:
            # Generate response using Ollama
            response = self.llm.invoke(prompt)

            # Add source information
            docs = self.search(query, k=k)
            sources = list(
                set([doc.metadata.get("document", "Unknown") for doc in docs])
            )

            if sources:
                response = f"{response}\n\nSources: {', '.join(sources)}"

            return response

        except Exception as e:
            _log.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {e}"

    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of processed documents"""
        return self.document_metadata

    def count_documents(self) -> int:
        """Get count of documents in vector store"""
        try:
            return self.vector_store._collection.count()
        except:
            return 0


def main():
    # Setup
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "data"

    # Only process files that actually exist
    potential_files = [
        "pdf_test.pdf",
        # "doc_test.docx",
        # "excel_test.xlsx",
        # "pptx_test.pptx",
        # "csv_test.csv",
    ]

    input_doc_paths = []
    for filename in potential_files:
        file_path = data_folder / filename
        if file_path.exists():
            input_doc_paths.append(file_path)
        else:
            _log.warning(f"File not found: {file_path}")

    # Initialize RAG pipeline
    rag = RAGPipeline(
        persist_directory="./rag_db",
        collection_name="my_documents",
        ollama_embedding_model="nomic-embed-text",
        ollama_llm_model="llama3.2",
        ollama_base_url="http://localhost:11434",
    )

    # Process all documents
    if input_doc_paths:
        doc_summary = rag.process_documents(input_doc_paths)

        # Print summary
        print("\n📊 Document Processing Summary:")
        for doc_name, info in doc_summary.items():
            print(
                f"  • {doc_name}: {info['pages']} pages, {info['tables']} tables, "
                f"{info['images']} images, {info['chunks']} chunks"
            )

        print(f"\nTotal chunks in vector store: {rag.count_documents()}")
    else:
        print("⚠️ No documents found to process.")
        return rag

    # Test search
    print("\n🔍 Testing search functionality...")
    test_query = "What information is in these documents?"
    results = rag.search(test_query, k=3)

    if results:
        print(f"\nTop 3 results for '{test_query}':")
        for i, doc in enumerate(results):
            print(
                f"\n{i + 1}. [{doc.metadata.get('document', 'Unknown')}] "
                f"(score: {doc.metadata.get('relevance_score', 'N/A'):.3f}):"
            )
            print(f"   {doc.page_content[:150]}...")
    else:
        print("No search results found.")

    # Test RAG with Ollama LLM
    # print("\n🤖 Testing RAG with Ollama LLM...")
    # answer = rag.answer_query("Tell me about the tables and images in these documents")
    # print(f"\nAnswer: {answer}")

    # Test filtered search (if you want to search within specific documents)
    print("\n🎯 Testing filtered search...")
    if doc_summary:
        first_doc = list(doc_summary.keys())[0]
        filtered_results = rag.search_with_filter(
            "content", k=2, filter_dict={"document": first_doc}
        )
        print(f"Found {len(filtered_results)} results in '{first_doc}'")

    # Save metadata
    with open("./document_store_metadata.json", "w") as f:
        json.dump(doc_summary, f, indent=2)
    print("\n💾 Document metadata saved to document_store_metadata.json")

    return rag


if __name__ == "__main__":
    rag_pipeline = main()
