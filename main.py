import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.core.pipeline import RAGPipeline

load_dotenv()
_log = logging.getLogger(__name__)


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

    # Initialize RAG pipeline using the new modular structure (config based)
    rag = RAGPipeline()

    # Process all documents
    doc_summary = {}
    if input_doc_paths:
        doc_summary = rag.process_documents(input_doc_paths)

        # Print summary
        print("\n📊 Document Processing Summary:")
        for doc_name, info in doc_summary.items():
            print(
                f"  • {doc_name}: {info['pages']} pages, {info['tables']} tables, "
                f"{info['images']} images, {info['chunks']} chunks"
            )

        print(f"\nTotal items in vector store: {rag.count_documents()}")
    else:
        print("⚠️ No documents found to process.")
        return rag

    # Test different search types
    print("\n🔍 Testing multimodal search...")

    # General search
    test_query = "What information is in these documents?"
    results = rag.search(test_query, k=3)
    print(f"\n📝 General search results for '{test_query}': {len(results)} results")

    # Table-specific search
    table_results = rag.query_tables("Show me data tables", k=2)
    print(f"\n📊 Table search results: {len(table_results)} tables found")
    for i, doc in enumerate(table_results):
        print(f"  Table {i + 1}: {doc.page_content[:100]}...")

    # Image-specific search
    image_results = rag.query_images("Describe the images", k=2)
    print(f"\n🖼️ Image search results: {len(image_results)} images found")
    for i, doc in enumerate(image_results):
        print(f"  Image {i + 1}: {doc.page_content[:100]}...")

    # Test RAG with Ollama LLM
    print("\n🤖 Testing RAG with Ollama LLM...")
    answer = rag.answer_query("Tell me about the tables and images in these documents")
    print(f"\nAnswer: {answer}")

    # Test filtered search
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
