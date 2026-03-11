from sentence_transformers import SentenceTransformer
from docling.chunking import HybridChunker

class DocumentChunker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer_model = SentenceTransformer(model_name)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer_model.tokenizer, merge_peers=True
        )

    def chunk(self, document):
        return list(self.chunker.chunk(document))
