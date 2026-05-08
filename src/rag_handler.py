"""
rag_handler.py
Handles document ingestion into ChromaDB and context retrieval
using sentence-transformers embeddings (runs 100% locally, no API key needed).
"""

import os
import tempfile
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Optional


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CHROMA_DB_PATH = "./chroma_store"
COLLECTION_NAME = "test_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"   # Fast, lightweight, local
CHUNK_SIZE = 500                    # Characters per chunk
CHUNK_OVERLAP = 50


# ─────────────────────────────────────────────
# RAGHandler Class
# ─────────────────────────────────────────────

class RAGHandler:
    """
    Handles:
    1. Document chunking and indexing into ChromaDB
    2. Semantic similarity search to retrieve relevant context
    """

    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    # ── Chunking ──────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ── Document Loading ──────────────────────

    def _load_text(self, uploaded_file) -> str:
        """Load text from uploaded .txt or .pdf file."""
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")

        elif uploaded_file.name.endswith(".pdf"):
            try:
                import fitz  # PyMuPDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                doc = fitz.open(tmp_path)
                text = "\n".join(page.get_text() for page in doc)
                os.unlink(tmp_path)
                return text
            except ImportError:
                return "PDF support requires PyMuPDF: pip install pymupdf"
        return ""

    # ── Index Document ─────────────────────────

    def index_document(self, uploaded_file) -> None:
        """
        Chunk and embed document, then store in ChromaDB.

        Args:
            uploaded_file: Streamlit UploadedFile object
        """
        raw_text = self._load_text(uploaded_file)
        chunks = self._chunk_text(raw_text)

        embeddings = self.embedder.encode(chunks).tolist()

        # Clear old docs for this file to avoid duplicates
        try:
            existing = self.collection.get()
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

    # ── Retrieve Context ───────────────────────

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Find top-k most relevant chunks for the query.

        Args:
            query : The user's input (story/feature/bug)
            top_k : Number of chunks to retrieve

        Returns:
            str: Concatenated relevant chunks as context string
        """
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count())
        )

        docs = results.get("documents", [[]])[0]
        return "\n\n---\n\n".join(docs) if docs else ""


# ─────────────────────────────────────────────
# CLI Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with a plain text string
    rag = RAGHandler()

    # Simulate indexing plain text
    sample_text = """
    Login Module Specification:
    - Users must enter registered email and password
    - System validates credentials against the database
    - On success: redirect to dashboard
    - On failure: show 'Invalid credentials' message
    - After 5 failed attempts: lock account for 30 minutes
    - Password reset via email OTP
    """

    # Manually chunk and index without uploaded file
    chunks = rag._chunk_text(sample_text)
    embeddings = rag.embedder.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    rag.collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    context = rag.retrieve_context("login failure and account lockout")
    print("Retrieved Context:\n", context)
