from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class RAGEngine:
    """Create section-aware chunks and perform similarity-based retrieval."""

    def __init__(self, model_name: str = MODEL_NAME):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def create_chunks(
        self,
        text: str,
        section_name: str = "Full Text",
        chunk_size: int = 700,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        cleaned = (text or "").strip()
        if not cleaned:
            return []

        paragraphs = re.split(r"\n{2,}", cleaned)
        chunks: List[Dict[str, Any]] = []
        buffer: List[str] = []
        buffer_length = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if buffer_length and buffer_length + len(paragraph) > chunk_size:
                chunk_text = "\n\n".join(buffer).strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "section": section_name})
                buffer = [paragraph]
                buffer_length = len(paragraph)
            else:
                buffer.append(paragraph)
                buffer_length += len(paragraph)

        if buffer:
            chunk_text = "\n\n".join(buffer).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "section": section_name})

        merged_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            chunk_text = chunk["text"]
            if len(chunk_text) <= chunk_size:
                merged_chunks.append(chunk)
                continue
            slice_start = 0
            while slice_start < len(chunk_text):
                slice_end = min(slice_start + chunk_size, len(chunk_text))
                segment = chunk_text[slice_start:slice_end].strip()
                if segment:
                    merged_chunks.append({"text": segment, "section": section_name})
                if slice_end >= len(chunk_text):
                    break
                slice_start = max(slice_start + chunk_size - overlap, slice_start + 1)

        return merged_chunks

    def index_document(self, text: str, sections: Optional[Dict[str, str]] = None) -> int:
        """Index the document and create numerical embeddings for retrieval."""
        self.chunks = []

        if sections:
            for section_name, section_text in sections.items():
                self.chunks.extend(self.create_chunks(section_text, section_name=section_name))
        else:
            self.chunks = self.create_chunks(text, section_name="Full Text")

        if not self.chunks:
            self.embeddings = None
            return 0

        texts = [chunk["text"] for chunk in self.chunks]
        print(f"Creating embeddings for {len(texts)} chunks...")
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a natural-language question."""
        if self.embeddings is None or not self.chunks or not str(query or "").strip():
            return []

        query_embedding = self.model.encode(
            str(query).strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = np.dot(self.embeddings, query_embedding)
        top_k = min(max(1, int(top_k)), len(self.chunks))
        best_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for index in best_indices:
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "chunk": chunk["text"],
                    "section": chunk.get("section", "Full Text"),
                    "score": float(scores[int(index)]),
                }
            )
        return results