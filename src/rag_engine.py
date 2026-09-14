import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class RAGEngine:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        self.chunks = []
        self.embeddings = None

    # ========================================================
    # CREATE CHUNKS
    # ========================================================

    def create_chunks(
        self,
        text,
        section_name="Full Text",
        chunk_size=700,
        overlap=100,
    ):

        text = text.strip()

        if not text:
            return []

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length,
            )

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "section": section_name,
                    }
                )

            if end >= text_length:
                break

            start = end - overlap

        return chunks

    # ========================================================
    # INDEX DOCUMENT
    # ========================================================

    def index_document(
        self,
        text,
        sections=None,
    ):

        self.chunks = []

        # ----------------------------------------------------
        # SECTION-AWARE INDEXING
        # ----------------------------------------------------

        if sections:

            for section_name, section_text in sections.items():

                section_chunks = self.create_chunks(
                    section_text,
                    section_name=section_name,
                )

                self.chunks.extend(
                    section_chunks
                )

        # ----------------------------------------------------
        # FALLBACK: FULL DOCUMENT
        # ----------------------------------------------------

        else:

            self.chunks = self.create_chunks(
                text,
                section_name="Full Text",
            )

        if not self.chunks:

            self.embeddings = None

            return 0

        # ----------------------------------------------------
        # CREATE EMBEDDINGS
        # ----------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        print(
            f"Creating embeddings for "
            f"{len(texts)} chunks..."
        )

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return len(self.chunks)

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=5,
    ):

        if (
            self.embeddings is None
            or not self.chunks
        ):

            return []

        # ----------------------------------------------------
        # QUERY EMBEDDING
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # ----------------------------------------------------
        # COSINE SIMILARITY
        # ----------------------------------------------------

        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        # ----------------------------------------------------
        # TOP RESULTS
        # ----------------------------------------------------

        top_k = min(
            top_k,
            len(self.chunks),
        )

        best_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in best_indices:

            chunk = self.chunks[index]

            results.append(
                {
                    "chunk": chunk["text"],
                    "section": chunk["section"],
                    "score": float(
                        scores[index]
                    ),
                }
            )

        return results