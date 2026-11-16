"""Utilities for embedding document chunks and searching them with FAISS."""

from typing import Any, TypedDict

from .document_processor import TextChunk


class SearchResult(TypedDict):
    """A matching chunk and its source page with a similarity score."""

    text: str
    page_number: int
    similarity_score: float


def load_embedding_model() -> Any:
    """Load the sentence-transformers model used to embed text."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Loading embeddings requires the sentence-transformers package."
        ) from exc

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_vector_store(chunks: list[TextChunk], model: Any) -> Any:
    """Embed document chunks and store normalized vectors in a FAISS index.

    The returned index keeps the same item order as ``chunks`` so search
    results can be matched back to their page and text.
    """
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Creating the vector store requires NumPy and faiss-cpu."
        ) from exc

    if not chunks:
        # An empty index still needs the embedding dimension for later searches.
        dimension = model.get_sentence_embedding_dimension()
        if not dimension:
            raise ValueError("The embedding model has no known vector dimension.")
        return faiss.IndexFlatIP(dimension)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = np.ascontiguousarray(
        model.encode(texts, convert_to_numpy=True), dtype=np.float32
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("The embedding model returned an invalid embedding array.")
    if embeddings.shape[1] == 0:
        raise ValueError("The embedding model returned vectors with no dimensions.")

    # Inner-product search gives cosine similarity when both sides are normalized.
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_vector_store(
    query: str,
    model: Any,
    index: Any,
    chunks: list[TextChunk],
    top_k: int = 5,
) -> list[SearchResult]:
    """Return the most similar chunks, including their original page numbers."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if index.ntotal != len(chunks):
        raise ValueError("The FAISS index and chunks must contain the same items.")
    if index.ntotal == 0 or not query.strip():
        return []

    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Searching the vector store requires NumPy and faiss-cpu."
        ) from exc

    query_embedding = np.ascontiguousarray(
        model.encode([query], convert_to_numpy=True), dtype=np.float32
    )
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    if query_embedding.shape != (1, index.d):
        raise ValueError("The query embedding does not match the FAISS index.")

    faiss.normalize_L2(query_embedding)
    scores, positions = index.search(query_embedding, min(top_k, index.ntotal))

    results: list[SearchResult] = []
    for score, position in zip(scores[0], positions[0]):
        chunk = chunks[int(position)]
        results.append(
            {
                "text": chunk["text"],
                "page_number": chunk["page_number"],
                "similarity_score": float(score),
            }
        )
    return results
