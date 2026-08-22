"""
Advanced retrieval pipeline:

    query
      -> query rewriting (optional, via LLM)
      -> hybrid retrieval (BM25 keyword search + ChromaDB dense vector search, merged)
      -> cross-encoder reranking (reorders by actual relevance, not just similarity)
      -> confidence scoring (decides if evidence is strong enough to answer)

This is deliberately NOT "chunk -> embed -> top-k -> stuff into prompt."
Hybrid retrieval catches exact keyword matches (names, numbers, IDs) that
pure embedding similarity often misses; reranking fixes cases where the
embedding model's similarity ranking doesn't match true relevance.
"""

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import os

from backend.config.settings import settings
from backend.rag.ingestion import Chunk


class RetrievalResult:
    def __init__(self, chunk_id: str, text: str, source: str, score: float):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.score = score

    def to_dict(self):
        return {"chunk_id": self.chunk_id, "text": self.text, "source": self.source, "score": self.score}


class HybridRetriever:
    def __init__(self):
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        # Explicit sentence-transformers embedding function (all-MiniLM-L6-v2) —
        # runs locally via the sentence-transformers lib we already depend on
        # for reranking, rather than Chroma's default ONNX embedder.
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            "max_documents", embedding_function=self._embedding_fn
        )

        # BM25 index is rebuilt in-memory from whatever's in Chroma at load time —
        # fine at portfolio scale; swap for a persistent inverted index at real scale.
        self._bm25 = None
        self._bm25_corpus_ids = []
        self._reranker = None  # lazy-loaded — it's a real model download, don't pay that cost until needed

    # -- Indexing --------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        all_docs = self._collection.get()
        ids = all_docs["ids"]
        texts = all_docs["documents"]
        if not ids:
            self._bm25 = None
            self._bm25_corpus_ids = []
            return
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus_ids = ids
        self._bm25_texts = {i: t for i, t in zip(ids, texts)}
        self._bm25_meta = {i: m for i, m in zip(ids, all_docs["metadatas"])}

    # -- Retrieval ---------------------------------------------------------
    def _vector_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(query_texts=[query], n_results=min(top_k, self._collection.count()))
        out = []
        for i in range(len(results["ids"][0])):
            # Chroma returns distance (lower = closer); convert to a similarity-ish score
            distance = results["distances"][0][i]
            score = 1.0 / (1.0 + distance)
            out.append(RetrievalResult(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                source=results["metadatas"][0][i].get("source", "unknown"),
                score=score,
            ))
        return out

    def _keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self._bm25_corpus_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = max((s for _, s in ranked), default=1.0) or 1.0
        return [
            RetrievalResult(
                chunk_id=cid,
                text=self._bm25_texts[cid],
                source=self._bm25_meta[cid].get("source", "unknown"),
                score=score / max_score,  # normalize to 0-1 so it's comparable with vector scores
            )
            for cid, score in ranked if score > 0
        ]

    def hybrid_search(self, query: str, top_k: int = None) -> list[RetrievalResult]:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        vector_results = self._vector_search(query, top_k)
        keyword_results = self._keyword_search(query, top_k)

        # Merge by chunk_id, combining scores (simple weighted sum — 60% vector, 40% keyword)
        merged: dict[str, RetrievalResult] = {}
        for r in vector_results:
            merged[r.chunk_id] = RetrievalResult(r.chunk_id, r.text, r.source, r.score * 0.6)
        for r in keyword_results:
            if r.chunk_id in merged:
                merged[r.chunk_id].score += r.score * 0.4
            else:
                merged[r.chunk_id] = RetrievalResult(r.chunk_id, r.text, r.source, r.score * 0.4)

        return sorted(merged.values(), key=lambda r: r.score, reverse=True)[:top_k]

    # -- Reranking ---------------------------------------------------------
    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def rerank(self, query: str, results: list[RetrievalResult], top_n: int = None) -> list[RetrievalResult]:
        if not results:
            return results
        top_n = top_n or settings.RERANK_TOP_N
        reranker = self._get_reranker()
        pairs = [[query, r.text] for r in results]
        scores = reranker.predict(pairs)
        for r, s in zip(results, scores):
            r.score = float(s)
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_n]

    # -- Full pipeline -------------------------------------------------
    def retrieve(self, query: str) -> list[RetrievalResult]:
        candidates = self.hybrid_search(query, top_k=settings.RETRIEVAL_TOP_K)
        return self.rerank(query, candidates, top_n=settings.RERANK_TOP_N)

    def document_count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        all_docs = self._collection.get()
        sources = {m.get("source", "unknown") for m in all_docs["metadatas"]}
        return sorted(sources)


def retrieval_confidence(results: list[RetrievalResult]) -> float:
    """
    Cheap, deterministic confidence signal based on top reranker score.
    Used to decide whether to answer from retrieval or fall back to
    'I don't have enough information' instead of hallucinating.
    Cross-encoder scores aren't bounded [0,1], so this is a practical
    squash rather than a calibrated probability.
    """
    if not results:
        return 0.0
    top_score = results[0].score
    return max(0.0, min(1.0, (top_score + 5) / 10))  # empirical squash for ms-marco-MiniLM score range
