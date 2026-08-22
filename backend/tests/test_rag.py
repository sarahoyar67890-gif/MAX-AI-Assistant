"""
Tests for the RAG pipeline.

Embedding and reranking models are mocked here on purpose — tests should
run fast, offline, and in CI without downloading multi-hundred-MB models
from Hugging Face. The chunking, hybrid-merge, and confidence-scoring
LOGIC is real and fully exercised; only the neural network calls are
stubbed with deterministic fakes.
"""

import sys
import os
import shutil
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.rag.ingestion import chunk_text, ingest_file, Chunk


# ---------------------------------------------------------------------------
# Ingestion / chunking tests — no mocking needed, pure logic
# ---------------------------------------------------------------------------
class TestChunking:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("This is a short sentence.", source="test.txt", chunk_size=800, overlap=100)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short sentence."

    def test_empty_text_no_chunks(self):
        chunks = chunk_text("   ", source="test.txt", chunk_size=800, overlap=100)
        assert chunks == []

    def test_long_text_multiple_chunks(self):
        text = "Sentence one. " * 200  # ~2800 chars
        chunks = chunk_text(text, source="test.txt", chunk_size=500, overlap=100)
        assert len(chunks) > 1
        # Every chunk should carry correct source and sequential index
        for i, c in enumerate(chunks):
            assert c.source == "test.txt"
            assert c.chunk_index == i

    def test_chunk_id_format(self):
        chunks = chunk_text("Some text here.", source="policy.txt", chunk_size=800, overlap=100)
        assert chunks[0].chunk_id == "policy.txt::chunk_0"

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("text", source="t.txt", chunk_size=100, overlap=100)

    def test_ingest_file_reads_and_chunks(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Remote work is allowed up to three days per week.")
        chunks = ingest_file(str(f), chunk_size=800, overlap=100)
        assert len(chunks) == 1
        assert "Remote work" in chunks[0].text


# ---------------------------------------------------------------------------
# Retrieval tests — embedding + reranker mocked, hybrid-merge logic is real
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_retriever(tmp_path, monkeypatch):
    """
    Builds a HybridRetriever with a fake, deterministic embedding function
    (so ChromaDB never touches the network) and a fake reranker whose
    scores we control directly in each test.
    """
    from backend.config import settings as settings_module
    monkeypatch.setattr(settings_module.settings, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

    from chromadb.api.types import EmbeddingFunction

    class _FakeEmbeddingFunction(EmbeddingFunction):
        """Deterministic, network-free stand-in satisfying Chroma's
        EmbeddingFunction protocol (a real subclass, not a MagicMock,
        since Chroma inspects __call__'s signature and calls .name())."""
        def __init__(self):
            pass

        def __call__(self, input):
            return [[float(hash(t) % 997) / 997.0] * 8 for t in input]

        def name(self) -> str:
            return "default"

        @staticmethod
        def build_from_config(config):
            return _FakeEmbeddingFunction()

        def get_config(self):
            return {}

    with patch("backend.rag.retrieval.embedding_functions.SentenceTransformerEmbeddingFunction") as mock_ef_cls:
        mock_ef_cls.return_value = _FakeEmbeddingFunction()

        from backend.rag.retrieval import HybridRetriever
        retriever = HybridRetriever()
        yield retriever


class TestHybridRetrieval:
    def test_add_and_count(self, mock_retriever):
        chunks = chunk_text(
            "The home office stipend is $200 for new employees. "
            "Remote work requires manager approval for up to three days.",
            source="policy.txt", chunk_size=100, overlap=20
        )
        mock_retriever.add_chunks(chunks)
        assert mock_retriever.document_count() == len(chunks)

    def test_list_sources(self, mock_retriever):
        chunks = chunk_text("Some policy text about leave.", source="leave.txt", chunk_size=800, overlap=100)
        mock_retriever.add_chunks(chunks)
        assert "leave.txt" in mock_retriever.list_sources()

    def test_keyword_search_finds_exact_match(self, mock_retriever):
        # BM25's IDF statistically collapses toward zero with only 2 documents —
        # this is a real property of BM25, not a bug — so this test uses a
        # slightly more realistic corpus size where discrimination is meaningful.
        chunks = [
            Chunk(text="The stipend amount is $200 exactly.", source="a.txt", chunk_index=0, start_char=0, end_char=10),
            Chunk(text="Completely unrelated content about weather patterns.", source="b.txt", chunk_index=0, start_char=0, end_char=10),
            Chunk(text="Remote work requires manager approval for three days.", source="c.txt", chunk_index=0, start_char=0, end_char=10),
            Chunk(text="Leave accrues at 1.5 days per month for employees.", source="d.txt", chunk_index=0, start_char=0, end_char=10),
        ]
        mock_retriever.add_chunks(chunks)
        results = mock_retriever._keyword_search("stipend $200", top_k=5)
        assert len(results) >= 1
        assert "stipend" in results[0].text.lower()

    def test_hybrid_search_merges_and_scores(self, mock_retriever):
        chunks = [
            Chunk(text="The stipend amount is $200 exactly.", source="a.txt", chunk_index=0, start_char=0, end_char=10),
            Chunk(text="Completely unrelated content about weather patterns.", source="b.txt", chunk_index=0, start_char=0, end_char=10),
        ]
        mock_retriever.add_chunks(chunks)
        results = mock_retriever.hybrid_search("stipend amount", top_k=5)
        assert len(results) >= 1
        # Every result should have a valid combined score
        for r in results:
            assert r.score >= 0

    def test_rerank_reorders_by_mocked_score(self, mock_retriever):
        from backend.rag.retrieval import RetrievalResult
        results = [
            RetrievalResult("c1", "irrelevant text", "a.txt", 0.5),
            RetrievalResult("c2", "highly relevant text", "b.txt", 0.4),
        ]
        with patch.object(mock_retriever, "_get_reranker") as mock_get_reranker:
            fake_reranker = MagicMock()
            fake_reranker.predict.return_value = [0.1, 0.9]  # c2 should win after rerank
            mock_get_reranker.return_value = fake_reranker

            reranked = mock_retriever.rerank("some query", results, top_n=2)
            assert reranked[0].chunk_id == "c2"
            assert reranked[0].score == 0.9

    def test_empty_index_returns_no_results(self, mock_retriever):
        results = mock_retriever.retrieve("anything")
        assert results == []


class TestRetrievalConfidence:
    def test_no_results_zero_confidence(self):
        from backend.rag.retrieval import retrieval_confidence
        assert retrieval_confidence([]) == 0.0

    def test_high_score_high_confidence(self):
        from backend.rag.retrieval import RetrievalResult, retrieval_confidence
        results = [RetrievalResult("c1", "text", "a.txt", 5.0)]
        conf = retrieval_confidence(results)
        assert 0.9 <= conf <= 1.0

    def test_low_score_low_confidence(self):
        from backend.rag.retrieval import RetrievalResult, retrieval_confidence
        results = [RetrievalResult("c1", "text", "a.txt", -4.5)]
        conf = retrieval_confidence(results)
        assert conf <= 0.1
