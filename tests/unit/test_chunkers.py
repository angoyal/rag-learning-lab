"""Tests for chunking strategies."""

import pytest
from src.ingest.chunkers import (
    Chunk,
    chunk_text,
    fixed_chunker,
    recursive_chunker,
    semantic_chunker,
    sentence_chunker,
)
from src.ingest.embedders import Embedder

# --- Fixed chunker tests ---


@pytest.mark.unit
def test_fixed_basic():
    text = "a" * 1200
    chunks = fixed_chunker(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text) <= 500 for c in chunks)


@pytest.mark.unit
def test_fixed_no_overlap():
    text = "a" * 1000
    chunks = fixed_chunker(text, chunk_size=500, chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0].text == "a" * 500
    assert chunks[1].text == "a" * 500


@pytest.mark.unit
def test_fixed_text_shorter_than_chunk():
    text = "short text"
    chunks = fixed_chunker(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


@pytest.mark.unit
def test_fixed_empty_text():
    chunks = fixed_chunker("", chunk_size=500, chunk_overlap=50)
    assert chunks == []


@pytest.mark.unit
def test_fixed_overlap_boundaries():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = fixed_chunker(text, chunk_size=500, chunk_overlap=50)
    for i in range(len(chunks) - 1):
        assert chunks[i].text[-50:] == chunks[i + 1].text[:50]


@pytest.mark.unit
def test_fixed_exact_multiple():
    text = "a" * 1000
    chunks = fixed_chunker(text, chunk_size=500, chunk_overlap=0)
    assert len(chunks) == 2
    total = "".join(c.text for c in chunks)
    assert total == text


@pytest.mark.unit
def test_fixed_chunk_metadata():
    text = "a" * 1200
    chunks = fixed_chunker(text, chunk_size=500, chunk_overlap=0)
    assert chunks[0].index == 0
    assert chunks[0].start == 0
    assert chunks[0].end == 500
    assert chunks[1].index == 1
    assert chunks[1].start == 500
    assert chunks[1].end == 1000
    assert chunks[2].index == 2
    assert chunks[2].start == 1000
    assert chunks[2].end == 1200


@pytest.mark.unit
def test_fixed_overlap_metadata():
    text = "a" * 600
    chunks = fixed_chunker(text, chunk_size=500, chunk_overlap=50)
    assert chunks[0].start == 0
    assert chunks[0].end == 500
    assert chunks[1].start == 450
    assert chunks[1].end == 600


# --- Recursive chunker tests ---


@pytest.mark.unit
def test_recursive_splits_on_paragraphs():
    paragraphs = ["Word " * 40 for _ in range(3)]  # ~200 chars each
    text = "\n\n".join(paragraphs)
    chunks = recursive_chunker(text, chunk_size=250, chunk_overlap=0)
    # Each paragraph should be its own chunk (they fit under 250)
    assert len(chunks) == 3


@pytest.mark.unit
def test_recursive_falls_back_to_sentences():
    # One long paragraph with multiple sentences
    sentences = ["This is sentence number " + str(i) + "." for i in range(20)]
    text = " ".join(sentences)
    chunks = recursive_chunker(text, chunk_size=200, chunk_overlap=0)
    assert len(chunks) > 1
    # Each chunk should respect the size limit
    assert all(len(c.text) <= 200 for c in chunks)


@pytest.mark.unit
def test_recursive_respects_chunk_size():
    text = ("This is a sentence. " * 50) + "\n\n" + ("Another sentence here. " * 50)
    chunks = recursive_chunker(text, chunk_size=100, chunk_overlap=0)
    assert all(len(c.text) <= 100 for c in chunks)


@pytest.mark.unit
def test_recursive_preserves_all_text():
    text = "Hello world. This is a test. Another paragraph.\n\nSecond paragraph here. More text."
    chunks = recursive_chunker(text, chunk_size=50, chunk_overlap=0)
    reconstructed = "".join(c.text for c in chunks)
    assert reconstructed == text


@pytest.mark.unit
def test_recursive_with_overlap():
    text = "Hello world. This is a test.\n\nSecond paragraph here. More text follows."
    chunks = recursive_chunker(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text) <= 40 for c in chunks)


@pytest.mark.unit
def test_recursive_empty_text():
    chunks = recursive_chunker("", chunk_size=100, chunk_overlap=0)
    assert chunks == []


@pytest.mark.unit
def test_recursive_text_shorter_than_chunk():
    text = "Short."
    chunks = recursive_chunker(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == "Short."


@pytest.mark.unit
def test_recursive_falls_back_to_characters():
    # A single long "word" with no spaces, sentences, or paragraphs
    text = "x" * 200
    chunks = recursive_chunker(text, chunk_size=50, chunk_overlap=0)
    assert len(chunks) == 4
    assert all(len(c.text) <= 50 for c in chunks)
    assert "".join(c.text for c in chunks) == text


# --- Sentence chunker tests ---


@pytest.mark.unit
def test_sentence_never_splits_mid_sentence():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = sentence_chunker(text, chunk_size=40, chunk_overlap=0)
    for chunk in chunks:
        # Each chunk should end with a sentence-ending pattern or be the last chunk
        assert chunk.text.rstrip().endswith(".") or chunk == chunks[-1]


@pytest.mark.unit
def test_sentence_packs_sentences():
    text = "Hi. OK. Go. Do. Ah."
    chunks = sentence_chunker(text, chunk_size=100, chunk_overlap=0)
    # All sentences fit in one chunk
    assert len(chunks) == 1
    assert chunks[0].text == text


@pytest.mark.unit
def test_sentence_single_long_sentence():
    text = "A" * 200
    chunks = sentence_chunker(text, chunk_size=50, chunk_overlap=0)
    # Can't split at a sentence boundary, so return as-is
    assert len(chunks) == 1
    assert chunks[0].text == text


@pytest.mark.unit
def test_sentence_with_various_terminators():
    text = "Hello! How are you? I am fine. Great."
    chunks = sentence_chunker(text, chunk_size=25, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)


@pytest.mark.unit
def test_sentence_empty_text():
    chunks = sentence_chunker("", chunk_size=100, chunk_overlap=0)
    assert chunks == []


@pytest.mark.unit
def test_sentence_with_overlap():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = sentence_chunker(text, chunk_size=35, chunk_overlap=15)
    assert len(chunks) > 1


@pytest.mark.unit
def test_sentence_with_newlines():
    text = "First line.\nSecond line.\nThird line."
    chunks = sentence_chunker(text, chunk_size=15, chunk_overlap=0)
    assert len(chunks) > 1


# --- Dispatcher tests ---


@pytest.mark.unit
def test_chunk_text_routes_to_fixed():
    text = "a" * 100
    result_dispatch = chunk_text(text, strategy="fixed", chunk_size=50, chunk_overlap=0)
    result_direct = fixed_chunker(text, chunk_size=50, chunk_overlap=0)
    assert len(result_dispatch) == len(result_direct)
    for d, r in zip(result_dispatch, result_direct):
        assert d.text == r.text


@pytest.mark.unit
def test_chunk_text_routes_to_recursive():
    text = "Hello world.\n\nSecond paragraph."
    result_dispatch = chunk_text(text, strategy="recursive", chunk_size=50, chunk_overlap=0)
    result_direct = recursive_chunker(text, chunk_size=50, chunk_overlap=0)
    assert len(result_dispatch) == len(result_direct)
    for d, r in zip(result_dispatch, result_direct):
        assert d.text == r.text


@pytest.mark.unit
def test_chunk_text_routes_to_sentence():
    text = "Hello world. Second sentence."
    result_dispatch = chunk_text(text, strategy="sentence", chunk_size=50, chunk_overlap=0)
    result_direct = sentence_chunker(text, chunk_size=50, chunk_overlap=0)
    assert len(result_dispatch) == len(result_direct)
    for d, r in zip(result_dispatch, result_direct):
        assert d.text == r.text


@pytest.mark.unit
def test_chunk_text_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_text("text", strategy="unknown", chunk_size=100, chunk_overlap=0)


# --- Semantic chunker tests ---


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Load embedder once for all semantic chunker tests."""
    return Embedder("all-MiniLM-L6-v2")


@pytest.mark.unit
def test_semantic_splits_on_topic_change(embedder: Embedder):
    text = (
        "The cat sat on the mat. The cat purred softly. The kitten played with yarn. "
        "Quantum mechanics describes the behavior of particles at the subatomic level. "
        "The Heisenberg uncertainty principle limits precision. "
        "Wave functions collapse upon measurement."
    )
    chunks = semantic_chunker(text, embedder, threshold=0.5)
    assert len(chunks) >= 2
    # Cat-related sentences and physics sentences should be in different chunks
    cat_chunk = next(c for c in chunks if "cat" in c.text)
    physics_chunk = next(c for c in chunks if "Quantum" in c.text)
    assert cat_chunk.index != physics_chunk.index


@pytest.mark.unit
def test_semantic_keeps_similar_together(embedder: Embedder):
    text = (
        "Dogs are loyal companions. Dogs love to play fetch. "
        "Dogs need daily walks for exercise."
    )
    chunks = semantic_chunker(text, embedder, threshold=0.3)
    # All sentences are about dogs — should stay in one chunk
    assert len(chunks) == 1


@pytest.mark.unit
def test_semantic_respects_sentence_boundaries(embedder: Embedder):
    text = (
        "The weather is sunny today. Birds are singing in the trees. "
        "Machine learning models require large datasets. "
        "Neural networks have multiple layers."
    )
    chunks = semantic_chunker(text, embedder, threshold=0.5)
    for chunk in chunks:
        # No chunk should start or end mid-sentence (partial word)
        stripped = chunk.text.strip()
        assert stripped[0].isupper() or stripped[0] == '"'
        assert stripped[-1] in ".!?"


@pytest.mark.unit
def test_semantic_single_sentence(embedder: Embedder):
    text = "Just one sentence here."
    chunks = semantic_chunker(text, embedder, threshold=0.5)
    assert len(chunks) == 1
    assert chunks[0].text == text


@pytest.mark.unit
def test_semantic_empty_text(embedder: Embedder):
    chunks = semantic_chunker("", embedder, threshold=0.5)
    assert chunks == []


@pytest.mark.unit
def test_semantic_whitespace_only(embedder: Embedder):
    chunks = semantic_chunker("   \n\n  ", embedder, threshold=0.5)
    assert chunks == []


@pytest.mark.unit
def test_semantic_chunk_offsets(embedder: Embedder):
    text = (
        "The cat sat on the mat. The cat purred softly. "
        "Quantum mechanics is fascinating. The uncertainty principle is key."
    )
    chunks = semantic_chunker(text, embedder, threshold=0.5)
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text


@pytest.mark.unit
def test_chunk_text_routes_to_semantic(embedder: Embedder):
    text = "Hello world. This is a test."
    result = chunk_text(text, strategy="semantic", chunk_size=0, chunk_overlap=0, embedder=embedder)
    assert len(result) >= 1
    assert all(isinstance(c, Chunk) for c in result)


@pytest.mark.unit
def test_chunk_text_semantic_requires_embedder():
    with pytest.raises(ValueError, match="embedder"):
        chunk_text("Hello.", strategy="semantic", chunk_size=0, chunk_overlap=0)
