"""Text chunking strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.ingest.embedders import Embedder


@dataclass
class Chunk:
    """A chunk of text with positional metadata.

    Attributes:
        text: The chunk content.
        index: Zero-based chunk index in the sequence.
        start: Start character offset in the original text.
        end: End character offset in the original text.
    """

    text: str
    index: int
    start: int
    end: int


def fixed_chunker(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split text into fixed-size chunks with overlap.

    Uses a sliding window of `chunk_size` characters, advancing by
    `chunk_size - chunk_overlap` each step.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    if not text:
        return []
    chunks = []
    step = chunk_size - chunk_overlap
    pos = 0
    idx = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunks.append(Chunk(text=text[pos:end], index=idx, start=pos, end=end))
        idx += 1
        pos += step
        if end == len(text):
            break
    return chunks


def recursive_chunker(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Split text recursively, trying coarser separators first.

    Tries splitting on paragraph breaks, then sentences, then words,
    then characters. Falls back to finer separators when pieces exceed
    chunk_size.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        separators: Ordered list of separators to try. Defaults to paragraph, sentence, word, char.

    Returns:
        List of Chunk objects.
    """
    if not text:
        return []
    if separators is None:
        separators = ["\n\n", ". ", " ", ""]

    pieces = _recursive_split(text, chunk_size, separators, 0)
    return _merge_pieces(pieces, chunk_size, chunk_overlap)


def _recursive_split(text: str, chunk_size: int, separators: list[str], sep_idx: int) -> list[str]:
    """Recursively split text using progressively finer separators."""
    if len(text) <= chunk_size or sep_idx >= len(separators):
        return [text]

    sep = separators[sep_idx]
    if sep == "":
        # Character-level: just slice
        parts = []
        for i in range(0, len(text), chunk_size):
            parts.append(text[i : i + chunk_size])
        return parts

    splits = text.split(sep)
    # Re-attach the separator to each piece (except the last)
    pieces = []
    for i, s in enumerate(splits):
        piece = s + sep if i < len(splits) - 1 else s
        if piece:
            pieces.append(piece)

    result = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            result.extend(_recursive_split(piece, chunk_size, separators, sep_idx + 1))
    return result


def _merge_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Merge small pieces into chunks up to chunk_size, tracking offsets."""
    chunks = []
    current = ""
    offset = 0
    idx = 0

    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            start = offset
            end = offset + len(current)
            chunks.append(Chunk(text=current, index=idx, start=start, end=end))
            idx += 1
            # Handle overlap
            if chunk_overlap > 0 and len(current) > chunk_overlap:
                overlap_text = current[-chunk_overlap:]
                offset = end - chunk_overlap
                current = overlap_text + piece
            else:
                offset = end
                current = piece
        else:
            current += piece

    if current:
        start = offset
        end = offset + len(current)
        chunks.append(Chunk(text=current, index=idx, start=start, end=end))

    return chunks


def sentence_chunker(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split text at sentence boundaries, packing sentences into chunks.

    Splits on sentence-ending punctuation (. ! ?) followed by a space,
    and on newlines. Sentences are packed into chunks up to chunk_size.
    A sentence longer than chunk_size is kept whole.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    if not text:
        return []

    # Split at sentence boundaries, keeping punctuation and trailing space with each sentence
    sentences = re.split(r"(?<=[.!?] )|(?<=\n)", text)
    sentences = [s for s in sentences if s]

    chunks = []
    current = ""
    offset = 0
    idx = 0

    for sentence in sentences:
        if current and len(current) + len(sentence) > chunk_size:
            chunks.append(Chunk(text=current, index=idx, start=offset, end=offset + len(current)))
            idx += 1
            if chunk_overlap > 0:
                # Overlap: take last overlap chars worth of sentences
                overlap_start = max(0, len(current) - chunk_overlap)
                overlap_text = current[overlap_start:]
                offset = offset + len(current) - len(overlap_text)
                current = overlap_text + sentence
            else:
                offset += len(current)
                current = sentence
        else:
            current += sentence

    if current:
        chunks.append(Chunk(text=current, index=idx, start=offset, end=offset + len(current)))

    return chunks


def semantic_chunker(
    text: str,
    embedder: Embedder,
    threshold: float = 0.5,
    return_embeddings: bool = False,
    batch_size: int = 64,
) -> list[Chunk] | tuple[list[Chunk], np.ndarray]:
    """Split text at semantic boundaries using embedding similarity.

    Splits text into sentences, embeds each, and groups consecutive
    sentences whose cosine similarity exceeds the threshold. A new
    chunk starts wherever similarity between consecutive sentences
    drops below the threshold.

    Args:
        text: The input text to chunk.
        embedder: An Embedder instance for computing sentence embeddings.
        threshold: Cosine similarity threshold (0-1). Lower values produce
            fewer, larger chunks. Higher values produce more, smaller chunks.
        return_embeddings: If True, also return mean-pooled chunk embeddings
            derived from the sentence embeddings (avoids re-embedding).
        batch_size: Number of sentences per GPU batch. Lower values use
            less GPU memory (important for large models like pplx-embed).

    Returns:
        List of Chunk objects, or (chunks, chunk_embeddings) when
        return_embeddings is True.
    """
    if not text:
        if return_embeddings:
            return [], np.empty((0, 0), dtype=np.float32)
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        if return_embeddings:
            return [], np.empty((0, 0), dtype=np.float32)
        return []

    if len(sentences) == 1:
        chunks = [Chunk(text=text, index=0, start=0, end=len(text))]
        if return_embeddings:
            embs = embedder.embed(sentences, batch_size=batch_size)
            return chunks, embs
        return chunks

    sent_embeddings = embedder.embed(sentences, batch_size=batch_size)
    # Cosine similarity between consecutive sentence embeddings
    norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
    normalized = sent_embeddings / norms
    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

    # Find split points where similarity drops below threshold
    split_indices = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    # Build sentence index ranges per chunk
    ranges = []
    prev = 0
    for idx in split_indices:
        ranges.append((prev, idx))
        prev = idx
    ranges.append((prev, len(sentences)))

    # Build chunks from sentence groups
    chunks = []
    offset = 0
    for i, (start_idx, end_idx) in enumerate(ranges):
        group = sentences[start_idx:end_idx]
        start = text.index(group[0], offset)
        last_sentence = group[-1]
        end = text.index(last_sentence, start) + len(last_sentence)
        chunks.append(Chunk(text=text[start:end], index=i, start=start, end=end))
        offset = end

    if return_embeddings:
        # Mean-pool sentence embeddings per chunk
        chunk_embeddings = np.array([
            sent_embeddings[start_idx:end_idx].mean(axis=0)
            for start_idx, end_idx in ranges
        ])
        return chunks, chunk_embeddings

    return chunks


def chunk_text(
    text: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Dispatch to the appropriate chunking strategy.

    Args:
        text: The input text to chunk.
        strategy: One of "fixed", "recursive", "sentence", or "semantic".
        chunk_size: Maximum number of characters per chunk (unused for semantic).
        chunk_overlap: Number of overlapping characters between chunks (unused for semantic).
        embedder: Required for "semantic" strategy. An Embedder instance.

    Returns:
        List of Chunk objects.

    Raises:
        ValueError: If strategy is not recognized, or if "semantic" is used without an embedder.
    """
    if strategy == "fixed":
        return fixed_chunker(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        return recursive_chunker(text, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        return sentence_chunker(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic strategy requires an embedder")
        return semantic_chunker(text, embedder)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
