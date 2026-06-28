from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from loguru import logger
from config import settings


class ChunkStrategy(str, Enum):
    FIXED = "fixed"           # Fixed token-size chunks
    RECURSIVE = "recursive"   # Split by paragraphs → sentences → words
    SEMANTIC = "semantic"     # Group sentences by semantic similarity (Phase 2+)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    strategy: str
    metadata: dict = field(default_factory=dict)


class DocumentChunker:
    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> list[Chunk]:
        metadata = metadata or {}
        if self.strategy == ChunkStrategy.FIXED:
            raw_chunks = self._fixed_chunk(text)
        elif self.strategy == ChunkStrategy.RECURSIVE:
            raw_chunks = self._recursive_chunk(text)
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not yet implemented")

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=i,
                strategy=self.strategy.value,
                metadata={**metadata, "chunk_index": i},
            ))

        logger.debug(f"Chunked doc {doc_id} into {len(chunks)} chunks via {self.strategy}")
        return chunks

    # ── Strategies ─────────────────────────────────────────────────────────────

    def _fixed_chunk(self, text: str) -> list[str]:
        """Split text into fixed character-size windows with overlap."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _recursive_chunk(self, text: str) -> list[str]:
        """
        Try splitting by double newline (paragraphs) first.
        If a paragraph is too large, split by sentence.
        If a sentence is too large, fall back to fixed splitting.
        """
        chunks = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            para_words = para.split()
            if len(para_words) <= self.chunk_size:
                # Paragraph fits — use as-is
                chunks.append(para)
            else:
                # Split paragraph into sentences
                sentences = self._split_sentences(para)
                current_chunk_words = []
                for sentence in sentences:
                    sentence_words = sentence.split()
                    if len(current_chunk_words) + len(sentence_words) <= self.chunk_size:
                        current_chunk_words.extend(sentence_words)
                    else:
                        if current_chunk_words:
                            chunks.append(" ".join(current_chunk_words))
                        # If single sentence is too large, fixed-split it
                        if len(sentence_words) > self.chunk_size:
                            chunks.extend(self._fixed_chunk(sentence))
                            current_chunk_words = []
                        else:
                            current_chunk_words = sentence_words
                if current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))

        # Add overlap between consecutive chunks
        return self._add_overlap(chunks)

    def _split_sentences(self, text: str) -> list[str]:
        """Basic sentence splitter — good enough for Phase 1."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Append the last N words of the previous chunk to each chunk."""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()[-self.chunk_overlap:]
            overlap_text = " ".join(prev_words)
            result.append(overlap_text + " " + chunks[i])
        return result