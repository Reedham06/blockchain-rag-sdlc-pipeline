"""RAG retriever: paragraph-aware chunking + TF-IDF cosine-similarity search.

Dependency-light on purpose (no embedding API, no model download), so it runs
offline. To upgrade to neural embeddings later, keep the same three methods
(`index`, `retrieve_scored`, `best_support`) in a new class and swap it in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str = ""


def _windows(words: List[str], max_words: int, overlap: int):
    step = max(max_words - overlap, 1)
    for start in range(0, len(words), step):
        yield words[start:start + max_words]
        if start + max_words >= len(words):
            break


def chunk_text(text: str, source: str, max_words: int = 120, overlap: int = 20) -> List[Chunk]:
    """Split on blank lines, merge short paragraphs up to `max_words`, and only
    fall back to overlapping word windows for a single over-long paragraph.
    Chunks therefore end on paragraph boundaries instead of mid-sentence."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            pieces.append(" ".join(buf))
            buf = []

    for para in paragraphs:
        words = para.split()
        if len(words) > max_words:
            flush()
            pieces.extend(" ".join(w) for w in _windows(words, max_words, overlap))
        elif len(buf) + len(words) > max_words:
            flush()
            buf = list(words)
        else:
            buf.extend(words)
    flush()
    return [Chunk(text=p, source=source, chunk_id=f"{source}#{i + 1}") for i, p in enumerate(pieces)]


class TfidfRetriever:
    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    def index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index - the document folder is empty.")
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

    @property
    def chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def retrieve_scored(self, query: str, top_k: int = 4, min_score: float = 0.0) -> List[Tuple[Chunk, float]]:
        if self._vectorizer is None:
            raise RuntimeError("Call index() before retrieving.")
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self._chunks[i], float(scores[i])) for i in ranked[:top_k] if scores[i] > min_score]

    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        return [c for c, _ in self.retrieve_scored(query, top_k)]

    def best_support(self, text: str, chunks: List[Chunk]) -> Tuple[Optional[Chunk], float]:
        """How well is `text` supported by this specific set of chunks?
        Returns the closest chunk and its cosine similarity (0 when nothing overlaps)."""
        if self._vectorizer is None:
            raise RuntimeError("Call index() before scoring.")
        if not chunks:
            return None, 0.0
        vecs = self._vectorizer.transform([text] + [c.text for c in chunks])
        sims = cosine_similarity(vecs[0], vecs[1:])[0]
        best = int(sims.argmax())
        return chunks[best], float(sims[best])
