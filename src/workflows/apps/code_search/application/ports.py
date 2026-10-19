"""Portas: o que o núcleo exige do mundo externo. Os adapters em ``adapters/driven`` as implementam.

Trocar o Ollama, o modelo de embedding ou o armazenamento significa escrever outro adapter que passe nos testes de
contrato (``tests/contract``) e escolhê-lo em ``bootstrap``; nada em ``domain`` ou ``application`` muda.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..domain.model import (
    BlobEntry,
    ChunkRecord,
    EmbeddingProfile,
    Filters,
    Hit,
    IndexSchema,
    Location,
    RepoEntry,
    SegmentInfo,
    SegmentRef,
)


@runtime_checkable
class Embedder(Protocol):
    """Gera vetores. O perfil (modelo, dimensão, normalização, prefixos) pertence ao adapter, não ao núcleo."""

    profile: EmbeddingProfile

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, normalizado. Falha com embedder_unavailable; nunca devolve zeros."""

    def embed_query(self, text: str) -> np.ndarray:
        """(dim,) float32, normalizado."""


@runtime_checkable
class SparseEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> list[dict[str, int]]:
        """Termo -> frequência, determinístico."""

    def encode_query(self, text: str) -> dict[str, float]:
        """Termo -> peso, determinístico."""


class IndexSnapshot(Protocol):
    """Visão consistente de uma geração publicada; uma por requisição."""

    generation_id: str
    created_at: str
    profile: EmbeddingProfile
    schema: IndexSchema

    @property
    def repos(self) -> list[str]: ...

    def segment_info(self, repo_id: str) -> SegmentInfo | None: ...

    def coverage(self, repos: Sequence[str]) -> tuple[list[str], list[str]]:
        """(com índice, sem índice), preservando a ordem pedida."""

    def search(
        self,
        repos: Sequence[str],
        *,
        dense_q: np.ndarray | None = None,
        lexical_terms: Iterable[str] | None = None,
        filters: Filters = Filters(),
        mode: str = "hybrid",
        depth: int = 100,
    ) -> dict[str, list[Hit]]:
        """Rankings por sinal: chaves ``dense`` e ``lexical``, ordenados por score decrescente."""

    def locate(self, hit: Hit) -> Location: ...

    def record(self, hit: Hit) -> dict[str, Any]: ...


class VectorStore(Protocol):
    def open_active(self) -> IndexSnapshot | None:
        """Geração ativa ou None se ainda não há índice."""

    def write_segment(
        self,
        *,
        repo_id: str,
        repo_revision: str,
        indexed_at: str,
        profile: EmbeddingProfile,
        schema: IndexSchema,
        chunks: Sequence[ChunkRecord],
        dense: np.ndarray,
        roots: Sequence[str] = (),
    ) -> SegmentRef:
        """Monta e valida um segmento sem publicá-lo."""

    def publish(
        self,
        *,
        profile: EmbeddingProfile,
        schema: IndexSchema,
        replace: Sequence[SegmentRef] = (),
        remove: Sequence[str] = (),
    ) -> str:
        """Nova geração (ativa + replace - remove) tornada ativa de forma atômica; devolve o id."""

    def rollback(self) -> str: ...

    def prune(self, *, tmp_max_age: float = 3600.0) -> dict[str, int]: ...

    def previous_generation_id(self) -> str | None: ...


class RepoRegistry(Protocol):
    def list(self) -> list[RepoEntry]: ...

    def enabled(self) -> list[RepoEntry]: ...


class SourceReader(Protocol):
    """Lê arquivos de uma revisão fixa de um repositório."""

    def revision(self) -> str: ...

    def list_blobs(self, roots: Iterable[str]) -> list[BlobEntry]: ...

    def read_blobs(self, shas: Sequence[str]) -> Iterable[tuple[str, bytes]]: ...


SourceReaderFactory = Callable[[RepoEntry], SourceReader]


class EmbeddingCache(Protocol):
    def lookup(self, keys: Sequence[str]) -> Mapping[str, np.ndarray]: ...

    def add(self, vectors: Mapping[str, np.ndarray]) -> None: ...
