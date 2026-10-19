"""Contratos, tipos e erros compartilhados. Sem I/O."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

_ID_NAMESPACE = uuid.UUID("6f1c9d52-0b0e-5a43-9d3e-2c7a1b8e4f10")

ERROR_HTTP_STATUS = {
    "invalid_request": 400,
    "no_repos_available": 404,
    "repo_unavailable": 404,
    "profile_mismatch": 409,
    "embedder_unavailable": 503,
    "vector_store_unavailable": 503,
    "store_busy": 503,
    "overloaded": 503,
    "timeout": 504,
    "embedder_invalid_output": 502,
}


class CodeSearchError(Exception):
    """Erro com código estável. A mensagem nunca carrega caminhos absolutos."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS.get(self.code, 500)


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def short_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()[:16]


@dataclass(frozen=True)
class EmbeddingProfile:
    """Tudo que, ao mudar, torna os vetores incompatíveis."""

    model: str
    digest: str
    dim: int
    metric: str = "cosine"
    normalized: bool = True
    doc_prefix: str = ""
    query_prefix: str = ""
    max_input_tokens: int = 0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.metric != "cosine":
            raise ValueError("only the cosine metric is supported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddingProfile":
        return cls(**dict(data))

    @property
    def profile_id(self) -> str:
        return short_digest(self.to_dict())


@dataclass(frozen=True)
class IndexSchema:
    """Parâmetros de construção do índice que não são do modelo."""

    chunker_version: str
    tokenizer_version: str
    policy_version: str
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IndexSchema":
        return cls(**dict(data))

    @property
    def schema_id(self) -> str:
        return short_digest(self.to_dict())


@runtime_checkable
class Embedder(Protocol):
    profile: EmbeddingProfile

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32. Falha com embedder_unavailable; nunca devolve zeros."""

    def embed_query(self, text: str) -> np.ndarray:
        """(dim,) float32."""


@runtime_checkable
class SparseEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> list[dict[str, int]]:
        """Termo -> frequência, determinístico."""

    def encode_query(self, text: str) -> dict[str, float]:
        """Termo -> peso, determinístico."""


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source: str  # caminho relativo ao repo, separador "/"
    language: str
    symbol: str
    parent_symbol: str
    start_line: int
    end_line: int
    text: str
    terms: Mapping[str, int] = field(default_factory=dict)


def make_chunk_id(
    profile_id: str, repo_id: str, source: str, symbol_path: str, chunk_hash: str, start_line: int
) -> str:
    """UUIDv5 determinístico: conteúdo inalterado mantém o mesmo ID."""
    name = "\x1f".join((profile_id, repo_id, source, symbol_path, chunk_hash, str(start_line)))
    return str(uuid.uuid5(_ID_NAMESPACE, name))


@dataclass(frozen=True)
class Filters:
    languages: frozenset[str] | None = None
    path_prefix: tuple[str, ...] = ()

    @classmethod
    def make(cls, languages: Sequence[str] | None = None, path_prefix: str | None = None) -> "Filters":
        langs = frozenset(languages) if languages else None
        parts: tuple[str, ...] = ()
        if path_prefix:
            normalized = path_prefix.replace("\\", "/")
            if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
                raise CodeSearchError("invalid_request", "path_prefix must be relative")
            parts = tuple(p for p in PurePosixPath(normalized).parts if p not in ("", "."))
            if ".." in parts:
                raise CodeSearchError("invalid_request", "path_prefix must not contain '..'")
        return cls(langs, parts)


@dataclass(frozen=True)
class Hit:
    repo_id: str
    row: int
    score: float
