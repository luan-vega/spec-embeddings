"""Tipos e regras puras do domínio. Sem I/O e sem dependência de adapters."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .errors import CodeSearchError

_ID_NAMESPACE = uuid.UUID("6f1c9d52-0b0e-5a43-9d3e-2c7a1b8e4f10")


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


@dataclass(frozen=True)
class SegmentRef:
    """Referência a um segmento já construído, trocada entre indexador e armazenamento."""

    repo_id: str
    segment_id: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"repo_id": self.repo_id, "segment_id": self.segment_id, "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class SegmentInfo:
    """O que o núcleo precisa saber de um segmento indexado, sem conhecer o formato de armazenamento."""

    repo_id: str
    repo_revision: str
    indexed_at: str
    roots: tuple[str, ...]
    profile_id: str
    schema_id: str
    chunks: int


@dataclass(frozen=True)
class Location:
    source: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class BlobEntry:
    path: str
    sha: str
    mode: str


DEFAULT_ROOTS = ("src/main/java",)


def normalize_root(root: str) -> str:
    normalized = root.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        raise CodeSearchError("invalid_request", "roots must be relative to the repository")
    parts = tuple(p for p in PurePosixPath(normalized).parts if p not in ("", "."))
    if not parts or ".." in parts:
        raise CodeSearchError("invalid_request", "invalid root")
    return "/".join(parts)


@dataclass(frozen=True)
class RepoEntry:
    repo_id: str
    path: str  # caminho local; nunca é exposto pela API
    roots: tuple[str, ...] = DEFAULT_ROOTS
    enabled: bool = True
    ref: str = "HEAD"

    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "path": self.path,
            "roots": list(self.roots),
            "enabled": self.enabled,
            "ref": self.ref,
        }
