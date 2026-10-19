"""Bootstrap único: as dependências são criadas uma vez e injetadas; CLI, HTTP e MCP usam o MESMO serviço."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import Paths
from .core import CodeSearchError, Embedder, SparseEncoder
from .embedding_cache import EmbeddingCache
from .file_store import FileVectorStore
from .indexer import IndexService
from .limits import BoundedRunner
from .ollama import DEFAULT_URL, OllamaEmbedder
from .policy import ExclusionPolicy
from .registry import Registry
from .search import SearchService
from .sparse import JavaSparseEncoder
from .testing import BagOfWordsEmbedder


def load_embedder(env: Mapping[str, str] | None = None) -> Embedder:
    env = os.environ if env is None else env
    if env.get("CODE_SEARCH_DEV_FAKE_EMBEDDER") == "1":  # só para desenvolvimento e testes de ponta a ponta
        return BagOfWordsEmbedder()
    return OllamaEmbedder(
        env.get("CODE_SEARCH_OLLAMA_MODEL", ""),
        url=env.get("CODE_SEARCH_OLLAMA_URL", DEFAULT_URL),
        doc_prefix=env.get("CODE_SEARCH_DOC_PREFIX", ""),
        query_prefix=env.get("CODE_SEARCH_QUERY_PREFIX", ""),
    )


@dataclass
class Runtime:
    paths: Paths
    registry: Registry
    store: FileVectorStore
    policy: ExclusionPolicy
    sparse: SparseEncoder
    runner: BoundedRunner
    embedder: Embedder | None = None
    cache: EmbeddingCache | None = None
    search: SearchService | None = None
    indexer: IndexService | None = None

    def status(self) -> dict[str, Any]:
        """Estado do índice e dos repos. Sem caminhos locais."""
        out: dict[str, Any] = {
            "policy": {"valid": self.policy.valid, "version_id": self.policy.version_id, "reason": self.policy.reason},
            "embedder": None,
            "index": None,
            "repos": [],
        }
        if self.embedder is not None:
            p = self.embedder.profile
            out["embedder"] = {"model": p.model, "dim": p.dim, "profile_id": p.profile_id}
        snapshot = None
        try:
            snapshot = self.store.open_active()
        except CodeSearchError as exc:
            out["index"] = {"error": exc.code}
        if snapshot is not None:
            out["index"] = {
                "generation": snapshot.generation_id,
                "created_at": snapshot.created_at,
                "profile_id": snapshot.profile.profile_id,
                "model": snapshot.profile.model,
                "dim": snapshot.profile.dim,
                "schema": snapshot.schema.to_dict(),
            }
        for entry in self.registry.list():
            seg = snapshot.segments.get(entry.repo_id) if snapshot else None
            out["repos"].append(
                {
                    "repo_id": entry.repo_id,
                    "enabled": entry.enabled,
                    "indexed": seg is not None,
                    "repo_revision": seg.manifest["repo_revision"] if seg else None,
                    "indexed_at": seg.manifest["indexed_at"] if seg else None,
                    "chunks": seg.count if seg else 0,
                }
            )
        return out

    def ready(self) -> tuple[bool, dict[str, Any]]:
        """Modelo disponível + índice íntegro + perfil compatível."""
        checks: dict[str, Any] = {"embedder": "ok", "index": "ok", "profile": "ok"}
        if self.embedder is None:
            checks["embedder"] = "not_configured"
        else:
            probe = getattr(self.embedder, "ready", None)
            try:
                if callable(probe) and not probe():
                    checks["embedder"] = "unavailable"
            except CodeSearchError:
                checks["embedder"] = "unavailable"
        snapshot = None
        try:
            snapshot = self.store.open_active()
            if snapshot is None:
                checks["index"] = "no_index"
        except CodeSearchError as exc:
            checks["index"] = exc.code
        if snapshot is not None and self.embedder is not None:
            if snapshot.profile.profile_id != self.embedder.profile.profile_id:
                checks["profile"] = "mismatch"
        else:
            checks["profile"] = "unknown"
        return all(v == "ok" for v in checks.values()), checks


def build_runtime(
    home: Path | str | None = None,
    *,
    embedder: Embedder | None = None,
    with_embedder: bool = True,
    max_concurrent: int = 4,
    timeout_s: float = 30.0,
) -> Runtime:
    paths = Paths(home)
    paths.home.mkdir(parents=True, exist_ok=True)
    registry = Registry(paths.registry)
    store = FileVectorStore(paths.home / "index")
    policy = ExclusionPolicy.load(paths.policy)
    sparse = JavaSparseEncoder()
    runtime = Runtime(paths, registry, store, policy, sparse, BoundedRunner(max_concurrent, timeout_s))
    if with_embedder:
        runtime.embedder = embedder if embedder is not None else load_embedder()
        runtime.cache = EmbeddingCache(paths.cache, runtime.embedder.profile)
        runtime.search = SearchService(store=store, registry=registry, embedder=runtime.embedder, sparse=sparse)
        runtime.indexer = IndexService(
            store=store, registry=registry, embedder=runtime.embedder, sparse=sparse, policy=policy,
            cache=runtime.cache,
        )
    return runtime
