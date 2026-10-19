"""Bootstrap único: as dependências são criadas uma vez e injetadas; CLI, HTTP e MCP usam o MESMO serviço."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .adapters.driven.embedding_cache import EmbeddingCache as FileEmbeddingCache
from .adapters.driven.fakes import BagOfWordsEmbedder
from .adapters.driven.file_store import FileVectorStore
from .adapters.driven.git_source import GitSource
from .adapters.driven.ollama import DEFAULT_URL, OllamaEmbedder
from .adapters.driven.policy_store import load_policy
from .adapters.driven.registry import Registry
from .application.indexer import IndexService
from .application.limits import BoundedRunner
from .application.ports import Embedder, SparseEncoder, VectorStore
from .application.search import SearchService
from .config import Paths
from .domain.errors import CodeSearchError
from .domain.policy import ExclusionPolicy
from .domain.sparse import JavaSparseEncoder


@dataclass(frozen=True)
class AdapterSpec:
    factory: Callable[..., Any]
    local: bool = True  # adapters que falam com a rede só entram com CODE_SEARCH_ALLOW_REMOTE=1


# Raiz de composição: para trocar o Ollama, o modelo ou o armazenamento, registre outro adapter aqui
# (ou com ``register_embedder`` / ``register_store``) e escolha-o por configuração. Ver docs/arquitetura.md.
EMBEDDERS: dict[str, AdapterSpec] = {
    "ollama": AdapterSpec(
        lambda env: OllamaEmbedder(
            env.get("CODE_SEARCH_OLLAMA_MODEL", ""),
            url=env.get("CODE_SEARCH_OLLAMA_URL", DEFAULT_URL),
            doc_prefix=env.get("CODE_SEARCH_DOC_PREFIX", ""),
            query_prefix=env.get("CODE_SEARCH_QUERY_PREFIX", ""),
        )
    ),
    "fake": AdapterSpec(lambda env: BagOfWordsEmbedder()),  # só para desenvolvimento e testes de ponta a ponta
}
STORES: dict[str, AdapterSpec] = {
    "file": AdapterSpec(lambda paths, env: FileVectorStore(paths.home / "index")),
}


def register_embedder(name: str, factory: Callable[[Mapping[str, str]], Embedder], *, local: bool = True) -> None:
    EMBEDDERS[name] = AdapterSpec(factory, local)


def register_store(name: str, factory: Callable[[Paths, Mapping[str, str]], VectorStore], *, local: bool = True) -> None:
    STORES[name] = AdapterSpec(factory, local)


def _select(kind: str, registry: dict[str, AdapterSpec], name: str, env: Mapping[str, str], code: str) -> AdapterSpec:
    spec = registry.get(name)
    if spec is None:
        raise CodeSearchError(code, f"unknown {kind} {name!r}; available: {', '.join(sorted(registry))}")
    if not spec.local and env.get("CODE_SEARCH_ALLOW_REMOTE") != "1":
        raise CodeSearchError(code, f"{kind} {name!r} is not local; set CODE_SEARCH_ALLOW_REMOTE=1 to opt in")
    return spec


def load_embedder(env: Mapping[str, str] | None = None) -> Embedder:
    env = os.environ if env is None else env
    name = env.get("CODE_SEARCH_EMBEDDER") or ("fake" if env.get("CODE_SEARCH_DEV_FAKE_EMBEDDER") == "1" else "ollama")
    return _select("embedder", EMBEDDERS, name, env, "embedder_unavailable").factory(env)


def load_store(paths: Paths, env: Mapping[str, str] | None = None) -> VectorStore:
    env = os.environ if env is None else env
    name = env.get("CODE_SEARCH_STORE") or "file"
    return _select("store", STORES, name, env, "vector_store_unavailable").factory(paths, env)


@dataclass
class Runtime:
    paths: Paths
    registry: Registry
    store: VectorStore
    policy: ExclusionPolicy
    sparse: SparseEncoder
    runner: BoundedRunner
    embedder: Embedder | None = None
    cache: Any = None
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
                "previous_generation": self.store.previous_generation_id(),
                "created_at": snapshot.created_at,
                "profile_id": snapshot.profile.profile_id,
                "model": snapshot.profile.model,
                "dim": snapshot.profile.dim,
                "schema": snapshot.schema.to_dict(),
            }
        for entry in self.registry.list():
            seg = snapshot.segment_info(entry.repo_id) if snapshot else None
            out["repos"].append(
                {
                    "repo_id": entry.repo_id,
                    "enabled": entry.enabled,
                    "indexed": seg is not None,
                    "repo_revision": seg.repo_revision if seg else None,
                    "indexed_at": seg.indexed_at if seg else None,
                    "chunks": seg.chunks if seg else 0,
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
    store = load_store(paths)
    policy = load_policy(paths.policy)
    sparse = JavaSparseEncoder()
    runtime = Runtime(paths, registry, store, policy, sparse, BoundedRunner(max_concurrent, timeout_s))
    if with_embedder:
        runtime.embedder = embedder if embedder is not None else load_embedder()
        runtime.cache = FileEmbeddingCache(paths.cache, runtime.embedder.profile)
        runtime.search = SearchService(store=store, registry=registry, embedder=runtime.embedder, sparse=sparse)
        runtime.indexer = IndexService(
            store=store, registry=registry, embedder=runtime.embedder, sparse=sparse, policy=policy,
            cache=runtime.cache, git_factory=lambda e: GitSource(e.path, e.ref),
        )
    return runtime
