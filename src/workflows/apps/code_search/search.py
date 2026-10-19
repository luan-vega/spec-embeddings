"""SearchService: valida, resolve repos, checa perfil, embeda a consulta, funde e responde."""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from .chunker import CHUNKER_VERSION
from .core import CodeSearchError, Embedder, Filters, Hit, SparseEncoder
from .file_store import FileVectorStore
from .fusion import RRF_K, collapse_overlaps, rrf
from .registry import Registry

MODES = ("hybrid", "semantic", "lexical")


class SearchService:
    def __init__(
        self,
        *,
        store: FileVectorStore,
        registry: Registry,
        embedder: Embedder,
        sparse: SparseEncoder,
        max_query_chars: int = 2000,
        max_top_k: int = 50,
        max_repos: int = 20,
        rrf_k: int = RRF_K,
    ) -> None:
        self._store = store
        self._registry = registry
        self._embedder = embedder
        self._sparse = sparse
        self._max_query = max_query_chars
        self._max_top_k = max_top_k
        self._max_repos = max_repos
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        repos: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        path_prefix: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        if not isinstance(query, str) or not query.strip() or len(query) > self._max_query:
            raise CodeSearchError("invalid_request", "query is empty or too long")
        if mode not in MODES:
            raise CodeSearchError("invalid_request", "mode must be hybrid, semantic or lexical")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= self._max_top_k:
            raise CodeSearchError("invalid_request", "top_k out of range")
        if repos is not None and (len(repos) == 0 or len(repos) > self._max_repos):
            raise CodeSearchError("invalid_request", "invalid number of repos")
        filters = Filters.make(languages, path_prefix)

        registered = {e.repo_id: e for e in self._registry.list()}
        requested = list(dict.fromkeys(repos)) if repos is not None else [r for r, e in registered.items() if e.enabled]
        snapshot = self._store.open_active()

        unavailable: list[dict[str, str]] = []
        searchable: list[str] = []
        for repo in requested:
            entry = registered.get(repo)
            if entry is None:
                unavailable.append({"repo": repo, "reason": "unknown_repo"})
            elif not entry.enabled:
                unavailable.append({"repo": repo, "reason": "disabled"})
            elif snapshot is None or repo not in snapshot.segments:
                unavailable.append({"repo": repo, "reason": "no_index"})
            else:
                searchable.append(repo)
        if not searchable or snapshot is None:
            raise CodeSearchError("no_repos_available", "none of the requested repositories has an index")

        profile = self._embedder.profile
        if snapshot.profile.profile_id != profile.profile_id or snapshot.schema.tokenizer_version != getattr(
            self._sparse, "version", snapshot.schema.tokenizer_version
        ) or snapshot.schema.chunker_version != CHUNKER_VERSION:
            raise CodeSearchError("profile_mismatch", "the active index was built with a different profile or schema")

        dense_q = self._embedder.embed_query(query) if mode in ("hybrid", "semantic") else None
        terms = list(self._sparse.encode_query(query)) if mode in ("hybrid", "lexical") else []
        depth = max(100, 5 * top_k)
        rankings = snapshot.search(
            searchable, dense_q=dense_q, lexical_terms=terms, filters=filters, mode=mode, depth=depth
        )

        fused = rrf([rankings["dense"], rankings["lexical"]], self._rrf_k)

        def locate(hit: Hit) -> tuple[str, int, int]:
            seg = snapshot.segments[hit.repo_id]
            return seg.sources[int(seg.source_idx[hit.row])], int(seg.start_line[hit.row]), int(seg.end_line[hit.row])

        top = collapse_overlaps(fused, locate, top_k)
        results = []
        for hit in top:
            rec = snapshot.record(hit)
            results.append(
                {
                    "repo": rec["repo"],
                    "source": rec["source"],
                    "symbol": rec["symbol"],
                    "parent_symbol": rec["parent_symbol"],
                    "start_line": rec["start_line"],
                    "end_line": rec["end_line"],
                    "language": rec["language"],
                    "score": hit.score,
                    "text": rec["text"],
                    "repo_revision": rec["repo_revision"],
                    "indexed_at": rec["indexed_at"],
                }
            )
        return {
            "status": "partial" if unavailable else "complete",
            "results": results,
            "coverage": {"searched": searchable, "unavailable": unavailable},
            "index": {"profile_id": snapshot.profile.profile_id, "generation": snapshot.generation_id},
            "request_id": request_id,
        }
