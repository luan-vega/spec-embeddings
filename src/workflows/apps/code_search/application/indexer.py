"""Indexação: Git HEAD -> política -> chunks -> embeddings (com cache) -> segmento -> geração."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..domain.chunker import CHUNKER_VERSION, ChunkDraft, chunk_java
from ..domain.errors import CodeSearchError
from ..domain.model import ChunkRecord, IndexSchema, RepoEntry, make_chunk_id
from ..domain.policy import ExclusionPolicy
from .ports import ChunkDescriber, EmbeddingCache, Embedder, RepoRegistry, SourceReaderFactory, SparseEncoder, VectorStore


@dataclass
class RepoReport:
    repo_id: str
    revision: str = ""
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped_policy: int = 0
    files_skipped_other: int = 0
    chunks: int = 0
    chunks_skipped_secret: int = 0
    embeddings_new: int = 0
    embeddings_cached: int = 0
    unchanged: bool = False


@dataclass
class IndexReport:
    status: str  # published | unchanged | skipped
    reason: str = ""
    generation_id: str | None = None
    repos: list[RepoReport] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class IndexService:
    def __init__(
        self,
        *,
        store: VectorStore,
        registry: RepoRegistry,
        embedder: Embedder,
        sparse: SparseEncoder,
        policy: ExclusionPolicy,
        cache: EmbeddingCache,
        batch_size: int = 32,
        now: Callable[[], str] = _utc_now,
        git_factory: SourceReaderFactory,
        describer: ChunkDescriber | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._embedder = embedder
        self._sparse = sparse
        self._policy = policy
        self._cache = cache
        self._batch = batch_size
        self._now = now
        self._git = git_factory
        self._describer = describer

    @property
    def _describer_id(self) -> str:
        return self._describer.describer_id if self._describer is not None else ""

    @property
    def schema(self) -> IndexSchema:
        return IndexSchema(
            chunker_version=CHUNKER_VERSION,
            tokenizer_version=getattr(self._sparse, "version", "1"),
            policy_version=self._policy.version_id,
            describer_id=self._describer_id,
        )

    def index(self, repo_ids: Sequence[str] | None = None, *, force: bool = False) -> IndexReport:
        """Reindexa os repos habilitados (ou só ``repo_ids``). Repo com o mesmo commit, roots, perfil e schema do
        segmento ativo é pulado (a menos de ``force``). Nada é publicado se algo falhar."""
        if not self._policy.valid:
            return IndexReport(status="skipped", reason=self._policy.reason)
        enabled = {e.repo_id: e for e in self._registry.enabled()}
        if repo_ids is None:
            targets = list(enabled.values())
        else:
            unknown = [r for r in repo_ids if r not in enabled]
            if unknown:
                raise CodeSearchError("invalid_request", "repository is unknown or disabled")
            targets = [enabled[r] for r in repo_ids]

        snapshot = self._store.open_active()
        refs, reports = [], []
        for entry in targets:
            if not force and self._is_current(snapshot, entry):
                reports.append(RepoReport(entry.repo_id, revision=snapshot.segment_info(entry.repo_id).repo_revision, unchanged=True))
                continue
            ref, report = self._index_repo(entry)
            refs.append(ref)
            reports.append(report)

        remove: list[str] = []
        if repo_ids is None and snapshot is not None:
            remove = [r for r in snapshot.repos if r not in enabled]
        if not refs and not remove and snapshot is not None:
            return IndexReport(status="unchanged", generation_id=snapshot.generation_id, repos=reports)
        generation = self._store.publish(
            profile=self._embedder.profile, schema=self.schema, replace=refs, remove=remove
        )
        return IndexReport(status="published", generation_id=generation, repos=reports, removed=remove)

    def _is_current(self, snapshot, entry: RepoEntry) -> bool:
        info = snapshot.segment_info(entry.repo_id) if snapshot is not None else None
        if info is None:
            return False
        profile = self._embedder.profile
        return (
            info.repo_revision == self._git(entry).revision()
            and info.roots == tuple(entry.roots)
            and info.profile_id == profile.profile_id
            and info.schema_id == self.schema.schema_id
        )

    def _index_repo(self, entry: RepoEntry):
        report = RepoReport(entry.repo_id)
        git = self._git(entry)
        report.revision = git.revision()
        blobs = git.list_blobs(entry.roots)
        report.files_seen = len(blobs)

        allowed = [b for b in blobs if self._policy.allows(b.path)]  # decidido antes de abrir qualquer arquivo
        report.files_skipped_policy = len(blobs) - len(allowed)
        path_of: dict[str, list[str]] = {}
        for b in allowed:
            path_of.setdefault(b.sha, []).append(b.path)

        drafts: list[ChunkDraft] = []
        for sha, raw in git.read_blobs(list(path_of)):
            for path in path_of[sha]:
                if len(raw) > self._policy.max_file_bytes:
                    report.files_skipped_other += 1
                    continue
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    report.files_skipped_other += 1
                    continue
                report.files_indexed += 1
                for draft in chunk_java(path, text):
                    if self._policy.contains_secret(draft.text):
                        report.chunks_skipped_secret += 1
                    else:
                        drafts.append(draft)
        drafts.sort(key=lambda d: (d.source, d.start_line, d.end_line, d.symbol))
        report.chunks = len(drafts)

        keys = [d.embed_key(entry.repo_id, self._describer_id) for d in drafts]
        vectors = self._embed(entry.repo_id, drafts, keys, report)
        dense = np.stack([vectors[k] for k in keys]) if keys else np.empty((0, self._embedder.profile.dim), np.float32)

        profile = self._embedder.profile
        lexical = self._sparse.encode_documents([f"{d.source}\n{d.symbol}\n{d.text}" for d in drafts])
        records = [
            ChunkRecord(
                make_chunk_id(profile.profile_id, entry.repo_id, d.source, d.symbol, d.content_hash, d.start_line),
                d.source, "java", d.symbol, d.parent_symbol, d.start_line, d.end_line, d.text, terms,
            )
            for d, terms in zip(drafts, lexical)
        ]
        ref = self._store.write_segment(
            repo_id=entry.repo_id,
            repo_revision=report.revision,
            indexed_at=self._now(),
            profile=profile,
            schema=self.schema,
            chunks=records,
            dense=dense,
            roots=entry.roots,
        )
        return ref, report

    def _embed(self, repo_id: str, drafts: list[ChunkDraft], keys: list[str], report: RepoReport) -> dict[str, np.ndarray]:
        found = self._cache.lookup(keys)
        report.embeddings_cached = len(found)
        pending_draft: dict[str, ChunkDraft] = {}
        for d, k in zip(drafts, keys):
            if k not in found and k not in pending_draft:
                pending_draft[k] = d
        pending = list(pending_draft)
        dim = self._embedder.profile.dim
        for i in range(0, len(pending), self._batch):
            batch = pending[i : i + self._batch]
            texts = [self._embed_text(repo_id, pending_draft[k]) for k in batch]  # descreve só o que não está no cache
            mat = np.asarray(self._embedder.embed_documents(texts), dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1) if mat.ndim == 2 else np.array([])
            if mat.shape != (len(batch), dim) or not np.all(np.isfinite(norms)) or np.any(norms < 1e-6):
                raise CodeSearchError("embedder_invalid_output", "embedder returned null or malformed vectors")
            fresh = {k: mat[j] for j, k in enumerate(batch)}
            self._cache.add(fresh)  # progresso parcial não se perde se um lote seguinte falhar
            found.update(fresh)
            report.embeddings_new += len(batch)
        return found

    def _embed_text(self, repo_id: str, draft: ChunkDraft) -> str:
        description = self._describer.describe(draft.symbol, draft.text) if self._describer is not None else ""
        return draft.embed_text(repo_id, description)
