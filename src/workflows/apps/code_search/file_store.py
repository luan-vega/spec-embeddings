"""FileVectorStore: segmentos imutáveis por repo, agrupados em gerações publicadas atomicamente.

Layout sob ``root``::

    active.json / previous.json      ponteiros {"generation", "sha256"}
    generations/<id>.json            perfil, schema e lista de segmentos
    segments/<segment_id>/           arrays .npy, chunks.jsonl, vocab.json, manifest.json
    tmp/                             staging no mesmo volume (rename atômico)
    write.lock                       lock de escritores

Leitores não usam lock: leem ``active.json`` uma vez e trabalham sobre arquivos imutáveis.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np

from .core import (
    ChunkRecord,
    CodeSearchError,
    EmbeddingProfile,
    Filters,
    Hit,
    IndexSchema,
    canonical_json,
)
from .filelock import FileLock

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SEGMENT_FILES = (
    "dense.npy",
    "ids.npy",
    "source_idx.npy",
    "lang_idx.npy",
    "start_line.npy",
    "end_line.npy",
    "sources.json",
    "languages.json",
    "chunks.jsonl",
    "chunks.offsets.npy",
    "vocab.json",
    "df.npy",
    "postings_offsets.npy",
    "postings_doc.npy",
    "postings_tf.npy",
    "doc_len.npy",
)
_MODES = ("hybrid", "semantic", "lexical")


def _unavailable(message: str = "vector store unavailable") -> CodeSearchError:
    return CodeSearchError("vector_store_unavailable", message)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _retry_on_windows(fn, attempts: int = 100, delay: float = 0.01):
    """No Windows, abrir ou substituir um arquivo que outro handle está trocando levanta PermissionError por instantes."""
    for attempt in range(attempts):
        try:
            return fn()
        except PermissionError:
            if sys.platform != "win32" or attempt == attempts - 1:
                raise
            time.sleep(delay)


def _read_bytes(path: Path) -> bytes:
    return _retry_on_windows(path.read_bytes)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _retry_on_windows(lambda: os.replace(tmp, path))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _save_npy(path: Path, array: np.ndarray) -> None:
    with open(path, "wb") as fh:
        np.save(fh, array, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())


def _load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:  # mmap de arquivo sem dados (0 linhas)
        return np.load(path, allow_pickle=False)


def _check_repo_id(repo_id: str) -> None:
    if not _REPO_ID_RE.match(repo_id):
        raise CodeSearchError("invalid_request", "invalid repo_id")


def _top(scores: np.ndarray, depth: int) -> np.ndarray:
    """Índices dos ``depth`` maiores scores, ordenados; empate pelo menor índice."""
    n = scores.shape[0]
    if n == 0 or depth <= 0:
        return np.empty(0, dtype=np.int64)
    if depth < n:
        idx = np.argpartition(-scores, depth - 1)[:depth]
    else:
        idx = np.arange(n)
    return idx[np.lexsort((idx, -scores[idx]))]


@dataclass(frozen=True)
class SegmentRef:
    repo_id: str
    segment_id: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"repo_id": self.repo_id, "segment_id": self.segment_id, "manifest_sha256": self.manifest_sha256}


class Segment:
    """Leitura de um segmento imutável via memmap."""

    def __init__(self, seg_dir: Path, ref: SegmentRef) -> None:
        self.ref = ref
        self._dir = seg_dir
        self.manifest: dict[str, Any] = json.loads((seg_dir / "manifest.json").read_bytes())
        self.dense = _load_npy(seg_dir / "dense.npy")
        self.source_idx = _load_npy(seg_dir / "source_idx.npy")
        self.lang_idx = _load_npy(seg_dir / "lang_idx.npy")
        self.start_line = _load_npy(seg_dir / "start_line.npy")
        self.end_line = _load_npy(seg_dir / "end_line.npy")
        self.offsets = _load_npy(seg_dir / "chunks.offsets.npy")
        self.df = _load_npy(seg_dir / "df.npy")
        self.post_offsets = _load_npy(seg_dir / "postings_offsets.npy")
        self.post_doc = _load_npy(seg_dir / "postings_doc.npy")
        self.post_tf = _load_npy(seg_dir / "postings_tf.npy")
        self.doc_len = _load_npy(seg_dir / "doc_len.npy")
        self.sources: list[str] = json.loads((seg_dir / "sources.json").read_bytes())
        self.languages: list[str] = json.loads((seg_dir / "languages.json").read_bytes())
        self.vocab: dict[str, int] = json.loads((seg_dir / "vocab.json").read_bytes())
        self.count = int(self.dense.shape[0])
        self._lex = self.manifest["lexical"]

    def row_mask(self, filters: Filters) -> np.ndarray | None:
        mask: np.ndarray | None = None
        if filters.languages is not None:
            ids = [i for i, lang in enumerate(self.languages) if lang in filters.languages]
            mask = np.isin(self.lang_idx, ids)
        if filters.path_prefix:
            k = len(filters.path_prefix)
            ids = [i for i, s in enumerate(self.sources) if PurePosixPath(s).parts[:k] == filters.path_prefix]
            by_path = np.isin(self.source_idx, ids)
            mask = by_path if mask is None else (mask & by_path)
        return mask

    def search_dense(self, q: np.ndarray, depth: int, mask: np.ndarray | None) -> list[Hit]:
        if self.count == 0:
            return []
        if mask is None:
            scores = np.asarray(self.dense) @ q
            rows = None
        else:
            rows = np.flatnonzero(mask)
            if rows.size == 0:
                return []
            scores = np.asarray(self.dense[rows]) @ q
        top = _top(scores, depth)
        return [
            Hit(self.ref.repo_id, int(rows[i] if rows is not None else i), float(scores[i])) for i in top
        ]

    def search_lexical(self, terms: Iterable[str], depth: int, mask: np.ndarray | None) -> list[Hit]:
        if self.count == 0:
            return []
        k1, b = float(self._lex["k1"]), float(self._lex["b"])
        avgdl = float(self._lex["avgdl"]) or 1.0
        n = self.count
        scores = np.zeros(n, dtype=np.float64)
        dl_norm = k1 * (1.0 - b + b * (np.asarray(self.doc_len, dtype=np.float64) / avgdl))
        for term in set(terms):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo, hi = int(self.post_offsets[tid]), int(self.post_offsets[tid + 1])
            docs = np.asarray(self.post_doc[lo:hi])
            tf = np.asarray(self.post_tf[lo:hi], dtype=np.float64)
            df = float(self.df[tid])
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            scores[docs] += idf * tf * (k1 + 1.0) / (tf + dl_norm[docs])
        valid = scores > 0
        if mask is not None:
            valid &= mask
        rows = np.flatnonzero(valid)
        if rows.size == 0:
            return []
        sub = scores[rows]
        return [Hit(self.ref.repo_id, int(rows[i]), float(sub[i])) for i in _top(sub, depth)]

    def record(self, row: int) -> dict[str, Any]:
        if not 0 <= row < self.count:
            raise IndexError(row)
        lo, hi = int(self.offsets[row]), int(self.offsets[row + 1])
        with open(self._dir / "chunks.jsonl", "rb") as fh:
            fh.seek(lo)
            payload = json.loads(fh.read(hi - lo))
        payload.update(
            repo=self.ref.repo_id,
            source=self.sources[int(self.source_idx[row])],
            language=self.languages[int(self.lang_idx[row])],
            start_line=int(self.start_line[row]),
            end_line=int(self.end_line[row]),
            repo_revision=self.manifest["repo_revision"],
            indexed_at=self.manifest["indexed_at"],
        )
        return payload


class GenerationSnapshot:
    """Visão consistente de uma geração; uma por requisição."""

    def __init__(self, generation: dict[str, Any], segments: dict[str, Segment]) -> None:
        self.generation_id: str = generation["generation_id"]
        self.created_at: str = generation["created_at"]
        self.profile = EmbeddingProfile.from_dict(generation["embedding_profile"])
        self.schema = IndexSchema.from_dict(generation["index_schema"])
        self.segments = segments

    @property
    def repos(self) -> list[str]:
        return sorted(self.segments)

    def coverage(self, repos: Sequence[str]) -> tuple[list[str], list[str]]:
        """(com índice, sem índice) preservando a ordem pedida."""
        have = [r for r in repos if r in self.segments]
        missing = [r for r in repos if r not in self.segments]
        return have, missing

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
        """Rankings por sinal (a fusão RRF acontece no serviço de busca)."""
        if mode not in _MODES:
            raise CodeSearchError("invalid_request", "unknown mode")
        use_dense = mode in ("hybrid", "semantic")
        use_lex = mode in ("hybrid", "lexical")
        q = None
        if use_dense:
            if dense_q is None:
                raise CodeSearchError("invalid_request", "dense query vector required")
            q = np.asarray(dense_q, dtype=np.float32)
            if q.shape != (self.profile.dim,):
                raise CodeSearchError("profile_mismatch", "query vector dimension differs from the index")
            norm = float(np.linalg.norm(q))
            if not math.isfinite(norm) or norm < 1e-6:
                raise CodeSearchError("embedder_invalid_output", "query vector is null")
            q = q / norm
        terms = list(lexical_terms or ()) if use_lex else []
        out: dict[str, list[Hit]] = {"dense": [], "lexical": []}
        for repo in repos:
            seg = self.segments.get(repo)
            if seg is None:
                continue
            mask = seg.row_mask(filters)
            if q is not None:
                out["dense"].extend(seg.search_dense(q, depth, mask))
            if terms:
                out["lexical"].extend(seg.search_lexical(terms, depth, mask))
        for key, hits in out.items():
            hits.sort(key=lambda h: (-h.score, h.repo_id, h.row))
            del hits[depth:]
        return out

    def record(self, hit: Hit) -> dict[str, Any]:
        return self.segments[hit.repo_id].record(hit.row)


class FileVectorStore:
    def __init__(self, root: Path | str, lock_timeout: float = 10.0) -> None:
        self.root = Path(root)
        self._lock_timeout = lock_timeout
        for sub in ("generations", "segments", "tmp"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ escrita

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
    ) -> SegmentRef:
        """Monta o segmento em tmp/, valida e move para segments/. Não publica."""
        _check_repo_id(repo_id)
        n = len(chunks)
        mat = np.ascontiguousarray(dense, dtype=np.float32)
        if mat.shape != (n, profile.dim):
            raise CodeSearchError("invalid_request", "dense matrix does not match chunks and profile dim")
        norms = np.linalg.norm(mat, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms < 1e-6):
            raise CodeSearchError("embedder_invalid_output", "null or non-finite vector in segment")
        mat = mat / norms[:, None].astype(np.float32)

        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != n:
            raise CodeSearchError("invalid_request", "duplicate chunk ids in segment")
        try:
            ids_arr = np.frombuffer(b"".join(uuid.UUID(i).bytes for i in ids), dtype=np.uint8).reshape(n, 16)
        except ValueError:
            raise CodeSearchError("invalid_request", "chunk id is not a UUID") from None

        sources = sorted({c.source for c in chunks})
        languages = sorted({c.language for c in chunks})
        if len(languages) > 255:
            raise CodeSearchError("invalid_request", "too many languages")
        src_map = {s: i for i, s in enumerate(sources)}
        lang_map = {s: i for i, s in enumerate(languages)}

        vocab_terms = sorted({t for c in chunks for t in c.terms})
        vocab = {t: i for i, t in enumerate(vocab_terms)}
        per_term: list[list[tuple[int, int]]] = [[] for _ in vocab_terms]
        doc_len = np.zeros(n, dtype=np.int32)
        for doc, c in enumerate(chunks):
            doc_len[doc] = sum(c.terms.values())
            for term, tf in c.terms.items():
                per_term[vocab[term]].append((doc, tf))
        post_offsets = np.zeros(len(vocab_terms) + 1, dtype=np.int64)
        post_offsets[1:] = np.cumsum([len(p) for p in per_term]) if per_term else 0
        flat = [pair for plist in per_term for pair in plist]
        post_doc = np.array([d for d, _ in flat], dtype=np.int32)
        post_tf = np.array([t for _, t in flat], dtype=np.float32)
        df = np.array([len(p) for p in per_term], dtype=np.int32)

        tmp_dir = self.root / "tmp" / f"seg-{uuid.uuid4().hex}"
        tmp_dir.mkdir()
        try:
            lines: list[bytes] = []
            offsets = np.zeros(n + 1, dtype=np.uint64)
            pos = 0
            for i, c in enumerate(chunks):
                line = (
                    json.dumps(
                        {"id": c.chunk_id, "symbol": c.symbol, "parent_symbol": c.parent_symbol, "text": c.text},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                lines.append(line)
                pos += len(line)
                offsets[i + 1] = pos
            with open(tmp_dir / "chunks.jsonl", "wb") as fh:
                fh.write(b"".join(lines))
                fh.flush()
                os.fsync(fh.fileno())

            arrays = {
                "dense.npy": mat,
                "ids.npy": ids_arr,
                "source_idx.npy": np.array([src_map[c.source] for c in chunks], dtype=np.int32),
                "lang_idx.npy": np.array([lang_map[c.language] for c in chunks], dtype=np.uint8),
                "start_line.npy": np.array([c.start_line for c in chunks], dtype=np.int32),
                "end_line.npy": np.array([c.end_line for c in chunks], dtype=np.int32),
                "chunks.offsets.npy": offsets,
                "df.npy": df,
                "postings_offsets.npy": post_offsets,
                "postings_doc.npy": post_doc,
                "postings_tf.npy": post_tf,
                "doc_len.npy": doc_len,
            }
            for name, arr in arrays.items():
                _save_npy(tmp_dir / name, arr)
            for name, value in (("sources.json", sources), ("languages.json", languages), ("vocab.json", vocab)):
                _atomic_write_bytes(tmp_dir / name, json.dumps(value, ensure_ascii=False).encode("utf-8"))

            files = {
                name: {"sha256": _sha256_file(tmp_dir / name), "bytes": (tmp_dir / name).stat().st_size}
                for name in _SEGMENT_FILES
            }
            manifest: dict[str, Any] = {
                "repo_id": repo_id,
                "repo_revision": repo_revision,
                "indexed_at": indexed_at,
                "profile_id": profile.profile_id,
                "schema_id": schema.schema_id,
                "counts": {"chunks": n, "sources": len(sources), "terms": len(vocab_terms), "dim": profile.dim},
                "lexical": {
                    "k1": schema.bm25_k1,
                    "b": schema.bm25_b,
                    "avgdl": float(doc_len.mean()) if n else 0.0,
                },
                "files": files,
            }
            digest = hashlib.sha256(canonical_json(manifest)).hexdigest()[:16]
            segment_id = f"{repo_id}-{digest}"
            manifest["segment_id"] = segment_id
            manifest_bytes = json.dumps(manifest, sort_keys=True, indent=1).encode("utf-8")
            _atomic_write_bytes(tmp_dir / "manifest.json", manifest_bytes)
            ref = SegmentRef(repo_id, segment_id, hashlib.sha256(manifest_bytes).hexdigest())

            self._verify_segment(tmp_dir, ref, full=True)
            dest = self.root / "segments" / segment_id
            if dest.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                os.rename(tmp_dir, dest)
            return ref
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def publish(
        self,
        *,
        profile: EmbeddingProfile,
        schema: IndexSchema,
        replace: Sequence[SegmentRef] = (),
        remove: Sequence[str] = (),
    ) -> str:
        """Cria uma geração (ativa + ``replace`` - ``remove``) e a torna ativa de forma atômica."""
        with FileLock(self.root / "write.lock", timeout=self._lock_timeout):
            current = self._read_pointer("active.json")
            carried: dict[str, dict[str, str]] = {}
            if current is not None:
                gen, _ = self._read_generation(current[0])
                carried = {s["repo_id"]: s for s in gen["segments"]}
            for repo in remove:
                carried.pop(repo, None)
            fresh = {ref.repo_id: ref for ref in replace}
            for repo, ref in fresh.items():
                carried[repo] = ref.to_dict()

            for repo, entry in carried.items():
                ref = SegmentRef(**entry)
                try:
                    manifest = self._verify_segment(self.root / "segments" / ref.segment_id, ref, full=repo in fresh)
                except FileNotFoundError:
                    raise _unavailable("segment is missing") from None
                if manifest["profile_id"] != profile.profile_id or manifest["schema_id"] != schema.schema_id:
                    raise CodeSearchError("profile_mismatch", "segment was built with a different profile or schema")

            created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            body = {
                "created_at": created,
                "embedding_profile": profile.to_dict(),
                "index_schema": schema.to_dict(),
                "segments": [carried[r] for r in sorted(carried)],
            }
            gen_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
            gen_bytes = json.dumps({"generation_id": gen_id, **body}, sort_keys=True, indent=1).encode("utf-8")
            _atomic_write_bytes(self.root / "generations" / f"{gen_id}.json", gen_bytes)

            old_active = _read_bytes(self.root / "active.json") if current is not None else None
            pointer = {"generation": gen_id, "sha256": hashlib.sha256(gen_bytes).hexdigest()}
            _atomic_write_bytes(self.root / "active.json", json.dumps(pointer).encode("utf-8"))  # commit
            if old_active is not None:
                _atomic_write_bytes(self.root / "previous.json", old_active)
            return gen_id

    def rollback(self) -> str:
        """Volta para a geração anterior (blue/green); a atual passa a ser a anterior."""
        with FileLock(self.root / "write.lock", timeout=self._lock_timeout):
            prev = self._read_pointer("previous.json")
            if prev is None:
                raise _unavailable("no previous generation to roll back to")
            self._read_generation(prev[0])
            active_path = self.root / "active.json"
            old_active = _read_bytes(active_path) if active_path.exists() else None
            _atomic_write_bytes(active_path, _read_bytes(self.root / "previous.json"))
            if old_active is not None:
                _atomic_write_bytes(self.root / "previous.json", old_active)
            return prev[0]

    def prune(self, *, tmp_max_age: float = 3600.0) -> dict[str, int]:
        """Remoção explícita: mantém ativa e anterior; apaga o resto que ninguém referencia."""
        removed = {"generations": 0, "segments": 0, "skipped": 0}
        with FileLock(self.root / "write.lock", timeout=self._lock_timeout):
            keep = {p[0] for name in ("active.json", "previous.json") if (p := self._read_pointer(name))}
            referenced: set[str] = set()
            for gen_file in (self.root / "generations").glob("*.json"):
                if gen_file.stem in keep:
                    referenced.update(s["segment_id"] for s in json.loads(gen_file.read_bytes())["segments"])
                else:
                    gen_file.unlink()
                    removed["generations"] += 1
            for seg_dir in (self.root / "segments").iterdir():
                if seg_dir.name not in referenced:
                    try:
                        shutil.rmtree(seg_dir)
                        removed["segments"] += 1
                    except OSError:  # Windows: memmap ainda aberto por um leitor
                        removed["skipped"] += 1
            now = time.time()
            for leftover in (self.root / "tmp").iterdir():
                if now - leftover.stat().st_mtime > tmp_max_age:
                    shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() else leftover.unlink(missing_ok=True)
        return removed

    # ------------------------------------------------------------------ leitura

    def active_generation_id(self) -> str | None:
        ptr = self._read_pointer("active.json")
        return ptr[0] if ptr else None

    def open_active(self) -> GenerationSnapshot | None:
        """Snapshot da geração ativa, ou None se ainda não há índice."""
        for _ in range(3):
            try:
                return self._open_active_once()
            except FileNotFoundError:
                continue  # publicação + prune concorrentes; relê o ponteiro
        raise _unavailable()

    def _open_active_once(self) -> GenerationSnapshot | None:
        ptr = self._read_pointer("active.json")
        if ptr is None:
            return None
        gen, _ = self._read_generation(ptr[0], expected_sha=ptr[1])
        segments: dict[str, Segment] = {}
        for entry in gen["segments"]:
            ref = SegmentRef(**entry)
            seg_dir = self.root / "segments" / ref.segment_id
            self._verify_segment(seg_dir, ref, full=False)
            segments[ref.repo_id] = Segment(seg_dir, ref)
        return GenerationSnapshot(gen, segments)

    # ---------------------------------------------------------------- internos

    def _read_pointer(self, name: str) -> tuple[str, str] | None:
        try:
            raw = _read_bytes(self.root / name)
        except FileNotFoundError:
            return None
        try:
            data = json.loads(raw)
            return str(data["generation"]), str(data["sha256"])
        except (ValueError, KeyError, TypeError):
            raise _unavailable(f"{name} is corrupt") from None

    def _read_generation(self, gen_id: str, expected_sha: str | None = None) -> tuple[dict[str, Any], str]:
        if not re.fullmatch(r"[0-9TZ]+-[0-9a-f]{8}", gen_id):
            raise _unavailable("invalid generation id")
        raw = _read_bytes(self.root / "generations" / f"{gen_id}.json")
        sha = hashlib.sha256(raw).hexdigest()
        if expected_sha is not None and sha != expected_sha:
            raise _unavailable("generation file does not match its pointer")
        return json.loads(raw), sha

    def _verify_segment(self, seg_dir: Path, ref: SegmentRef, *, full: bool) -> dict[str, Any]:
        manifest_path = seg_dir / "manifest.json"
        try:
            raw = _read_bytes(manifest_path)
        except FileNotFoundError:
            if full:
                raise _unavailable("segment is missing") from None
            raise
        if hashlib.sha256(raw).hexdigest() != ref.manifest_sha256:
            raise _unavailable("segment manifest hash mismatch")
        manifest = json.loads(raw)
        for name, meta in manifest["files"].items():
            path = seg_dir / name
            if path.stat().st_size != meta["bytes"] or (full and _sha256_file(path) != meta["sha256"]):
                raise _unavailable("segment file integrity check failed")
        if manifest["counts"]["chunks"] != int(_load_npy(seg_dir / "dense.npy").shape[0]):
            raise _unavailable("segment chunk count mismatch")
        return manifest
