"""Cache de embeddings por (perfil, chave de conteúdo). Evita reembedar código inalterado.

Cada gravação cria um shard novo e imutável; o arquivo de chaves é publicado por último,
então um shard só existe para leitura quando os dois arquivos estão completos.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...domain.errors import CodeSearchError
from ...domain.model import EmbeddingProfile
from .file_store import _load_npy, _save_npy


def _key_bytes(key: str) -> bytes:
    return bytes.fromhex(key)[:16]


class EmbeddingCache:
    def __init__(self, root: Path | str, profile: EmbeddingProfile) -> None:
        self.profile = profile
        self._dir = Path(root) / profile.profile_id
        self._dir.mkdir(parents=True, exist_ok=True)

    def lookup(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        wanted = {_key_bytes(k): k for k in keys}
        found: dict[str, np.ndarray] = {}
        if not wanted:
            return found
        for keys_file in sorted(self._dir.glob("shard-*.keys.npy")):
            vecs_file = keys_file.with_name(keys_file.name.replace(".keys.", ".vecs."))
            if not vecs_file.exists():
                continue
            try:
                shard_keys = _load_npy(keys_file)
                vecs = _load_npy(vecs_file)
            except (OSError, ValueError):
                continue
            if shard_keys.shape[0] != vecs.shape[0] or vecs.ndim != 2 or vecs.shape[1] != self.profile.dim:
                continue
            for row in range(shard_keys.shape[0]):
                key = wanted.get(shard_keys[row].tobytes())
                if key is not None and key not in found:
                    found[key] = np.array(vecs[row], dtype=np.float32)
        return found

    def add(self, vectors: Mapping[str, np.ndarray]) -> None:
        if not vectors:
            return
        keys = list(vectors)
        mat = np.stack([np.asarray(vectors[k], dtype=np.float32) for k in keys])
        norms = np.linalg.norm(mat, axis=1)
        if mat.shape[1] != self.profile.dim or not np.all(np.isfinite(norms)) or np.any(norms < 1e-6):
            raise CodeSearchError("embedder_invalid_output", "refusing to cache a null or malformed vector")
        key_arr = np.frombuffer(b"".join(_key_bytes(k) for k in keys), dtype=np.uint8).reshape(len(keys), 16)
        name = f"shard-{uuid.uuid4().hex[:12]}"
        vecs_tmp = self._dir / f"{name}.vecs.npy.tmp"
        keys_tmp = self._dir / f"{name}.keys.npy.tmp"
        try:
            _save_npy(vecs_tmp, mat)
            os.replace(vecs_tmp, self._dir / f"{name}.vecs.npy")
            _save_npy(keys_tmp, key_arr)
            os.replace(keys_tmp, self._dir / f"{name}.keys.npy")  # publica o shard
        finally:
            for tmp in (vecs_tmp, keys_tmp):
                tmp.unlink(missing_ok=True)
