"""Contrato que todo adapter de ``Embedder`` precisa cumprir. Para um adapter novo, adicione-o em ``test_embedders.py``."""

import numpy as np
import pytest

from workflows.apps.code_search.application.ports import Embedder
from workflows.apps.code_search.domain.model import EmbeddingProfile

TEXTS = ["calcula juros compostos", "valida o CPF do cliente", "estorna o pagamento", "calcula juros compostos"]


class EmbedderContract:
    @pytest.fixture
    def embedder(self):  # sobrescrito por cada adapter
        raise NotImplementedError

    def test_satisfies_port_and_declares_profile(self, embedder):
        assert isinstance(embedder, Embedder)
        p = embedder.profile
        assert isinstance(p, EmbeddingProfile) and p.dim > 0 and p.metric == "cosine" and p.profile_id

    def test_documents_are_float32_unit_norm_and_deterministic(self, embedder):
        a = embedder.embed_documents(TEXTS)
        b = embedder.embed_documents(TEXTS)
        assert a.dtype == np.float32 and a.shape == (len(TEXTS), embedder.profile.dim)
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
        assert np.allclose(a, b, atol=1e-6)
        assert np.allclose(a[0], a[3], atol=1e-6)  # mesmo texto, mesmo vetor, independente da posição no lote

    def test_empty_batch(self, embedder):
        out = embedder.embed_documents([])
        assert out.shape == (0, embedder.profile.dim) and out.dtype == np.float32

    def test_query_is_unit_vector_of_profile_dim(self, embedder):
        q = embedder.embed_query("onde valida o CPF?")
        assert q.dtype == np.float32 and q.shape == (embedder.profile.dim,)
        assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5

    def test_different_texts_do_not_collapse_to_one_vector(self, embedder):
        v = embedder.embed_documents(TEXTS[:3])
        assert not np.allclose(v[0], v[1]) and not np.allclose(v[1], v[2])
