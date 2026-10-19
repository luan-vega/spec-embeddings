import socket

import numpy as np
import pytest

from workflows.apps.code_search.domain.errors import CodeSearchError
from workflows.apps.code_search.domain.model import EmbeddingProfile, Filters, IndexSchema, make_chunk_id
from workflows.apps.code_search.application.ports import Embedder
from workflows.apps.code_search.adapters.driven.fakes import FakeEmbedder, simple_terms


def test_profile_id_changes_with_any_field():
    base = EmbeddingProfile(model="m", digest="d", dim=8)
    variants = [
        EmbeddingProfile(model="m2", digest="d", dim=8),
        EmbeddingProfile(model="m", digest="d2", dim=8),
        EmbeddingProfile(model="m", digest="d", dim=9),
        EmbeddingProfile(model="m", digest="d", dim=8, doc_prefix="doc: "),
        EmbeddingProfile(model="m", digest="d", dim=8, query_prefix="q: "),
        EmbeddingProfile(model="m", digest="d", dim=8, max_input_tokens=512),
    ]
    ids = {base.profile_id, *(v.profile_id for v in variants)}
    assert len(ids) == len(variants) + 1


def test_schema_id_changes_with_chunker_version():
    a = IndexSchema("1", "1", "1")
    assert a.schema_id != IndexSchema("2", "1", "1").schema_id
    assert a.schema_id == IndexSchema("1", "1", "1").schema_id


def test_chunk_id_is_deterministic_and_content_sensitive():
    a = make_chunk_id("p", "repo", "A.java", "A.m", "h1", 10)
    assert a == make_chunk_id("p", "repo", "A.java", "A.m", "h1", 10)
    assert a != make_chunk_id("p", "repo", "A.java", "A.m", "h2", 10)
    assert a != make_chunk_id("p2", "repo", "A.java", "A.m", "h1", 10)
    assert a != make_chunk_id("p", "repo", "A.java", "A.m", "h1", 11)


def test_fake_embedder_is_deterministic_and_never_null():
    emb = FakeEmbedder(dim=8)
    assert isinstance(emb, Embedder)
    docs = emb.embed_documents(["a", "b", ""])
    assert docs.shape == (3, 8) and docs.dtype == np.float32
    assert np.all(np.linalg.norm(docs, axis=1) > 0.99)
    assert np.array_equal(docs, emb.embed_documents(["a", "b", ""]))
    assert not np.array_equal(docs[0], docs[1])


def test_simple_terms_splits_camel_and_snake():
    assert simple_terms("calcularJurosComposto") == {"calcular": 1, "juros": 1, "composto": 1}
    assert simple_terms("max_retry_count max") == {"max": 2, "retry": 1, "count": 1}


def test_filters_reject_absolute_and_parent_paths():
    for bad in ("/etc", "C:/x", "C:\\x", "a/../b"):
        with pytest.raises(CodeSearchError) as exc:
            Filters.make(path_prefix=bad)
        assert exc.value.code == "invalid_request"
    assert Filters.make(path_prefix="src\\main/java").path_prefix == ("src", "main", "java")


def test_error_http_status():
    assert CodeSearchError("profile_mismatch", "x").http_status == 409
    assert CodeSearchError("timeout", "x").http_status == 504


def test_external_network_is_blocked():
    with pytest.raises(AssertionError):
        socket.create_connection(("93.184.216.34", 80), timeout=1)
    with pytest.raises(AssertionError):
        socket.getaddrinfo("example.com", 80)


def test_loopback_is_allowed():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname(), timeout=2)
    client.close()
    server.close()
