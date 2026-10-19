"""Roda o contrato de Embedder contra cada adapter. Adicione aqui o seu adapter novo."""

import pytest

from embedder_contract import EmbedderContract
from ollama_fake import FakeOllama
from workflows.apps.code_search.adapters.driven.fakes import BagOfWordsEmbedder, FakeEmbedder
from workflows.apps.code_search.adapters.driven.ollama import OllamaEmbedder


class TestBagOfWords(EmbedderContract):
    @pytest.fixture
    def embedder(self):
        return BagOfWordsEmbedder()


class TestFakeEmbedder(EmbedderContract):
    @pytest.fixture
    def embedder(self):
        return FakeEmbedder(dim=16)


class TestOllama(EmbedderContract):
    @pytest.fixture
    def embedder(self):
        server = FakeOllama(dim=12)
        yield OllamaEmbedder("nomic-embed-text", url=server.url)
        server.close()
