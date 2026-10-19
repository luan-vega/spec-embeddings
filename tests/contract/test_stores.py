"""Roda o contrato de VectorStore contra cada adapter. Adicione aqui o seu adapter novo."""

import pytest

from store_contract import StoreContract
from workflows.apps.code_search.adapters.driven.file_store import FileVectorStore


class TestFileVectorStore(StoreContract):
    @pytest.fixture
    def store(self, tmp_path):
        return FileVectorStore(tmp_path / "index")
