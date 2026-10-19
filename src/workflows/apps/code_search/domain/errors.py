"""Erros com código estável, compartilhados por todas as camadas."""

from __future__ import annotations

ERROR_HTTP_STATUS = {
    "invalid_request": 400,
    "no_repos_available": 404,
    "repo_unavailable": 404,
    "profile_mismatch": 409,
    "embedder_unavailable": 503,
    "vector_store_unavailable": 503,
    "store_busy": 503,
    "overloaded": 503,
    "timeout": 504,
    "payload_too_large": 413,
    "not_found": 404,
    "method_not_allowed": 405,
    "internal_error": 500,
    "embedder_invalid_output": 502,
}


class CodeSearchError(Exception):
    """Erro com código estável. A mensagem nunca carrega caminhos absolutos."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS.get(self.code, 500)
