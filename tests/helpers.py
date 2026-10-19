import numpy as np

from workflows.apps.code_search.core import ChunkRecord, EmbeddingProfile, IndexSchema, make_chunk_id
from workflows.apps.code_search.testing import simple_terms

PROFILE = EmbeddingProfile(model="fake", digest="sha256:abc", dim=4)
SCHEMA = IndexSchema(chunker_version="1", tokenizer_version="1", policy_version="1")


def chunk(repo, source, text, start=1, language="java", symbol="", parent="", profile=PROFILE):
    cid = make_chunk_id(profile.profile_id, repo, source, symbol or text[:10], f"h{abs(hash(text))}", start)
    return ChunkRecord(
        chunk_id=cid,
        source=source,
        language=language,
        symbol=symbol,
        parent_symbol=parent,
        start_line=start,
        end_line=start + 9,
        text=text,
        terms=simple_terms(text),
    )


def basis(*indexes, dim=4):
    """Vetores unitários (um por índice) para rankings conhecidos."""
    out = np.zeros((len(indexes), dim), dtype=np.float32)
    for row, i in enumerate(indexes):
        out[row, i] = 1.0
    return out


def write(store, repo, chunks, dense, revision="rev1", profile=PROFILE, schema=SCHEMA):
    return store.write_segment(
        repo_id=repo,
        repo_revision=revision,
        indexed_at="2026-10-19T12:00:00Z",
        profile=profile,
        schema=schema,
        chunks=chunks,
        dense=dense,
    )
