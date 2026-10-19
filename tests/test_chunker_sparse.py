import hashlib

import numpy as np
import pytest

from workflows.apps.code_search.domain.chunker import MAX_CHARS, MAX_LINES, OVERLAP, chunk_java, mask_java, split_windows
from workflows.apps.code_search.domain.errors import CodeSearchError
from workflows.apps.code_search.domain.model import EmbeddingProfile
from workflows.apps.code_search.adapters.driven.embedding_cache import EmbeddingCache
from workflows.apps.code_search.domain.sparse import JavaSparseEncoder, tokenize

SRC = '''package a.b;

import java.util.List;

/** Serviço de pagamento. */
public class Pagamento {
    private int taxa = 3; // campo
    @Autowired private Repo repo;

    /**
     * Calcula juros } { "falsos".
     */
    @Override
    public double calcularJurosComposto(double v, int prazo) throws Exception {
        String s = "}{ ; (";
        return v * Math.pow(1 + taxa, prazo);
    }

    public Pagamento(int taxa) { this.taxa = taxa; }

    static class Interno {
        void ping() { }
    }

    enum Tipo { A, B(1) { }; void m() { } }
    interface Porta { int abrir(String x); default void d() { } }
    record R(int a) { R { } int dobro() { return a*2; } }
}
'''


def by_symbol(chunks):
    out = {}
    for c in chunks:
        out.setdefault(c.symbol, []).append(c)
    return out


def test_mask_ignores_comments_and_literals_but_keeps_offsets():
    src = 'a // x { \n/* } */ "}" \'{\' b \n"""\n{ } \\""" """ c'
    masked = mask_java(src)
    assert len(masked) == len(src) and masked.count("\n") == src.count("\n")
    assert "{" not in masked and "}" not in masked
    assert masked.split() == ["a", "b", "c"]


def test_symbols_and_boundaries():
    chunks = chunk_java("src/main/java/a/b/Pagamento.java", SRC)
    sym = by_symbol(chunks)
    assert {"Pagamento.calcularJurosComposto", "Pagamento.Pagamento", "Pagamento.Interno.ping", "Pagamento.Tipo.m",
            "Pagamento.Porta.abrir", "Pagamento.Porta.d", "Pagamento.R.R", "Pagamento.R.dobro"} <= set(sym)
    calc = sym["Pagamento.calcularJurosComposto"][0]
    assert (calc.start_line, calc.end_line) == (10, 17)  # inclui Javadoc e @Override
    assert calc.text.lstrip().startswith("/**") and calc.text.rstrip().endswith("}")
    assert calc.parent_symbol == "Pagamento" and calc.class_name == "Pagamento"
    assert sym["Pagamento.Interno.ping"][0].parent_symbol == "Pagamento.Interno"
    # braces dentro de strings e comentários não quebram a delimitação
    assert "return v * Math.pow" in calc.text and "Interno" not in calc.text
    # cabeçalho e imports ficam num chunk do tipo envolvente / arquivo
    head = sym["Pagamento"][0]
    assert head.start_line == 1 and "import java.util.List;" in head.text and "private int taxa" in head.text


def test_every_code_line_is_covered_and_nothing_is_empty():
    chunks = chunk_java("A.java", SRC)
    lines = SRC.split("\n")
    covered = set()
    for c in chunks:
        assert c.text.strip()
        covered.update(range(c.start_line, c.end_line + 1))
    for i, line in enumerate(lines, start=1):
        if line.strip() and line.strip() not in {"}", "{"}:
            assert i in covered, f"linha {i} sem chunk: {line!r}"


def test_long_method_is_split_in_bounded_overlapping_windows():
    body = "\n".join(f"        total += valor{i} * 2; // passo {i}" for i in range(230))
    src = f"class Grande {{\n    void processar() {{\n{body}\n    }}\n}}\n"
    chunks = [c for c in chunk_java("Grande.java", src) if c.symbol == "Grande.processar"]
    assert len(chunks) >= 3
    for c in chunks:
        assert c.end_line - c.start_line + 1 <= MAX_LINES and len(c.text) <= MAX_CHARS
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_line - b.start_line + 1 == OVERLAP
    assert chunks[0].start_line == 2 and chunks[-1].end_line == 233


def test_window_limit_by_chars_and_single_giant_line():
    lines = ["x" * 2500] * 7
    for s, e in split_windows(lines, 1, 7):
        assert (e - s + 1) <= 2 and (e - s + 1) * 2501 <= MAX_CHARS + 2501
    src = "class A {\n  void m() { String s = \"" + "y" * 15000 + "\"; }\n}\n"
    big = [c for c in chunk_java("A.java", src) if c.symbol == "A.m"]
    assert len(big) == 3 and all(len(c.text) <= MAX_CHARS for c in big)


def test_chunk_identity_and_embed_text():
    a = chunk_java("A.java", "class A {\n  int f() { return 1; }\n}\n")
    shifted = chunk_java("A.java", "// novo\n\nclass A {\n  int f() { return 1; }\n}\n")
    fa = next(c for c in a if c.symbol == "A.f")
    fs = next(c for c in shifted if c.symbol == "A.f")
    assert fa.start_line != fs.start_line
    assert fa.embed_key("r") == fs.embed_key("r")  # só mudou de posição: reaproveita o vetor
    assert fa.embed_key("r") != fa.embed_key("outro")
    changed = chunk_java("A.java", "class A {\n  int f() { return 2; }\n}\n")
    assert next(c for c in changed if c.symbol == "A.f").embed_key("r") != fa.embed_key("r")
    header = fa.embed_text("svc-a")
    assert header.startswith("repo: svc-a\npath: A.java\nclass: A\nsymbol: A.f\nlines: 2-2\n\n")


def test_crlf_and_empty_files():
    assert chunk_java("E.java", "") == []
    assert chunk_java("E.java", "\n\n") == []
    unix = chunk_java("A.java", "class A {\n  int f() { return 1; }\n}\n")
    crlf = chunk_java("A.java", "class A {\r\n  int f() { return 1; }\r\n}\r\n")
    assert [(c.symbol, c.start_line, c.end_line, c.text) for c in unix] == [
        (c.symbol, c.start_line, c.end_line, c.text) for c in crlf
    ]


def test_tokenizer_camel_snake_and_whole_identifier():
    assert tokenize("calcularJurosComposto") == ["calcularjuroscomposto", "calcular", "juros", "composto"]
    assert tokenize("max_retry_count") == ["maxretrycount", "max", "retry", "count"]
    assert "http" in tokenize("HTTPServer") and "server" in tokenize("HTTPServer")
    enc = JavaSparseEncoder()
    assert enc.encode_documents(["calcular calcular juros"])[0]["calcular"] == 2
    assert enc.encode_query("validarCpf")["validarcpf"] == 1.0
    assert enc.encode_query("validarCpf") == enc.encode_query("validarCpf")


def test_embedding_cache_roundtrip_and_partial_shards(tmp_path):
    profile = EmbeddingProfile("m", "d", dim=4)
    cache = EmbeddingCache(tmp_path, profile)
    k1, k2, k3, k9 = (hashlib.sha256(str(i).encode()).hexdigest() for i in (1, 2, 3, 9))
    assert cache.lookup([k1]) == {}
    cache.add({k1: np.array([1, 0, 0, 0]), k2: np.array([0, 2, 0, 0])})
    cache.add({k3: np.array([0, 0, 3, 0])})
    got = EmbeddingCache(tmp_path, profile).lookup([k1, k2, k3, k9])
    assert set(got) == {k1, k2, k3} and got[k2].tolist() == [0, 2, 0, 0]

    # shard sem o arquivo de chaves (escrita interrompida) é ignorado
    (tmp_path / profile.profile_id / "shard-dead.vecs.npy").write_bytes(b"junk")
    assert set(EmbeddingCache(tmp_path, profile).lookup([k1])) == {k1}
    # outro perfil não enxerga os vetores
    other = EmbeddingProfile("m2", "d", dim=4)
    assert EmbeddingCache(tmp_path, other).lookup([k1]) == {}


def test_embedding_cache_refuses_null_vectors(tmp_path):
    cache = EmbeddingCache(tmp_path, EmbeddingProfile("m", "d", dim=4))
    with pytest.raises(CodeSearchError) as exc:
        cache.add({hashlib.sha256(b"1").hexdigest(): np.zeros(4)})
    assert exc.value.code == "embedder_invalid_output"
