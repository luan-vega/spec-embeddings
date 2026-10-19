"""Chunking de código Java: um chunk por método/construtor, janelas quando não cabe.

O scanner mascara comentários e literais, acompanha chaves e reconhece tipos (class, interface,
enum, record) e seus membros. Não é um parser Java completo: é o suficiente para delimitar símbolos.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

CHUNKER_VERSION = "1"
MAX_LINES = 80
MAX_CHARS = 6000
OVERLAP = 10

_CONTROL = {"if", "for", "while", "switch", "catch", "synchronized", "try", "do", "else", "return", "new", "super", "this"}
_TYPE_RE = re.compile(r"(?<![\w$.])(class|interface|enum|record)\s+([A-Za-z_$][\w$]*)")
_ANNOT_RE = re.compile(r"@(?!interface\b)[\w.$]+")
_IDENT_END_RE = re.compile(r"([A-Za-z_$][\w$]*)$")
_DOC_PREFIXES = ("/*", "*", "//", "@")


@dataclass(frozen=True)
class ChunkDraft:
    source: str
    symbol: str
    parent_symbol: str
    class_name: str
    start_line: int
    end_line: int
    text: str

    def embed_text(self, repo_id: str, description: str = "") -> str:
        """Texto embedado: cabeçalho com repo, caminho, classe, símbolo e linhas (+ resumo opcional) + código."""
        summary = f"summary: {description}\n" if description else ""
        return (
            f"repo: {repo_id}\npath: {self.source}\nclass: {self.class_name}\n"
            f"symbol: {self.symbol}\nlines: {self.start_line}-{self.end_line}\n{summary}\n{self.text}"
        )

    def embed_key(self, repo_id: str, describer_id: str = "") -> str:
        """Chave do cache de embeddings. Ignora números de linha: código que só mudou de
        posição reaproveita o vetor. Com descritor, a chave muda junto com ele."""
        parts = (repo_id, self.source, self.class_name, self.symbol, self.text)
        raw = "\x1f".join(parts + (describer_id,) if describer_id else parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class _Symbol:
    kind: str  # type | method
    name: str
    path: tuple[str, ...]  # tipos envolventes (inclui o próprio se for tipo)
    start: int  # offset no texto
    end: int  # offset do fecho (inclusive)


def mask_java(src: str) -> str:
    """Troca comentários, strings e chars por espaços, preservando offsets e quebras de linha."""
    out = list(src)
    n = len(src)

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < n:
        c = src[i]
        if c == "/" and src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif c == "/" and src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif src.startswith('"""', i):
            j = i + 3
            while j < n and not src.startswith('"""', j):
                j += 2 if src[j] == "\\" else 1
            blank(i, j + 3)
            i = j + 3
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _match_braces(masked: str) -> dict[int, int]:
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for i, c in enumerate(masked):
        if c == "{":
            stack.append(i)
        elif c == "}" and stack:
            pairs[stack.pop()] = i
    for i in stack:  # chaves sem fecho: vão até o fim do texto
        pairs[i] = len(masked) - 1
    return pairs


def _strip_annotations(head: str) -> str:
    out: list[str] = []
    pos = 0
    while True:
        m = _ANNOT_RE.search(head, pos)
        if not m:
            out.append(head[pos:])
            break
        out.append(head[pos : m.start()])
        j = m.end()
        k = j
        while k < len(head) and head[k].isspace():
            k += 1
        if k < len(head) and head[k] == "(":
            depth = 0
            while k < len(head):
                if head[k] == "(":
                    depth += 1
                elif head[k] == ")":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
            j = k
        pos = j
    return "".join(out)


def _has_top_level_assign(head: str) -> bool:
    depth = 0
    for i, c in enumerate(head):
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            depth = max(0, depth - 1)
        elif c == "=" and depth == 0:
            nxt = head[i + 1 : i + 2]
            if nxt != "=":
                return True
    return False


def _type_decl(head: str) -> tuple[str, str] | None:
    h = _strip_annotations(head)
    if _has_top_level_assign(h):
        return None
    m = _TYPE_RE.search(h)
    if m:
        return m.group(1), m.group(2)
    am = re.search(r"@interface\s+([A-Za-z_$][\w$]*)", head)
    return ("interface", am.group(1)) if am else None


def _method_name(head: str, cls: str) -> str | None:
    h = _strip_annotations(head).strip()
    if not h or _has_top_level_assign(h):
        return None
    idx = h.find("(")
    if idx < 0:  # construtor compacto de record: "public Foo {"
        m = _IDENT_END_RE.search(h)
        return m.group(1) if m and m.group(1) == cls else None
    m = _IDENT_END_RE.search(h[:idx].rstrip())
    if not m or m.group(1) in _CONTROL:
        return None
    before = h[: m.start()].rstrip()
    if before.endswith("new"):
        return None
    return m.group(1)


def _parse_symbols(masked: str) -> list[_Symbol]:
    pairs = _match_braces(masked)
    symbols: list[_Symbol] = []

    def first_code(a: int, b: int) -> int:
        k = a
        while k < b and masked[k].isspace():
            k += 1
        return k

    def walk(start: int, end: int, path: tuple[str, ...], parent_kind: str, cls: str) -> None:
        i = start
        mstart = start
        paren = 0
        constants = parent_kind == "enum"  # região de constantes antes do primeiro ';'
        while i < end:
            c = masked[i]
            if c == "(":
                paren += 1
            elif c == ")":
                paren = max(0, paren - 1)
            elif paren == 0 and c == ";":
                if constants:
                    constants = False
                elif parent_kind == "interface":
                    name = _method_name(masked[mstart:i], cls)
                    if name and "(" in masked[mstart:i]:
                        symbols.append(_Symbol("method", name, path, first_code(mstart, i), i))
                mstart = i + 1
            elif paren == 0 and c == "{":
                close = pairs.get(i, end)
                head = masked[mstart:i]
                decl = _type_decl(head)
                if decl is not None:
                    kind, name = decl
                    npath = path + (name,)
                    symbols.append(_Symbol("type", name, npath, first_code(mstart, i), close))
                    walk(i + 1, close, npath, kind, name)
                elif not constants:
                    name = _method_name(head, cls)
                    if name:
                        symbols.append(_Symbol("method", name, path, first_code(mstart, i), close))
                i = close
                mstart = close + 1
            i += 1

    walk(0, len(masked), (), "file", "")
    return symbols


def split_windows(lines: list[str], first: int, last: int) -> list[tuple[int, int]]:
    """Janelas 1-based [s, e] com no máximo MAX_LINES linhas e MAX_CHARS caracteres, com overlap."""
    out: list[tuple[int, int]] = []
    s = first
    while s <= last:
        e = s
        chars = len(lines[s - 1]) + 1
        while e < last and (e - s + 1) < MAX_LINES and chars + len(lines[e]) + 1 <= MAX_CHARS:
            e += 1
            chars += len(lines[e - 1]) + 1
        out.append((s, e))
        if e >= last:
            break
        s = max(s + 1, e - OVERLAP + 1)
    return out


def _with_doc_comment(lines: list[str], line: int) -> int:
    """Sobe a linha inicial para incluir Javadoc/comentários colados no símbolo."""
    start = line
    while start > 1 and line - start < 60:
        prev = lines[start - 2].strip()
        if prev and (prev.startswith(_DOC_PREFIXES) or prev.endswith("*/")):
            start -= 1
        else:
            break
    return start


def _pieces(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    return [text[i : i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]  # linha única gigante


def chunk_java(source: str, text: str) -> list[ChunkDraft]:
    """Quebra um arquivo Java em chunks por símbolo. ``source`` é o caminho relativo ao repo."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return []
    masked = mask_java(text)
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())

    def line_of(offset: int) -> int:
        return bisect.bisect_right(starts, offset)

    symbols = _parse_symbols(masked)
    stem = PurePosixPath(source).stem
    types = sorted((s for s in symbols if s.kind == "type"), key=lambda s: s.start)
    methods = sorted((s for s in symbols if s.kind == "method"), key=lambda s: s.start)

    def enclosing(line: int) -> tuple[str, ...]:
        best: tuple[str, ...] = ()
        for t in types:
            if line_of(t.start) <= line <= line_of(t.end) and len(t.path) >= len(best):
                best = t.path
        return best

    drafts: list[ChunkDraft] = []
    covered = [False] * (len(lines) + 2)

    def emit(first: int, last: int, symbol: str, parent: str, cls: str) -> None:
        for s, e in split_windows(lines, first, last):
            body = "\n".join(l.rstrip("\r") for l in lines[s - 1 : e])
            if not body.strip():
                continue
            for piece in _pieces(body):
                drafts.append(ChunkDraft(source, symbol, parent, cls, s, e, piece))

    for m in methods:
        first = _with_doc_comment(lines, line_of(m.start))
        last = min(line_of(m.end), len(lines))
        path = ".".join(m.path)
        cls = m.path[-1] if m.path else stem
        emit(first, last, ".".join((*m.path, m.name)) if m.path else m.name, path, cls)
        for ln in range(first, last + 1):
            covered[ln] = True

    ln = 1
    while ln <= len(lines):
        if covered[ln]:
            ln += 1
            continue
        end = ln
        while end + 1 <= len(lines) and not covered[end + 1]:
            end += 1
        meaningful = "".join(l.strip() for l in lines[ln - 1 : end] if l.strip() not in {"{", "}", "};", "});"})
        if len(meaningful) >= 12:
            first_code_line = next(k for k in range(ln, end + 1) if lines[k - 1].strip())
            last_code_line = next(k for k in range(end, ln - 1, -1) if lines[k - 1].strip())
            path = enclosing(first_code_line)
            symbol = ".".join(path) if path else stem
            emit(first_code_line, last_code_line, symbol, "", path[-1] if path else stem)
        ln = end + 1

    drafts.sort(key=lambda d: (d.start_line, d.end_line, d.symbol))
    return drafts
