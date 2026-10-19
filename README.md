# code-search-embeddings

Busca local de código Java multi-repo com embeddings híbridos (denso via Ollama + lexical BM25, fusão RRF).
Python 3.12+, NumPy como única dependência, zero egress, uso monousuário em 127.0.0.1.

Desenho completo, revisão da spec e plano de fases: [docs/arquitetura.md](docs/arquitetura.md).

## Estado

| Fase | Conteúdo | Situação |
|---|---|---|
| 0 | Fundação: contratos, fakes, teste de zero egress, CI (Ubuntu + Windows) | feita |
| 1 | `FileVectorStore`: segmentos imutáveis, gerações atômicas, lock, rollback, prune | feita |
| 2 | Registro, política de exclusão, leitura do Git, chunker Java, cache de embeddings | pendente |
| 3 | Busca (RRF), CLI, HTTP, MCP, harness de avaliação | pendente |
| 4 | Embedder Ollama, `ollama-probe`, gate UC1/UC2/UC3 | pendente |
| 5 | Operação | pendente |

## Desenvolvimento

```
python -m pip install -e ".[dev]"
python -m pytest
```

## Layout em disco (fase 1)

Um segmento por repo, imutável (`segments/<repo>-<hash>/`), com `dense.npy` (float32, L2-normalizado, aberto com memmap),
índice invertido em CSR para o BM25, metadados em colunas e `chunks.jsonl` com offsets. Uma geração
(`generations/<id>.json`) lista os segmentos; `active.json` aponta para ela e é trocado com `os.replace`.
Os arquivos de um segmento ficam na raiz da pasta (sem as subpastas `meta/` e `lexical/` do desenho inicial).
