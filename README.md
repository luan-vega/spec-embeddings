# code-search-embeddings

Busca local de código Java multi-repo com embeddings híbridos (denso via Ollama + lexical BM25, fusão RRF).
Python 3.12+, NumPy como única dependência, zero egress, uso monousuário em 127.0.0.1.

Desenho completo, revisão da spec e plano de fases: [docs/arquitetura.md](docs/arquitetura.md).

## Estado

| Fase | Conteúdo | Situação |
|---|---|---|
| 0 | Fundação: contratos, fakes, teste de zero egress, CI (Ubuntu + Windows) | feita |
| 1 | `FileVectorStore`: segmentos imutáveis, gerações atômicas, lock, rollback, prune | feita |
| 2 | Registro, política de exclusão, leitura do Git, chunker Java, cache de embeddings, indexador e busca mínima (RRF) | feita |
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

## Fase 2: como o índice é construído

1. `Registry` (`repositories.json`) lista os repos habilitados, com `ref=HEAD` e roots explícitas (padrão `src/main/java`).
2. `GitSource` lê a árvore do commit (`git ls-tree` + `git cat-file --batch`), nunca o working tree. Symlinks e submódulos ficam de fora.
3. `ExclusionPolicy` decide pelo caminho, por componentes, **antes** de abrir o arquivo. Política ausente ou inválida significa que nada é indexado.
4. `chunk_java` gera um chunk por método ou construtor (janelas de até 80 linhas / 6000 caracteres, overlap 10) e um chunk por região restante, com cabeçalho repo/caminho/classe/símbolo/linhas no texto embedado.
5. O filtro de segredos descarta chunks suspeitos (heurística, não um scanner).
6. `EmbeddingCache` guarda vetores por perfil e conteúdo; reindexar código inalterado não chama o modelo.
7. `IndexService` grava um segmento por repo e publica uma geração. Se o embedding falhar, a geração ativa continua a mesma.
8. `SearchService` (mínimo): valida, resolve repos, confere o perfil, embeda a consulta, funde os rankings denso e lexical por RRF e colapsa chunks sobrepostos.
