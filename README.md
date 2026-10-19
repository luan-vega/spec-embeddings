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
| 3 | Busca com limites, CLI, HTTP local, MCP somente leitura, logs estruturados, harness de avaliação | feita |
| 4 | Embedder Ollama, `ollama-probe`; gate UC1/UC2/UC3 | embedder e probe feitos; dataset e rodada do gate pendentes |
| 5 | Operação: reindex incremental por repo, `rollback`, `prune`, `status` com geração anterior, falhas sem vazamento | feita |

## Desenvolvimento

```
python -m pip install -e ".[dev]"
python -m pytest
```

## Arquitetura em camadas (hexagonal)

```
domain/        regras puras: tipos, erros, chunker Java, tokenização, RRF, política de exclusão
application/   casos de uso (IndexService, SearchService, avaliação, limites) e as portas em ports.py
adapters/
  driven/      o que o núcleo usa: armazenamento em arquivos NumPy, Git, Ollama, cache, registro, política, logs
  driving/     quem chama o núcleo: CLI, HTTP local e MCP
bootstrap.py   raiz de composição: escolhe os adapters por configuração e injeta
```

`domain` não depende de nada; `application` depende só de `domain`; só `bootstrap` conhece os adapters concretos
(`tests/test_architecture.py` verifica). Para trocar o Ollama, o modelo ou o armazenamento, veja a seção 8 de
[docs/arquitetura.md](docs/arquitetura.md): `CODE_SEARCH_EMBEDDER` e `CODE_SEARCH_STORE` escolhem o adapter, e todo
adapter novo precisa passar nos testes de contrato em `tests/contract`. Adapters que saem da máquina ficam desligados
a menos que `CODE_SEARCH_ALLOW_REMOTE=1`.

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

## Fase 3: uso

```
code-search repo add svc-pagamento C:\\src\\svc-pagamento     # roots padrão: src/main/java
code-search policy init                                         # grava a política de exclusão padrão
code-search index                                               # reindexa os repos habilitados
code-search search "onde valida o CPF?" --mode hybrid --top-k 5
code-search status
code-search serve                                               # HTTP em 127.0.0.1:8765
code-search mcp                                                 # MCP por stdio, somente leitura
code-search eval --dataset eval/datasets/uc.jsonl --thresholds eval/thresholds.json --out eval/report.json
```

- **HTTP:** `GET /health`, `GET /ready`, `POST /v1/search`. Só escuta em loopback, exige `Content-Type: application/json`
  e confere o `Host`. Erros: `{"error": {"code", "message", "request_id"}}`. Excesso de carga devolve 503 com `Retry-After`.
- **Limites:** tamanho da consulta, `top_k`, número de repos, corpo da requisição e tarefas em execução (sem fila infinita).
- **Logs:** uma linha JSON por busca, com `request_id`, duração por etapa, modo, perfil, geração e contagens. Só entram
  números e strings curtas de ids; consulta, código, vetores e caminhos nunca aparecem.
- **Embedder:** ver a seção da fase 4. `CODE_SEARCH_DEV_FAKE_EMBEDDER=1` continua ligando um embedder falso
  só para desenvolvimento e testes.
- **Gate:** `eval/thresholds.json` guarda os limiares iniciais. O relatório registra o SHA-256 do arquivo usado.
  Os valores são pontos de partida para você aprovar antes da primeira rodada com o modelo real.

## Fase 4: embedder Ollama

O modelo precisa estar instalado antes no Ollama local (`ollama pull <modelo>`); o serviço nunca baixa modelos.

```
export CODE_SEARCH_OLLAMA_MODEL=nomic-embed-text      # obrigatória (ou qwen3-embedding:0.6b)
export CODE_SEARCH_OLLAMA_URL=http://127.0.0.1:11434  # opcional; só aceita http em loopback
export CODE_SEARCH_DOC_PREFIX="" CODE_SEARCH_QUERY_PREFIX=""   # opcionais, entram no perfil
code-search ollama-probe                              # modelo, digest, dimensão, contexto e profile_id
```

- O perfil (`profile_id`) vem do modelo, do digest, da dimensão e dos prefixos; trocar qualquer um exige reindexar.
- Chamadas usam `truncate: false`: entrada maior que o contexto do modelo falha com `embedder_invalid_output`, sem truncar em silêncio.
- Vetor zerado, não finito ou de dimensão errada também falha com `embedder_invalid_output`; os vetores são normalizados no cliente.
- Ollama fora do ar ou modelo ausente: `embedder_unavailable`.
- Dataset e rodada do gate: ver [eval/README.md](eval/README.md). O dataset depende dos repositórios reais e ainda não existe.

## Fase 5: operação

```
code-search index                    # só reindexa repos cujo commit, roots, perfil ou schema mudaram
code-search index --repo svc-a       # um repo; os demais seguem na geração atual
code-search index --force            # reindexa mesmo sem mudança (o cache evita chamar o modelo)
code-search status                   # geração ativa e anterior, repos, revisões
code-search rollback                 # volta para a geração anterior (a atual vira a anterior)
code-search prune                    # apaga gerações e segmentos que ninguém referencia
```

- Se nada mudou, `index` não publica geração nova e informa qual continua ativa.
- Falha de embedding ou de leitura do Git nunca troca a geração ativa.
- O `prune` mantém a ativa e a anterior; no Windows, segmento ainda aberto por um leitor fica para a próxima rodada.
- Erro inesperado na CLI mostra só `erro [internal_error]`, sem traceback, caminhos ou trechos de código.
