# Gate de avaliação (fase 4)

Dataset: JSONL, uma consulta por linha, ~30 por caso de uso (UC1, UC2, UC3):

```json
{"id": "uc2-001", "uc": "UC2", "query": "calcula juros compostos da parcela", "relevant": [{"repo": "svc-a", "source": "src/main/java/br/Pagamento.java", "symbol": "calcularJuros"}]}
```

- `repo` é o `repo_id` do registro; `source` é o caminho relativo no repositório; `symbol` é opcional.
- Os rótulos precisam vir dos repositórios reais de quem usa o serviço, por isso o dataset não está versionado aqui.
- Os limiares ficam em `thresholds.json` e valem desde o commit anterior à primeira rodada com o modelo real.

Rodada (modelo já baixado manualmente no Ollama):

```
export CODE_SEARCH_OLLAMA_MODEL=nomic-embed-text   # ou qwen3-embedding:0.6b
code-search ollama-probe
code-search index
code-search eval --help
```

Compare hybrid × semantic × lexical para cada modelo candidato; o relatório grava o SHA-256 dos limiares.

## Exemplo para exercitar o gate (sem Ollama)

`eval/example/` tem um corpus Java sintético (3 repositórios, 36 métodos) e um dataset de 108 consultas, 36 por caso de uso:

- **UC1** (conceito → código): descrição em linguagem natural, sem repetir o nome do método.
- **UC2** (identificador exato): o nome do método.
- **UC3** (código → código): o próprio método com nomes trocados.

```
python eval/example/build_example.py /tmp/ex           # cria /tmp/ex/repos/* (Git) e /tmp/ex/dataset.jsonl
export CODE_SEARCH_HOME=/tmp/ex/home CODE_SEARCH_DEV_FAKE_EMBEDDER=1
code-search policy init
for r in svc-pagamento svc-cliente svc-estoque; do code-search repo add $r /tmp/ex/repos/$r; done
code-search index
code-search eval --dataset eval/example/dataset.jsonl --thresholds eval/thresholds.json --out /tmp/ex/report.json
```

Com o embedder falso (bag of words) os números só provam que o harness funciona de ponta a ponta; não medem qualidade
semântica, e o gate pode reprovar os ganhos de UC1/UC3 sobre o lexical. Para a rodada que vale, troque o embedder falso
pelo Ollama (`CODE_SEARCH_OLLAMA_MODEL`, sem `CODE_SEARCH_DEV_FAKE_EMBEDDER`) e use o seu dataset.
`tests/test_example_gate.py` mantém `dataset.jsonl` igual ao que o gerador produz.
