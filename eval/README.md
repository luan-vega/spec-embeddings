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
