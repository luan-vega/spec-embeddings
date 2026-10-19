from pathlib import Path

from workflows.apps.code_search.adapters.driven.policy_store import write_default_policy
from workflows.apps.code_search.bootstrap import build_runtime
from workflows.apps.code_search.adapters.driven.fakes import BagOfWordsEmbedder

from gitutil import make_repo

PAGAMENTO = '''package br.pagamento;

public class Pagamento {
    /** Calcula os juros compostos de um valor para um prazo em meses. */
    public double calcularJurosComposto(double valor, int meses) {
        return valor * Math.pow(1.02, meses);
    }

    public void estornar(String contaId) { }
}
'''
CLIENTE = '''package br.cliente;

public class Cliente {
    /** Valida o documento CPF do cliente. */
    public boolean validarCpf(String documento) {
        return documento != null && documento.length() == 11;
    }
}
'''


def make_indexed_runtime(tmp_path: Path, *, index: bool = True, **kwargs):
    home = tmp_path / "home"
    repo_a = make_repo(tmp_path / "repos" / "svc-pagamento", {"src/main/java/br/pagamento/Pagamento.java": PAGAMENTO})
    repo_b = make_repo(tmp_path / "repos" / "svc-cliente", {"src/main/java/br/cliente/Cliente.java": CLIENTE})
    write_default_policy(home / "exclusion-policy.json")
    rt = build_runtime(home, embedder=kwargs.pop("embedder", BagOfWordsEmbedder()), **kwargs)
    rt.registry.add("svc-pagamento", repo_a)
    rt.registry.add("svc-cliente", repo_b)
    if index:
        rt.indexer.index()
    return rt
