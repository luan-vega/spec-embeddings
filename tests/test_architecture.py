"""Regras de dependência da arquitetura hexagonal."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "workflows" / "apps" / "code_search"
PKG = "workflows.apps.code_search"


def imported_modules(path: Path):
    """Módulos do pacote importados por ``path``, como nomes absolutos."""
    package = [*PKG.split("."), *path.relative_to(ROOT).parent.parts]
    mods = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package[: len(package) - (node.level - 1)]
                mods.append(".".join(anchor + ([node.module] if node.module else [])))
            else:
                mods.append(node.module or "")
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return [m for m in mods if m.startswith(PKG)]


def layer_files(layer: str):
    return sorted((ROOT / layer).rglob("*.py"))


FORBIDDEN = {
    "domain": ("application", "adapters", "bootstrap"),
    "application": ("adapters", "bootstrap"),
}


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_inner_layers_do_not_depend_on_outer_ones(layer):
    files = layer_files(layer)
    assert files
    bad = []
    for f in files:
        for mod in imported_modules(f):
            sub = mod[len(PKG) + 1:].split(".")[0]
            if sub in FORBIDDEN[layer]:
                bad.append(f"{f.relative_to(ROOT)} -> {mod}")
    assert not bad, bad


def test_domain_has_no_io_or_numpy():
    banned = {"numpy", "subprocess", "socket", "http", "urllib", "json"}
    for f in layer_files("domain"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
            for n in names:
                if n.split(".")[0] in banned and not (n == "json" and f.name == "model.py"):
                    pytest.fail(f"{f.relative_to(ROOT)} imports {n}")


def test_driving_adapters_do_not_reach_into_driven_ones_except_composition_helpers():
    allowed = {"policy_store", "obslog"}
    for f in layer_files("adapters/driving"):
        for mod in imported_modules(f):
            if ".adapters.driven." in mod + ".":
                assert mod.rsplit(".", 1)[-1] in allowed, f"{f.name} -> {mod}"
