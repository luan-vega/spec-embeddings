import json

import pytest

from workflows.apps.code_search.core import CodeSearchError
from workflows.apps.code_search.policy import DEFAULT_POLICY, ExclusionPolicy, write_default_policy
from workflows.apps.code_search.registry import Registry

from gitutil import make_repo


@pytest.fixture
def policy():
    p = ExclusionPolicy.from_dict(DEFAULT_POLICY)
    assert p.valid
    return p


def test_missing_policy_indexes_nothing(tmp_path):
    p = ExclusionPolicy.load(tmp_path / "nope.json")
    assert not p.valid and p.reason == "policy_missing"
    assert not p.allows("src/main/java/A.java")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("exclude_dirs"),
        lambda d: d.update(extra=1),
        lambda d: d.update(version=""),
        lambda d: d.update(include_extensions=["java"]),
        lambda d: d.update(exclude_dirs=["a/b"]),
        lambda d: d.update(max_file_bytes=0),
        lambda d: d.update(secret_patterns=["("]),
        lambda d: d.update(exclude_file_globs="*.pem"),
    ],
)
def test_invalid_policy_indexes_nothing(tmp_path, mutate):
    data = json.loads(json.dumps(DEFAULT_POLICY))
    mutate(data)
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data))
    p = ExclusionPolicy.load(path)
    assert not p.valid and p.reason == "policy_invalid"
    assert not p.allows("src/main/java/A.java")
    path.write_text("{not json")
    assert ExclusionPolicy.load(path).reason == "policy_invalid"


def test_default_policy_roundtrip_and_version(tmp_path):
    write_default_policy(tmp_path / "p.json")
    p = ExclusionPolicy.load(tmp_path / "p.json")
    assert p.valid and p.version_id.startswith("1-")
    changed = json.loads(json.dumps(DEFAULT_POLICY))
    changed["exclude_dirs"].append("legacy")
    assert ExclusionPolicy.from_dict(changed).version_id != p.version_id


@pytest.mark.parametrize(
    "path,allowed",
    [
        ("src/main/java/a/B.java", True),
        ("src/main/java/a/B.JAVA", True),
        ("src/main/java/a/B.kt", False),
        (".git/hooks/X.java", False),
        ("src/main/java/node_modules/x/Y.java", False),
        ("src/main/java/build/Y.java", False),
        # componentes, não prefixos de string: "builder" e "targets" não são "build" e "target"
        ("src/main/java/builder/Y.java", True),
        ("src/main/java/targets/Y.java", True),
        ("src/main/java/.env.java", False),
        ("src/main/java/.envrc.java", False),
        ("src/main/java/Server.pem", False),
        ("src/main/java/a/id_rsa.java", False),
        ("../outside/A.java", False),
        ("/abs/A.java", False),
        ("", False),
    ],
)
def test_policy_allows_by_components(policy, path, allowed):
    assert policy.allows(path) is allowed


def test_secret_heuristics(policy):
    assert policy.contains_secret('String password = "hunter2hunter2";')
    assert policy.contains_secret("-----BEGIN RSA PRIVATE KEY-----\nabc")
    assert policy.contains_secret('key = "AKIAABCDEFGHIJKLMNOP"')
    assert policy.contains_secret("ghp_" + "a" * 36)
    assert not policy.contains_secret("String passwordField = request.getParameter(name);")
    assert not policy.contains_secret("public double calcularJurosComposto(double v) { return v; }")


def test_registry_roundtrip(tmp_path):
    repo = make_repo(tmp_path / "svc-a", {"src/main/java/A.java": "class A {}"})
    reg = Registry(tmp_path / "home" / "repositories.json")
    assert reg.list() == []
    entry = reg.add("svc-a", repo, roots=["src\\main\\java", "src/extra/./java"])
    assert entry.roots == ("src/main/java", "src/extra/java") and entry.enabled
    assert [e.repo_id for e in reg.list()] == ["svc-a"]
    reg.set_enabled("svc-a", False)
    assert reg.enabled() == [] and reg.get("svc-a").enabled is False
    assert reg.remove("svc-a") is True and reg.remove("svc-a") is False
    assert reg.list() == []


def test_registry_rejects_bad_input(tmp_path):
    repo = make_repo(tmp_path / "svc-a", {"A.java": "class A {}"})
    plain = tmp_path / "plain"
    plain.mkdir()
    reg = Registry(tmp_path / "r.json")
    for bad_id in ("../x", "a/b", "", ".x"):
        with pytest.raises(CodeSearchError):
            reg.add(bad_id, repo)
    for bad_root in ("/etc", "C:/x", "../up", "a/../../b", "."):
        with pytest.raises(CodeSearchError):
            reg.add("ok", repo, roots=[bad_root])
    with pytest.raises(CodeSearchError):
        reg.add("ok", plain)
    reg.add("ok", repo)
    with pytest.raises(CodeSearchError):
        reg.add("ok", repo)
    (tmp_path / "r.json").write_text("{broken")
    with pytest.raises(CodeSearchError):
        reg.list()
