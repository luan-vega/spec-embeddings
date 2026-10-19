import pytest

from workflows.apps.code_search.core import CodeSearchError
from workflows.apps.code_search.git_source import GitSource

from gitutil import add_symlink_entry, commit_all, git, make_repo


def test_lists_only_tracked_regular_files_under_roots(tmp_path):
    repo = make_repo(
        tmp_path / "svc",
        {
            "src/main/java/a/Pagamento.java": "class Pagamento {}",
            "src/main/java/a/leia me.java": "class Espaco {}",
            "src/main/java/a/Ação.java": "class Acao {}",
            "src/main/resources/app.properties": "k=v",
            "src/test/java/a/PagamentoTest.java": "class T {}",
            "src/main/javaextra/Other.java": "class O {}",
            "README.md": "x",
        },
    )
    add_symlink_entry(repo, "src/main/java/a/Link.java", "../../../../../etc/passwd")
    git(repo, "commit", "-q", "-m", "link")
    (repo / "src/main/java/a/Untracked.java").write_text("class U {}")

    src = GitSource(repo)
    paths = [b.path for b in src.list_blobs(["src/main/java"])]
    assert paths == ["src/main/java/a/Ação.java", "src/main/java/a/Pagamento.java", "src/main/java/a/leia me.java"]


def test_reads_head_not_working_tree(tmp_path):
    repo = make_repo(tmp_path / "svc", {"src/main/java/A.java": "class A { /* v1 */ }"})
    rev = GitSource(repo).revision()
    (repo / "src/main/java/A.java").write_text("class A { /* sujo */ }")
    src = GitSource(repo)
    blobs = src.list_blobs(["src/main/java"])
    assert src.revision() == rev
    contents = dict(src.read_blobs([b.sha for b in blobs]))
    assert contents[blobs[0].sha] == b"class A { /* v1 */ }"

    new_rev = commit_all(repo)
    assert GitSource(repo).revision() == new_rev != rev
    blobs = GitSource(repo).list_blobs(["src/main/java"])
    assert dict(GitSource(repo).read_blobs([blobs[0].sha]))[blobs[0].sha] == b"class A { /* sujo */ }"


def test_missing_root_is_empty_and_bad_repo_is_unavailable(tmp_path):
    repo = make_repo(tmp_path / "svc", {"src/main/java/A.java": "class A {}"})
    assert GitSource(repo).list_blobs(["nao/existe"]) == []
    with pytest.raises(CodeSearchError) as exc:
        GitSource(tmp_path / "nao-e-repo").revision()
    assert exc.value.code == "repo_unavailable"
    assert str(tmp_path) not in exc.value.message


def test_read_many_blobs_in_batches(tmp_path):
    files = {f"src/main/java/F{i}.java": f"class F{i} {{ int v = {i}; }}" for i in range(450)}
    repo = make_repo(tmp_path / "svc", files)
    src = GitSource(repo)
    blobs = src.list_blobs(["src/main/java"])
    got = dict(src.read_blobs([b.sha for b in blobs]))
    assert len(got) == 450
    assert got[next(b.sha for b in blobs if b.path.endswith("F449.java"))] == b"class F449 { int v = 449; }"
