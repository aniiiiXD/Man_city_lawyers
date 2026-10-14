from pathlib import Path

import pytest

from dev_web.paths import PathSafetyError, PathScope, SecureRoot
from dev_web.repo_tools import RepoToolError, RepoTools, sha256_bytes


def test_scope_and_secure_root_reject_traversal_and_secrets(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "ok.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".env").write_text("TOP_SECRET=x\n", encoding="utf-8")
    tools = RepoTools(root, PathScope(("src/",)))

    assert tools.read_file("src/ok.py") == "x = 1\n"
    with pytest.raises(RepoToolError, match="outside the task scope"):
        tools.read_file(".env")
    with pytest.raises(PathSafetyError, match="traversal"):
        SecureRoot(root).resolve("../outside")


def test_secure_root_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)

    with pytest.raises(PathSafetyError, match="Symlink"):
        SecureRoot(root).resolve("link.txt", must_exist=True)


def test_patch_requires_expected_hash_and_exact_scope(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "file.py").write_text("old\n", encoding="utf-8")
    tools = RepoTools(root, PathScope(("file.py",)))

    with pytest.raises(RepoToolError, match="Stale patch"):
        tools.patch_file("file.py", "new\n", expected_sha256="wrong")
    tools.patch_file("file.py", "new\n", expected_sha256=sha256_bytes(b"old\n"))
