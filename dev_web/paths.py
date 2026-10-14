"""Filesystem boundaries for repository, workspaces, and task path scopes."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil


class PathSafetyError(ValueError):
    """Raised when a requested path leaves its root or enters a denied area."""


_DENIED_PARTS = {
    ".git", ".dev_web", ".ssh", ".aws", "aws", ".azure", "azure", ".kube", "kube",
    "gcloud", "credentials", "secrets", ".terraform",
}
_SKIP_DIRS = {
    ".git", ".dev_web", ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
}
_SKIP_FILES = {".ds_store", "thumbs.db"}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
_SECRET_FILES = {
    ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519", "known_hosts",
    "authorized_keys", "credentials.json", "service-account.json", "auth.json", ".dev.vars",
}


def _parts(relative: str | os.PathLike[str]) -> tuple[str, ...]:
    raw = os.fspath(relative)
    if not raw or "\x00" in raw or raw.startswith(("/", "\\")):
        raise PathSafetyError(f"Expected a non-empty relative path, got {raw!r}")
    normalized = raw.replace("\\", "/")
    parts = tuple(part for part in PurePosixPath(normalized).parts if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise PathSafetyError(f"Path traversal is not allowed: {raw!r}")
    return parts


def is_sensitive_path(relative: str | os.PathLike[str]) -> bool:
    """Return true for common secret stores and private-key files."""
    try:
        parts = _parts(relative)
    except PathSafetyError:
        return True
    for part in parts:
        lowered = part.lower()
        if lowered in _DENIED_PARTS or lowered.startswith(".env") or lowered in _SECRET_FILES:
            return True
        if Path(lowered).suffix in _SECRET_SUFFIXES:
            return True
        if ".tfstate" in lowered or "secret" in lowered or "credential" in lowered:
            return True
    return False


class PathScope:
    """Allow only exact files or explicitly declared directory prefixes."""

    def __init__(self, scopes: tuple[str, ...] | list[str]) -> None:
        if not scopes:
            raise PathSafetyError("At least one file scope is required")
        normalized: list[tuple[tuple[str, ...], bool]] = []
        for scope in scopes:
            directory = scope.endswith(("/", "\\"))
            raw = scope[:-1] if directory else scope
            pieces = _parts(raw)
            if is_sensitive_path("/".join(pieces)):
                raise PathSafetyError(f"Sensitive paths cannot be included in a task scope: {scope!r}")
            normalized.append((pieces, directory))
        self._scopes = tuple(normalized)

    def allows(self, relative: str | os.PathLike[str]) -> bool:
        try:
            pieces = _parts(relative)
        except PathSafetyError:
            return False
        if is_sensitive_path("/".join(pieces)):
            return False
        for scope, is_directory in self._scopes:
            if pieces == scope or (is_directory and pieces[: len(scope)] == scope):
                return True
        return False

    def require(self, relative: str | os.PathLike[str]) -> tuple[str, ...]:
        pieces = _parts(relative)
        if not self.allows("/".join(pieces)):
            raise PathSafetyError(f"Path is outside the task scope: {relative!s}")
        return pieces


class SecureRoot:
    """Resolve paths under a root while rejecting traversal and symlink hops."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise PathSafetyError(f"Root is not a directory: {self.root}")

    def resolve(self, relative: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
        pieces = _parts(relative)
        if is_sensitive_path("/".join(pieces)):
            raise PathSafetyError(f"Sensitive path access is denied: {relative!s}")
        candidate = self.root
        for index, part in enumerate(pieces):
            candidate = candidate / part
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                if must_exist:
                    raise PathSafetyError(f"Path does not exist: {relative!s}") from None
                continue
            if candidate.is_symlink():
                raise PathSafetyError(f"Symlink path components are not allowed: {relative!s}")
            if index < len(pieces) - 1 and not candidate.is_dir():
                raise PathSafetyError(f"Parent component is not a directory: {relative!s}")
            if must_exist and index == len(pieces) - 1 and not info:
                raise PathSafetyError(f"Path does not exist: {relative!s}")
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PathSafetyError(f"Path escaped its root: {relative!s}") from None
        if must_exist and not candidate.exists():
            raise PathSafetyError(f"Path does not exist: {relative!s}")
        return candidate

    def ensure_parent(self, relative: str | os.PathLike[str]) -> Path:
        pieces = _parts(relative)
        parent = self.root.joinpath(*pieces[:-1])
        current = self.root
        for part in pieces[:-1]:
            current = current / part
            try:
                current.mkdir()
            except FileExistsError:
                pass
            if current.is_symlink() or not current.is_dir():
                raise PathSafetyError(f"Unsafe parent path: {relative!s}")
        return parent


def copy_safe_tree(source: str | Path, destination: str | Path) -> None:
    """Copy a repository snapshot without VCS internals, symlinks, or secrets."""
    source_root = SecureRoot(source)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=False)
    for current, dirs, files in os.walk(source_root.root, topdown=True, followlinks=False):
        current_path = Path(current)
        rel_dir = current_path.relative_to(source_root.root)
        dirs[:] = sorted(
            name for name in dirs
            if name.lower() not in _SKIP_DIRS
            and not is_sensitive_path((rel_dir / name).as_posix())
            and not (current_path / name).is_symlink()
        )
        relative_dir = rel_dir.as_posix()
        target_dir = destination_path if relative_dir == "." else destination_path / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            source_file = current_path / name
            relative_file = (rel_dir / name).as_posix()
            if name.lower() in _SKIP_FILES or is_sensitive_path(relative_file) or source_file.is_symlink():
                continue
            try:
                if not source_file.is_file():
                    continue
                if source_file.stat().st_size > 10 * 1024 * 1024:
                    continue
                target = target_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, target)
            except OSError:
                # A changing or unreadable file should not prevent a safe task
                # workspace from being created; it will simply be absent.
                continue
