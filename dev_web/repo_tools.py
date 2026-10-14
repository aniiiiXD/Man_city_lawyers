"""Narrow local repository tools with path scoping and an argv check allowlist."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable

from .paths import PathSafetyError, PathScope, SecureRoot, is_sensitive_path


_GENERATED_DIRS = {
    ".git", ".dev_web", ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
}
_GENERATED_FILES = {".ds_store", "thumbs.db"}


class RepoToolError(RuntimeError):
    """Raised when a repository operation is rejected or fails safely."""


@dataclass(frozen=True, slots=True)
class CheckSpec:
    check_id: str
    argv: tuple[str, ...]
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.check_id):
            raise ValueError(f"Invalid check id: {self.check_id!r}")
        if not self.argv or any(not part or "\x00" in part for part in self.argv):
            raise ValueError("A check must have a non-empty argv vector")
        if self.timeout_seconds < 1 or self.timeout_seconds > 1800:
            raise ValueError("Check timeout must be between 1 and 1800 seconds")

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> CheckSpec:
        if not isinstance(raw, dict):
            raise ValueError("Each check entry must be an object")
        argv = raw.get("argv")
        if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
            raise ValueError("Check argv must be a JSON array of strings")
        check_id = raw.get("check_id")
        if not isinstance(check_id, str):
            raise ValueError("Check entries need a string check_id")
        return cls(
            check_id=check_id, argv=tuple(argv),
            timeout_seconds=int(raw.get("timeout_seconds", 300)),
        )


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    passed: bool
    return_code: int | None
    stdout: str
    stderr: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    before_sha256: str | None
    after_sha256: str | None
    content: str | None
    is_binary: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> FileChange:
        return cls(**raw)  # type: ignore[arg-type]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RepoToolError(f"Could not read {path}: {exc}") from exc


class CheckRunner:
    """Run only named, pre-registered commands using argument vectors."""

    def __init__(self, specs: Iterable[CheckSpec] = ()) -> None:
        values = tuple(specs)
        ids = [spec.check_id for spec in values]
        if len(set(ids)) != len(ids):
            raise ValueError("Check ids must be unique")
        self.specs = {spec.check_id: spec for spec in values}

    @classmethod
    def from_config(cls, path: str | Path) -> CheckRunner:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config root must be an object")
        checks = config.get("checks", [])
        if not isinstance(checks, list):
            raise ValueError("config checks must be an array")
        return cls(CheckSpec.from_dict(item) for item in checks)

    @staticmethod
    def _clean_environment() -> dict[str, str]:
        # Keep only execution essentials; API keys and application credentials
        # from the invoking agent process are not forwarded to repository code.
        kept = ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "TMPDIR", "LANG", "LC_ALL")
        env = {name: os.environ[name] for name in kept if name in os.environ}
        env["PYTHONNOUSERSITE"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    def run(self, check_id: str, cwd: str | Path) -> CheckResult:
        spec = self.specs.get(check_id)
        if spec is None:
            raise RepoToolError(
                f"Check {check_id!r} is not allowlisted. Register a fixed argv in .dev_web/config.json."
            )
        working_dir = Path(cwd).resolve(strict=True)
        if not working_dir.is_dir():
            raise RepoToolError("Check working directory must be a directory")
        for current, dirs, files in os.walk(working_dir, topdown=True, followlinks=False):
            current_path = Path(current)
            relative_dir = current_path.relative_to(working_dir)
            for name in (*dirs, *files):
                candidate = current_path / name
                relative = (relative_dir / name).as_posix()
                if candidate.is_symlink():
                    raise RepoToolError(f"Refusing to run checks with a workspace symlink: {relative}")
                if is_sensitive_path(relative):
                    raise RepoToolError(f"Refusing to run checks with a sensitive workspace path: {relative}")
        try:
            completed = subprocess.run(
                list(spec.argv), cwd=working_dir, env=self._clean_environment(),
                capture_output=True, text=True, timeout=spec.timeout_seconds,
                check=False, shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CheckResult(
                check_id, False, None,
                (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                f"Check timed out after {spec.timeout_seconds}s",
            )
        except OSError as exc:
            return CheckResult(check_id, False, None, "", "", f"Could not start check: {exc}")
        return CheckResult(
            check_id, completed.returncode == 0, completed.returncode,
            completed.stdout[-20000:], completed.stderr[-20000:],
        )


class RepoTools:
    """Read/search/patch/diff API constrained to one task's declared files."""

    def __init__(self, root: str | Path, scope: PathScope, checks: CheckRunner | None = None,
                 baseline: str | Path | None = None) -> None:
        self.secure_root = SecureRoot(root)
        self.root = self.secure_root.root
        self.scope = scope
        self.checks = checks or CheckRunner()
        self.baseline = Path(baseline).resolve(strict=True) if baseline is not None else None

    def _resolve(self, relative: str, *, exists: bool = False) -> Path:
        try:
            self.scope.require(relative)
            return self.secure_root.resolve(relative, must_exist=exists)
        except PathSafetyError as exc:
            raise RepoToolError(str(exc)) from exc

    def read_file(self, relative: str, *, max_bytes: int = 1_000_000) -> str:
        path = self._resolve(relative, exists=True)
        if not path.is_file():
            raise RepoToolError(f"Not a regular file: {relative}")
        data = _read_bytes(path)
        if len(data) > max_bytes:
            raise RepoToolError(f"File exceeds read limit ({max_bytes} bytes): {relative}")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepoToolError(f"File is not UTF-8 text: {relative}") from exc

    def search(self, query: str, paths: Iterable[str] | None = None, *, regex: bool = False,
               max_results: int = 200) -> list[dict[str, object]]:
        if not query:
            raise RepoToolError("Search query must not be empty")
        matcher = re.compile(query) if regex else None
        candidates = list(paths) if paths is not None else self._scope_files()
        hits: list[dict[str, object]] = []
        for relative in candidates:
            path = self._resolve(relative, exists=True)
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            data = _read_bytes(path)
            if b"\x00" in data:
                continue
            try:
                lines = data.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, start=1):
                found = bool(matcher.search(line)) if matcher else query in line
                if found:
                    hits.append({"path": relative, "line": number, "text": line[:1000]})
                    if len(hits) >= max_results:
                        return hits
        return hits

    def patch_file(self, relative: str, new_content: str, *, expected_sha256: str | None) -> str:
        """Replace one scoped file only if its current content matches the read hash."""
        path = self._resolve(relative)
        if path.exists():
            if not path.is_file():
                raise RepoToolError(f"Not a regular file: {relative}")
            current_hash = sha256_bytes(_read_bytes(path))
        else:
            current_hash = None
        if current_hash != expected_sha256:
            raise RepoToolError(f"Stale patch rejected for {relative}; reread it and retry")
        self.secure_root.ensure_parent(relative)
        # Re-resolve after directory creation, then atomically replace.
        path = self._resolve(relative)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{path.name}.devweb-",
                dir=path.parent, delete=False,
            ) as stream:
                stream.write(new_content)
                temporary = Path(stream.name)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return sha256_bytes(new_content.encode("utf-8"))

    def diff(self, paths: Iterable[str] | None = None) -> str:
        if self.baseline is None:
            raise RepoToolError("diff needs a baseline snapshot for comparison")
        changes = collect_file_changes(self.baseline, self.root, self.scope)
        if paths is not None:
            selected = set(paths)
            for relative in selected:
                self.scope.require(relative)
            changes = tuple(change for change in changes if change.path in selected)
        return render_changes(self.baseline, changes)

    def run_check(self, check_id: str) -> CheckResult:
        return self.checks.run(check_id, self.root)

    def _scope_files(self) -> list[str]:
        result: list[str] = []
        for current, dirs, files in os.walk(self.root, topdown=True, followlinks=False):
            current_path = Path(current)
            rel_dir = current_path.relative_to(self.root)
            dirs[:] = sorted(
                name for name in dirs
                if not (current_path / name).is_symlink()
                and not is_sensitive_path((rel_dir / name).as_posix())
            )
            for name in sorted(files):
                path = current_path / name
                relative = (rel_dir / name).as_posix()
                if path.is_symlink() or is_sensitive_path(relative):
                    continue
                if self.scope.allows(relative):
                    result.append(relative)
        return result


def collect_file_changes(baseline: str | Path, workspace: str | Path,
                         scope: PathScope) -> tuple[FileChange, ...]:
    """Create safe file-level proposals by comparing isolated tree snapshots."""
    base_root = SecureRoot(baseline)
    work_root = SecureRoot(workspace)

    def indexed(root: SecureRoot, *, is_workspace: bool) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        for current, dirs, files in os.walk(root.root, topdown=True, followlinks=False):
            current_path = Path(current)
            rel_dir = current_path.relative_to(root.root)
            for name in dirs:
                path = current_path / name
                relative = (rel_dir / name).as_posix()
                if path.is_symlink():
                    raise RepoToolError(f"Symlink found in task workspace: {relative}")
                if is_workspace and is_sensitive_path(relative):
                    raise RepoToolError(f"Sensitive path found in task workspace: {relative}")
            dirs[:] = sorted(name for name in dirs if name not in _GENERATED_DIRS)
            for name in files:
                path = current_path / name
                relative = (rel_dir / name).as_posix()
                if name.lower() in _GENERATED_FILES or name.endswith(".pyc"):
                    continue
                if path.is_symlink():
                    raise RepoToolError(f"Symlink found in task workspace: {relative}")
                if is_sensitive_path(relative):
                    if is_workspace:
                        raise RepoToolError(f"Sensitive path found in task workspace: {relative}")
                    continue
                if path.is_file():
                    if path.stat().st_size > 10 * 1024 * 1024:
                        raise RepoToolError(f"File exceeds snapshot limit: {relative}")
                    found[relative] = path.read_bytes()
        return found

    before = indexed(base_root, is_workspace=False)
    after = indexed(work_root, is_workspace=True)
    changes: list[FileChange] = []
    for relative in sorted(before.keys() | after.keys()):
        old, new = before.get(relative), after.get(relative)
        if old == new:
            continue
        if not scope.allows(relative):
            raise RepoToolError(f"Out-of-scope file changed in task workspace: {relative}")
        binary = bool((old or b"").count(b"\x00") or (new or b"").count(b"\x00"))
        content: str | None = None
        if new is not None and not binary:
            try:
                content = new.decode("utf-8")
            except UnicodeDecodeError:
                binary = True
        changes.append(FileChange(
            path=relative,
            before_sha256=sha256_bytes(old) if old is not None else None,
            after_sha256=sha256_bytes(new) if new is not None else None,
            content=content,
            is_binary=binary,
        ))
    return tuple(changes)


def render_changes(baseline: str | Path, changes: Iterable[FileChange]) -> str:
    """Render a human-reviewable unified diff for a proposal."""
    root = SecureRoot(baseline)
    pieces: list[str] = []
    for change in changes:
        if change.is_binary:
            pieces.append(f"Binary change: {change.path}\n")
            continue
        old_path = root.resolve(change.path)
        old_text = old_path.read_text(encoding="utf-8").splitlines(keepends=True) if old_path.exists() else []
        new_text = (change.content or "").splitlines(keepends=True) if change.after_sha256 is not None else []
        pieces.extend(difflib.unified_diff(
            old_text, new_text,
            fromfile=f"a/{change.path}" if change.before_sha256 is not None else "/dev/null",
            tofile=f"b/{change.path}" if change.after_sha256 is not None else "/dev/null",
        ))
    return "".join(pieces)
