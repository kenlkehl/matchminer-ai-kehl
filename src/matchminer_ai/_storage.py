"""Small atomic, content-addressed persistence primitives."""

import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def atomic_bytes(path: Path, data: bytes):
    """Save an original binary source without decoding or re-encoding it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def directory_lock(directory: Path):
    """Fail promptly when another process owns a run, releasing on process exit."""
    with (directory / ".run.lock").open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError(f"Another process is using {directory}") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"Another process is using {directory}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class QueryCache:
    """A folder of JSON values, one file per 64-hex-digit key, written atomically.

    A ``MutableMapping``-like cache for compiled queries (``compile_workup_queries``,
    ``compile_workup_sql`` and the criteria compilers): any key it cannot read is
    missing, so a damaged entry is compiled again.
    """

    def __init__(self, folder):
        self.folder = Path(folder)

    def _path(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", str(key)):
            raise KeyError(key)
        return self.folder / f"{key}.json"

    def get(self, key, default=None):
        try:
            return json.loads(self._path(key).read_text())
        except (OSError, ValueError, KeyError):
            return default

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        return self.get(key) is not None

    def __setitem__(self, key, value):
        path = self._path(key)
        self.folder.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".query-", dir=self.folder)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(value, handle, ensure_ascii=False)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
