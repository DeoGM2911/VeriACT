"""The model's two notes files, `GUIDE.md` and `WORKING.md`, stored in the run directory.

Writes replace a whole file. Notes are not included in the turn prompt; the model reads them with
`read_notes` when it wants them (the one exception: `loop.py` copies them into the digest message
at a context cut). When a level ends, `WORKING.md` is archived to `working/level_<n>.md` for
offline analysis.

Argument validation is `tools.py`'s job; this module trusts its caller and only does the I/O.
"""
import os
import tempfile
from pathlib import Path

NOTES_FILES = ("GUIDE.md", "WORKING.md")


def _atomic_write_text(path: Path, content: str) -> None:
    """write_text() is neither atomic nor fsync'd: a crash mid-write can leave a truncated or
    corrupt file, and even a clean write isn't guaranteed to survive a crash immediately after
    (the data can still be sitting in an OS page cache buffer). Write to a temp file in the same
    directory (so the final os.replace is same-filesystem, hence atomic on POSIX and Windows both),
    fsync it, then replace -- the target either ends up with the old content or the new content in
    full, never something in between."""
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class NotesStore:
    def __init__(self, run_dir):
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._working_archive_dir = self._run_dir / "working"

    def read(self, file: str) -> str:
        path = self._run_dir / file
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write(self, file: str, content: str) -> None:
        _atomic_write_text(self._run_dir / file, content)

    def archive_working(self, level: int) -> None:
        """Snapshot WORKING.md as it stood when `level` just ended, for offline analysis. The
        model cannot read these snapshots."""
        working = self._run_dir / "WORKING.md"
        if not working.exists():
            return
        self._working_archive_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self._working_archive_dir / f"level_{level}.md", working.read_text(encoding="utf-8")
        )
