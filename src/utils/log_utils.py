from __future__ import annotations

from pathlib import Path


def log_line(message: str, path: Path | str | None) -> None:
    """Print and prepend the message to a log file if provided."""
    print(message)
    if path is None:
        return
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            existing = log_path.read_text(encoding="utf-8").splitlines()
        else:
            existing = []
        lines = [message] + existing
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        return
