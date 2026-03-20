from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import TextIO, Tuple


LOGS_DIR = Path("Temps/Logs")


def create_session_log_file() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return LOGS_DIR / f"{timestamp}_log.txt"


def redirect_process_output_to_log() -> Tuple[Path, TextIO]:
    log_path = create_session_log_file()
    handle = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = handle
    sys.stderr = handle
    return log_path, handle
