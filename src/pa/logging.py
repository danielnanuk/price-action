"""JSON-line structured logger keyed by pipeline stage and run_id."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class StageLogger:
    def __init__(self, *, stage: str, run_id: str, log_path: Path) -> None:
        self.stage = stage
        self.run_id = run_id
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = log_path.open("a", encoding="utf-8")

    def _emit(self, level: str, msg: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "stage": self.stage,
            "run_id": self.run_id,
            "msg": msg,
            **fields,
        }
        self._fp.write(json.dumps(record, default=str) + "\n")
        self._fp.flush()

    def info(self, msg: str, **fields: Any) -> None:
        self._emit("info", msg, **fields)

    def warn(self, msg: str, **fields: Any) -> None:
        self._emit("warn", msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit("error", msg, **fields)

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
