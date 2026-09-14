"""
cedge_core/logging_config.py

Platform logging setup. One entry point, called once at process start:

    from cedge_core.logging_config import configure_logging
    configure_logging()

After that, every module:

    import logging
    logger = logging.getLogger(__name__)

Records land in <log_dir>/<module>_<date>.log. 

A single dispatching handler sits on the root logger and picks the
destination file per record, which gives per-module files AND a
single attached handler at the same time.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Sequence, Tuple

DEFAULT_LOG_DIR = "/home/research/work/cedge/logs"
DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] [%(processName)s] %(message)s"

_configured_roots: Dict[str, list[logging.Handler]] = {}

class PerModuleDailyFileHandler(logging.Handler):
    """Routes each record to <log_dir>/<module>_<YYYY-MM-DD>.log.

    The module is the last dotted segment of the logger name, so a logger
    named "cedge_core.risk.var_es_model" writes to var_es_model_<date>.log.

    The date comes from record.created, not from "now", so a long-running
    process rolls over at midnight on its own — no restart, and no
    TimedRotatingFileHandler needed. One child handler is kept per module and
    replaced (with the old one closed) when its date changes, which bounds
    open file descriptors to the number of modules that have logged.

    Thread safety: logging.Handler.handle() acquires this handler's lock
    before calling emit(), so the cache below is only mutated under that lock.
    """

    def __init__(self, log_dir: str, max_bytes: int = 10_000_000, backup_count: int = 5):
        super().__init__()
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        # module -> (date_str, handler)
        self._handlers: Dict[str, Tuple[str, RotatingFileHandler]] = {}
        os.makedirs(log_dir, exist_ok=True)

    def _handler_for(self, module: str, day: str) -> RotatingFileHandler:
        entry = self._handlers.get(module)
        if entry is not None and entry[0] == day:
            return entry[1]
        if entry is not None:
            entry[1].close()          # date changed — release the old file
        handler = RotatingFileHandler(
            os.path.join(self.log_dir, f"{module}_{day}.log"),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(self.formatter)
        self._handlers[module] = (day, handler)
        return handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            module = record.name.rsplit(".", 1)[-1]
            day = _dt.date.fromtimestamp(record.created).isoformat()
            self._handler_for(module, day).emit(record)
        except Exception: # noqa: BLE001 — logging must never crash the app it's logging for
            self.handleError(record)

    def setFormatter(self, fmt) -> None:
        super().setFormatter(fmt)
        for _day, handler in self._handlers.values():
            handler.setFormatter(fmt)

    def close(self) -> None:
        for _day, handler in self._handlers.values():
            handler.close()
        self._handlers.clear()
        super().close()


def configure_logging(
    log_dir: str = DEFAULT_LOG_DIR,
    level: Optional[str] = None,
    root_names: Sequence[str] = ("cedge_core",),
    console: bool = True,
) -> None:
    """Attach handlers once, at process start (cron entry point, Flask app
    factory, Streamlit startup).

    Calling this again for an already-configured root is a no-op, so an import
    cycle or a Streamlit re-run cannot stack handlers.

    root_names: logger names to attach to. Records propagate up the dotted
    hierarchy, so "cedge_core" catches everything under cedge_core.*. A
    service or app whose own modules are not under that prefix passes its own
    name too, e.g. root_names=("cedge_core", "portfolio_performance").

    level: defaults to the LOG_LEVEL environment variable, then INFO.
    """
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    formatter = logging.Formatter(DEFAULT_FORMAT)

    for name in root_names:
        if name in _configured_roots:
            continue

        logger = logging.getLogger(name)
        logger.setLevel(level)

        dispatcher = PerModuleDailyFileHandler(log_dir)
        dispatcher.setFormatter(formatter)
        logger.addHandler(dispatcher)
        attached: list[logging.Handler] = [dispatcher]

        if console:
            stream = logging.StreamHandler()
            stream.setFormatter(formatter)
            logger.addHandler(stream)
            attached.append(stream)

        _configured_roots[name] = attached


def reset_logging() -> None:
    """Detach and close everything configure_logging attached.

    Exists so a test can start from a clean slate, and so a long-lived process
    can reconfigure (e.g. change log_dir) without leaking handlers.
    """
    for name, handlers in _configured_roots.items():
        logger = logging.getLogger(name)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
    _configured_roots.clear()