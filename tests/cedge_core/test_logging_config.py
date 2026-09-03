import datetime as dt
import logging
import time

import pytest

from cedge_core.logging_config import (
    PerModuleDailyFileHandler,
    configure_logging,
    reset_logging,
)


@pytest.fixture
def clean_logging():
    reset_logging()
    yield
    reset_logging()


def _make_record(name: str, msg: str, created: float) -> logging.LogRecord:
    """A LogRecord with a chosen timestamp.

    Lets the date-rollover test cross midnight without freezing the clock:
    the handler reads record.created, so a record simply carries its own date.
    """
    record = logging.LogRecord(
        name=name, level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    record.created = created
    return record


# --------------------------------------------------------------------------
# the defect this module was written to remove
# --------------------------------------------------------------------------

def test_configure_logging_is_idempotent(clean_logging, tmp_path):
    """The bug in ch_logging.setup_logger: a second call stacked a handler.

    getLogger(name) is a registry lookup, so the second call was handed the
    SAME logger that already had a handler, and added another one.
    """
    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core",), console=False)
    first = list(logging.getLogger("cedge_core").handlers)
    assert len(first) == 1

    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core",), console=False)
    second = list(logging.getLogger("cedge_core").handlers)

    assert len(second) == 1
    assert second[0] is first[0], "second call replaced the handler instead of skipping"


def test_one_record_writes_exactly_one_line(clean_logging, tmp_path):
    """The observable symptom: stacked handlers duplicated every line.

    Paired with the handler-count assertion above on purpose — that one names
    the mechanism, this one names what a reader of the log actually sees.
    """
    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core",), console=False)
    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core",), console=False)

    logging.getLogger("cedge_core.demo").info("one event")

    written = list(tmp_path.glob("demo_*.log"))
    assert len(written) == 1
    assert written[0].read_text().count("one event") == 1


# --------------------------------------------------------------------------
# routing: one attached handler, many destination files
# --------------------------------------------------------------------------

def test_each_module_gets_its_own_file(clean_logging, tmp_path):
    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core",), console=False)

    logging.getLogger("cedge_core.risk.var_es_model").info("from var")
    logging.getLogger("cedge_core.portfolio.performance").info("from perf")

    today = dt.date.today().isoformat()
    var_log = tmp_path / f"var_es_model_{today}.log"
    perf_log = tmp_path / f"performance_{today}.log"

    assert var_log.exists() and perf_log.exists()
    assert "from var" in var_log.read_text()
    assert "from var" not in perf_log.read_text()


def test_only_one_handler_serves_all_modules(clean_logging, tmp_path):
    """Per-module files WITHOUT per-module handlers — the whole point.

    Attaching a handler per module is the arrangement that made the old code
    fragile, so the invariant is asserted directly rather than implied.
    """
    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core",), console=False)
    for module in ("alpha", "beta", "gamma"):
        logging.getLogger(f"cedge_core.{module}").info("hi")

    assert len(logging.getLogger("cedge_core").handlers) == 1
    assert len(list(tmp_path.glob("*.log"))) == 3


# --------------------------------------------------------------------------
# date rollover — no clock mocking required
# --------------------------------------------------------------------------

def test_records_on_different_days_go_to_different_files(clean_logging, tmp_path):
    handler = PerModuleDailyFileHandler(str(tmp_path))
    handler.setFormatter(logging.Formatter("%(message)s"))

    monday = dt.datetime(2026, 9, 1, 23, 59, 0).timestamp()
    tuesday = dt.datetime(2026, 9, 2, 0, 0, 1).timestamp()

    handler.handle(_make_record("cedge_core.demo", "before midnight", monday))
    handler.handle(_make_record("cedge_core.demo", "after midnight", tuesday))

    assert (tmp_path / "demo_2026-09-01.log").read_text().strip() == "before midnight"
    assert (tmp_path / "demo_2026-09-02.log").read_text().strip() == "after midnight"


def test_rollover_closes_the_previous_days_file(clean_logging, tmp_path):
    """Bounds open file descriptors to the module count, not to uptime.

    Keying the cache on (module, date) instead would accumulate one open file
    per module per day — a leak that only appears after weeks, so no test that
    merely runs the code would ever catch it. Asserted structurally instead.
    """
    handler = PerModuleDailyFileHandler(str(tmp_path))
    handler.setFormatter(logging.Formatter("%(message)s"))

    monday = dt.datetime(2026, 9, 1, 12, 0, 0).timestamp()
    handler.handle(_make_record("cedge_core.demo", "day one", monday))
    stale = handler._handlers["demo"][1]
    assert stale.stream is not None

    tuesday = dt.datetime(2026, 9, 2, 12, 0, 0).timestamp()
    handler.handle(_make_record("cedge_core.demo", "day two", tuesday))

    assert stale.stream is None, "previous day's file was never closed"
    assert len(handler._handlers) == 1, "cache grew instead of evicting"


# --------------------------------------------------------------------------
# handler contract
# --------------------------------------------------------------------------

def test_formatter_reaches_children_created_later(clean_logging, tmp_path):
    """setFormatter must propagate BOTH ways.

    Children created after setFormatter pick it up in _handler_for; children
    created before it must be updated by the setFormatter override. The second
    direction is the one an inherited setFormatter would silently miss.
    """
    handler = PerModuleDailyFileHandler(str(tmp_path))
    handler.setFormatter(logging.Formatter("first:%(message)s"))
    day = dt.datetime(2026, 9, 1, 12, 0, 0).timestamp()
    handler.handle(_make_record("cedge_core.demo", "a", day))

    handler.setFormatter(logging.Formatter("second:%(message)s"))
    handler.handle(_make_record("cedge_core.demo", "b", day))

    body = (tmp_path / "demo_2026-09-01.log").read_text()
    assert "first:a" in body
    assert "second:b" in body


def test_emit_failure_does_not_reach_the_caller(clean_logging, tmp_path):
    """Logging must never be able to kill a running calculation."""
    handler = PerModuleDailyFileHandler(str(tmp_path))

    class Exploding(logging.Formatter):
        def format(self, record):
            raise OSError("No space left on device")

    handler.setFormatter(Exploding())
    logging.raiseExceptions = False
    try:
        handler.handle(_make_record("cedge_core.demo", "x", time.time()))
    finally:
        logging.raiseExceptions = True


def test_close_deregisters_from_the_logging_module(clean_logging, tmp_path):
    """Handler.close() must reach the base class.

    Its docstring requires it: the base removes the handler from the module
    -level _handlers name map. Omitting super().close() leaves a stale entry.
    """
    handler = PerModuleDailyFileHandler(str(tmp_path))
    handler.set_name("cedge-dispatcher")
    assert logging._handlers.get("cedge-dispatcher") is handler

    handler.close()

    assert "cedge-dispatcher" not in logging._handlers


# --------------------------------------------------------------------------
# scoping and teardown
# --------------------------------------------------------------------------

def test_third_party_loggers_are_untouched(clean_logging, tmp_path):
    """Configuring the real root would create sqlalchemy_*.log, urllib3_*.log …

    Scoping to the cedge_core prefix is what keeps the log directory readable.
    Deliberately relies on the DEFAULT root_names, so widening that default
    breaks this test.
    """
    configure_logging(log_dir=str(tmp_path), console=False)   # default root_names
    logging.getLogger("sqlalchemy.engine.Engine").warning("SELECT 1")

    assert not list(tmp_path.glob("Engine_*.log"))


def test_additional_root_names_are_configured(clean_logging, tmp_path):
    """services/ and apps/ modules are not under the cedge_core prefix."""
    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core", "portfolio_performance"),
                      console=False)
    logging.getLogger("portfolio_performance.app").info("service up")

    today = dt.date.today().isoformat()
    assert (tmp_path / f"app_{today}.log").exists()


def test_reset_logging_detaches_and_closes(clean_logging, tmp_path):
    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core",), console=False)
    dispatcher = logging.getLogger("cedge_core").handlers[0]

    reset_logging()

    assert logging.getLogger("cedge_core").handlers == []
    assert dispatcher._handlers == {}


def test_configure_after_reset_works_again(clean_logging, tmp_path):
    """reset must clear the guard, or a process could never reconfigure."""
    configure_logging(log_dir=str(tmp_path), root_names=("cedge_core",), console=False)
    reset_logging()

    second_dir = tmp_path / "second"
    configure_logging(log_dir=str(second_dir), root_names=("cedge_core",), console=False)
    logging.getLogger("cedge_core.demo").info("after reset")

    assert len(logging.getLogger("cedge_core").handlers) == 1
    assert list(second_dir.glob("demo_*.log"))
