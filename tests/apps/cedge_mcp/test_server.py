"""Server registration tests. Needs the MCP SDK (Python >= 3.10); skipped otherwise."""
from __future__ import annotations

import pytest

pytest.importorskip("mcp")

import anyio
from mcp.server.mcpserver.exceptions import ToolError

from apps.cedge_mcp import server as srv
from apps.cedge_mcp.client import CedgeClient


def test_server_lists_four_read_only_tools():
    listed = anyio.run(srv.server.list_tools)
    assert {t.name for t in listed} == {"cedge_get_regime", "cedge_get_regime_history",
                                        "cedge_list_portfolios", "cedge_var_backtest_summary"}
    assert all(t.annotations and t.annotations.read_only_hint for t in listed)


def test_service_error_becomes_tool_error(monkeypatch, fake_session, urls):
    sess = fake_session({("GET", "http://r/api/regime"): (404, {"error": "no data"})})
    monkeypatch.setattr(srv, "_client", CedgeClient(sess, urls))
    with pytest.raises(ToolError) as ei:
        anyio.run(srv.server.call_tool, "cedge_get_regime", {"as_of": "1990-01-01"})
    assert "no data" in str(ei.value)