"""
apps/cedge_mcp/server.py

Cedge MCP server: exposes read-only cedge analytics as MCP tools.
Built on the MCP Python SDK v2 (MCPServer; FastMCP in v1). Requires Python >= 3.10.

Run (from the repo root):
    python -m apps.cedge_mcp.server                                       # stdio
    python -m apps.cedge_mcp.server --transport streamable-http --port 8010

LAYER RULE: apps/ never imports cedge_core or touches the database.
In stdio mode never print() to stdout; stdout is the protocol channel.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import tools
from .client import CedgeClient, ServiceError

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("cedge.mcp")

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                            idempotent_hint=True, open_world_hint=False)

server = MCPServer(
    name="cedge",
    instructions=(
        "Read-only access to Cedge quant analytics (regime, portfolios, VaR/ES backtests). "
        "Every result includes a 'provenance' block; cite it when quoting numbers. "
        "Always pass explicit dates. Do not infer trading recommendations from these tools."
    ),
)

_client = CedgeClient()


def _run(fn, *args, **kwargs) -> Dict[str, Any]:
    """Turn service and input errors into MCP tool errors the agent can read."""
    try:
        return fn(_client, *args, **kwargs)
    except (ServiceError, ValueError) as exc:
        log.warning("tool error: %s", exc)
        raise ToolError(str(exc)) from exc


@server.tool(annotations=READ_ONLY)
def cedge_get_regime(as_of: str) -> Dict[str, Any]:
    """Macro regime for one date, computed from prices up to as_of (YYYY-MM-DD):
    score, posterior probabilities, policy (gross/beta caps), and the input z-scores."""
    return _run(tools.get_regime, as_of)


@server.tool(annotations=READ_ONLY)
def cedge_get_regime_history(start: str, end: str, score_name: str = "macro_v1") -> Dict[str, Any]:
    """Daily regime scores between start and end (YYYY-MM-DD), as written by the
    production job. Returns at most the last 260 rows."""
    return _run(tools.get_regime_history, start, end, score_name)


@server.tool(annotations=READ_ONLY)
def cedge_list_portfolios() -> Dict[str, Any]:
    """List available books (pf_name, experiment_id, branch, source, date coverage).
    Use these keys when calling cedge_var_backtest_summary."""
    return _run(tools.list_portfolios)


@server.tool(annotations=READ_ONLY)
def cedge_var_backtest_summary(
    pf_name: str,
    as_of_date: str,
    start_date: str,
    end_date: str,
    experiment_id: Optional[str] = None,
    branch: Optional[str] = None,
    source: Optional[str] = None,
    window: int = 252,
    levels: Optional[List[float]] = None,
    methods: Optional[List[str]] = None,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """VaR/ES backtest for a book: weights in force on as_of_date, fixed-weight returns
    over [start_date, end_date], rolling window, levels as tail probabilities
    (0.05 = 95% VaR), methods in {parametric, historic}. Returns Kupiec, Christoffersen,
    and Acerbi-Szekely results per cell, plus caveats. Can take tens of seconds."""
    return _run(tools.var_backtest_summary, pf_name, as_of_date, start_date, end_date,
                experiment_id=experiment_id, branch=branch, source=source, window=window,
                levels=levels, methods=methods, n_boot=n_boot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cedge MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()