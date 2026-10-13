"""
scripts/smoke_stdio.py

End-to-end check over the real MCP protocol: launch the server over stdio,
list tools, call each tool, and confirm a bad date returns is_error.

    python scripts/smoke_stdio.py --fake                                    # fake services
    python scripts/smoke_stdio.py --as-of 2026-09-30 --pf <pf_name>         # real services
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _text(result) -> str:
    return result.content[0].text if result.content else ""


async def main(args) -> int:
    env = dict(os.environ)
    if args.fake:
        env.update({"CEDGE_REGIME_API_URL": "http://127.0.0.1:18004",
                    "CEDGE_MARKETDATA_API_URL": "http://127.0.0.1:18002",
                    "CEDGE_RISK_API_URL": "http://127.0.0.1:18001"})
    params = StdioServerParameters(command=sys.executable, args=["-m", "apps.cedge_mcp.server"],
                                   env=env, cwd=REPO_ROOT)
    failures = 0
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        print("tools:", sorted(t.name for t in (await session.list_tools()).tools))

        r = await session.call_tool("cedge_get_regime", {"as_of": args.as_of})
        print("\ncedge_get_regime ->", "ERROR" if r.is_error else "ok")
        print(_text(r)[:600])
        failures += int(r.is_error)

        bad = await session.call_tool("cedge_get_regime", {"as_of": "1990-01-01"})
        print("\nbad date -> is_error =", bad.is_error, "|", _text(bad)[:200])
        failures += int(not bad.is_error)

        r = await session.call_tool("cedge_list_portfolios", {})
        print("\ncedge_list_portfolios ->", "ERROR" if r.is_error else "ok")
        failures += int(r.is_error)

        r = await session.call_tool("cedge_var_backtest_summary", {
            "pf_name": args.pf, "as_of_date": args.as_of,
            "start_date": args.start, "end_date": args.as_of, "n_boot": 500})
        print("\ncedge_var_backtest_summary ->", "ERROR" if r.is_error else "ok")
        if r.is_error:
            print(_text(r)[:400])
        else:
            body = json.loads(_text(r))
            print(json.dumps(body["data"]["cells"][0], indent=2)[:800])
            print("provenance:", json.dumps(body["provenance"])[:400])
        failures += int(r.is_error)

    print("\nRESULT:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default="2026-09-30")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--pf", default="SPX_LS_FACTOR_V1")
    p.add_argument("--fake", action="store_true", help="use scripts/fake_services.py ports")
    a = p.parse_args()
    if a.fake:
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        import fake_services
        fake_services.serve_all(block=False)
    sys.exit(anyio.run(main, a))