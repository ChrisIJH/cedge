"""
apps/cedge_mcp/client.py

HTTP client for the cedge services, used by the MCP server.

This module must never import cedge_core,
open a database connection, or issue SQL. Every number arrives over HTTP.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

REQUEST_TIMEOUT = float(os.getenv("CEDGE_MCP_TIMEOUT", "120"))

SERVICE_URLS = {
    "portfolio_performance": os.getenv(
        "CEDGE_PORTFOLIO_PERFORMANCE_API_URL", "http://localhost:8000"),
    "risk": os.getenv("CEDGE_RISK_API_URL", "http://localhost:8001"),
    "marketdata": os.getenv("CEDGE_MARKETDATA_API_URL", "http://localhost:8002"),
    "optimization": os.getenv("CEDGE_OPTIMIZATION_API_URL", "http://localhost:8003"),
    "regime": os.getenv("CEDGE_REGIME_API_URL", "http://localhost:8004"),
}



class ServiceError(RuntimeError):
    """A cedge service refused the request or could not be reached.
    
    status is None when the service was unreachable."""

    def __init__(self, service: str, status: Optional[int], message: str):
        self.service = service
        self.status = status
        self.message = message
        where = f"HTTP {status}" if status is not None else "unreachable"
        super().__init__(f"{service} service ({where}): {message}")



class CedgeClient:
    """Thin GET/POST wrapper. Inject `session` in tests to avoid real HTTP."""

    def __init__(self, session: Optional[Any] = None,
                 urls: Optional[Dict[str, str]] = None):
        self._session = session if session is not None else requests.Session()
        self._urls = dict(urls) if urls is not None else dict(SERVICE_URLS)

    def base_url(self, service: str) -> str:
        try:
            return self._urls[service]
        except KeyError:
            raise ValueError(f"Unknown service {service!r}")

    def get(self, service: str, path: str, **params: Any) -> Dict[str, Any]:
        url = f"{self.base_url(service)}{path}"
        # requests drops None-valued params, which matches the services'
        # convention that an absent query parameter means NULL.
        try:
            resp = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            raise ServiceError(service, None, f"cannot reach {url} ({exc.__class__.__name__})")
        return self._parse(service, resp)

    def post(self, service: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url(service)}{path}"
        try:
            resp = self._session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            raise ServiceError(service, None, f"cannot reach {url} ({exc.__class__.__name__})")
        return self._parse(service, resp)

    @staticmethod
    def _parse(service: str, resp: Any) -> Dict[str, Any]:
        if 200 <= resp.status_code < 300:
            return resp.json()
        # The services return {"error": "..."} on 400/404.
        try:
            message = resp.json().get("error", "") or resp.text[:300]
        except ValueError:
            message = resp.text[:300]
        raise ServiceError(service, resp.status_code, message)