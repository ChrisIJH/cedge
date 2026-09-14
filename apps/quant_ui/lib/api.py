"""
apps/quant_ui/lib/api.py

HTTP client for the cedge services.

LAYER RULE — this module, and everything under apps/, must never import
cedge_core, open a database connection, or issue SQL.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
import streamlit as st

REQUEST_TIMEOUT = 30

SERVICE_URLS = {
    "portfolio_performance": os.getenv(
        "CEDGE_PORTFOLIO_PERFORMANCE_API_URL", "http://localhost:8000"),
    "risk": os.getenv(
        "CEDGE_RISK_API_URL", "http://localhost:8001"),
    "marketdata": os.getenv(
        "CEDGE_MARKETDATA_API_URL", "http://localhost:8002"),
}

def _base_url(service: str) -> str:
    try:
        return SERVICE_URLS[service]
    except KeyError:
        raise ValueError(f"Unknown service {service!r}.")

def _handle_error_and_stop(service: str, exc: Optional[Exception],
                           response: Optional[requests.Response])-> None:
    """Render one consistent error banner and halt this page's execution.""" 
    if response is not None and response.status_code == 400:
        st.error(f"**{service}** rejected the request:"
                  "{response.json().get('error', 'not found')}")
    elif response is not None and response.status_code == 404:
        st.warning(f"**{service}**: {response.json().get('error', 'not found')}")
    else:
        st.error(
            f"Cannot reach the **{service}** service at `{_base_url(service)}`.\n\n"
            f"Start it with:\n\n"
            f"    docker run --rm -p {_base_url(service).rsplit(':', 1)[-1]}:8000 cedge-{service.replace('_', '-')}\n\n"
            f"({exc.__class__.__name__ if exc else 'HTTP ' + str(response.status_code)})"
        )
    st.stop()

def get(service: str, path: str, **params: Any) -> Dict:
    """GET {service}{path} and return the parsed JSON body, or halt the page."""
    url = f"{_base_url(service)}{path}"
    try:
        response =  requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        _handle_error_and_stop(service, exc, None)
        return {}  # unreachable — st.stop() raises, but this satisfies type checkers

    if not response.ok:
        _handle_error_and_stop(service, None, response)
        return {}

    return response.json()


def post(service: str, path: str, payload: Dict) -> Dict:
    """POST payload to {service}{path} and return the parsed JSON body, or halt the page."""
    url = f"{_base_url(service)}{path}"
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        _handle_error_and_stop(service, exc, None)
        return {}

    if not response.ok:
        _handle_error_and_stop(service, None, response)
        return {}

    return response.json()


def healthy(service: str) -> bool:
    """Non-halting check — used for a status indicator, not a page gate."""
    try:
        response = requests.get(f"{_base_url(service)}/healthz", timeout=5)
        return response.ok
    except requests.exceptions.RequestException:
        return False 