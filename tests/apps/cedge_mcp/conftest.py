"""Shared fakes for cedge_mcp tests. pytest loads this automatically."""
from __future__ import annotations

import pytest

URLS = {"regime": "http://r", "marketdata": "http://m", "risk": "http://k",
        "portfolio_performance": "http://p", "optimization": "http://o"}


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


class FakeSession:
    """Routes (METHOD, URL) -> (status, body). Records calls."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _hit(self, method, url, **kw):
        self.calls.append((method, url, kw))
        status, body = self.routes[(method, url)]
        return FakeResp(status, body)

    def get(self, url, params=None, timeout=None):
        return self._hit("GET", url, params=params)

    def post(self, url, json=None, timeout=None):
        return self._hit("POST", url, json=json)


@pytest.fixture
def urls():
    return dict(URLS)


@pytest.fixture
def fake_session():
    """Usage: sess = fake_session({("GET", "http://r/api/regime"): (200, {...})})"""
    return FakeSession