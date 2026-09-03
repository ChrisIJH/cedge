# Cedge

Quant platform: factor models, risk analytics, portfolio construction,
and decision analysis.

## Structure

| Path | Role |
|------|------|
| `core/cedge_core/` | Quant engine — pure calculation + data access. No HTTP, no UI. |
| `services/` | HTTP API layer. Wraps `core` functions as deployable services. |
| `apps/` | User-facing applications (Streamlit dashboards). |
| `tests/` | Test suite, mirroring the `core/` structure. |

## Layer rule

Dependencies flow one way: `apps` → `services` → `core`.
`core` never imports from `services` or `apps`, and never imports HTTP or
UI frameworks.

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Design decisions

Architecture and migration notes live in `docs/`.

