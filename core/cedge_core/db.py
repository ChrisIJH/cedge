"""
cedge_core/db.py

Database engine access. Singleton SQLAlchemy engine with connection pooling.

Credentials are resolved in this order:
    1. Environment variables (CEDGE_DB_USER / CEDGE_DB_PASSWORD) — preferred.
     This is the path a container, CI runner, or a secrets manager
     (Azure Key Vault, AWS Secrets Manager) injects into.
    2. A YAML config file (CEDGE_CONFIG_PATH) — local development fallback.



"""

import os

import yaml
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

DEFAULT_CONFIG_PATH = "/home/research/work/conf/config.yaml"
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_NAME = "securities_master"

_engine_instance = None



def _resolve_credentials() -> tuple:
    """
    Returns (user, password, host, database).

    Environment variables win. Falling back to the YAML file is a local-dev
    convenience, not the intended production path.
    """
    user = os.environ.get("CEDGE_DB_USER")
    password = os.environ.get("CEDGE_DB_PASSWORD")

    if not (user and password):
        config_path = os.environ.get("CEDGE_CONFIG_PATH", DEFAULT_CONFIG_PATH)
        if not os.path.isfile(config_path):
            raise RuntimeError(
                "No DB credentials found. Set CEDGE_DB_USER and "
                f"CEDGE_DB_PASSWORD, or provide a config file at {config_path} "
                "(override the path with CEDGE_CONFIG_PATH)."
            )
        with open(config_path) as f:
            data_map = yaml.safe_load(f)
        user = data_map["securities_master"]["id"]
        password = data_map["securities_master"]["pw"]

    host = os.environ.get("CEDGE_DB_HOST", DEFAULT_DB_HOST)
    database = os.environ.get("CEDGE_DB_NAME", DEFAULT_DB_NAME)
    return user, password, host, database


def ch_engine():
    """
    Returns a singleton SQLAlchemy engine with connection pooling.

    - pool_size: 5 (default connections kept open)
    - max_overflow: 10 (additional connections when pool is exhausted)
    - pool_recycle: 3600 (recycle after 1 hour to avoid MySQL timeout)
    - pool_pre_ping: True (check connection validity before use)
    """
    global _engine_instance

    if _engine_instance is None:
        user, password, host, database = _resolve_credentials()
        _engine_instance = create_engine(
            f"mysql+pymysql://{user}:{password}@{host}/{database}",
            echo=False,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            pool_pre_ping=True,
        )

    return _engine_instance


def ch_engine_dispose():
    """Dispose all pooled connections (call when done with batch operations)."""
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.dispose()
        _engine_instance = None
    
