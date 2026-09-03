"""
cedge_core/regime/scores.py

Regime score access and label inference.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd 
import numpy as np 

from sqlalchemy import text
from sqlalchemy.engine import engine

from cedge_core.db import ch_engine

