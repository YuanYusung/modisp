"""Keep third-party JIT caches inside the test workspace."""
import os
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).parent/'.numba-cache'))
