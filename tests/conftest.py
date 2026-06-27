"""Put the repo root on sys.path so the namespace packages `eb_jepa` and
`gray_scott` import the same way they do when run as `python -m gray_scott.main`
from the repo root (gray_scott is not pip-installed)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
