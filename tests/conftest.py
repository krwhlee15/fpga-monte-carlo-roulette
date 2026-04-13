import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Ensure tests can import the project package when run from the repo root.
    sys.path.insert(0, str(ROOT))
