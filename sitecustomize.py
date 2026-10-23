"""Select this immutable disk release before the original /app package."""
from pathlib import Path
import sys

release = str(Path(__file__).resolve().parent)
if release in sys.path:
    sys.path.remove(release)
sys.path.insert(0, release)

# Pin package __path__ before python -m prepends its working directory.
import observer
import wallex_engine
