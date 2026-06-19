"""Put code/ on sys.path so `import argus` / `import evaluation` work under pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
