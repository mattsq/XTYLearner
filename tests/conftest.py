import sys
from pathlib import Path

# Ensure project root is on sys.path for test discovery when Hydra sets CWD.
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
