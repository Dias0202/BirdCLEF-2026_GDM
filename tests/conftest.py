import sys
from pathlib import Path

# The training scripts in src/ use flat imports (``from metrics import ...``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
