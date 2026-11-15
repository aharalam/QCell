import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum_kernel import load_env_file  # noqa: E402

# Credentials for the Quantum Rings tests come from the git-ignored .env at the repo root.
load_env_file(Path(__file__).resolve().parents[2] / ".env")
