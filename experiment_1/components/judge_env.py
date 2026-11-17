"""Load judge configuration without exporting or logging secrets."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_judge_env(path=None):
    path = Path(path) if path is not None else ROOT / ".env"
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install .env support: pip install 'python-dotenv>=1.0,<2'") from exc
    # Shell environment wins over the local file; disable variable substitution in secrets.
    load_dotenv(path, override=False, interpolate=False, encoding="utf-8-sig")


def env_value(name):
    return os.environ.get(name, "").strip() or None
