from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
