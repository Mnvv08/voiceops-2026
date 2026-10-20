"""Central configuration. All secrets come from environment variables (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.getenv("DATASET_DIR", ROOT / "Capstone_Project_Dataset_FTTT_2026")).expanduser()
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"
TOOLS_DIR = DATASET_DIR / "tools"
CACHE_DIR = ROOT / ".cache"

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

GNANI_TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_VOICE = os.getenv("GNANI_VOICE", "Kaveri")
GNANI_LANG = os.getenv("GNANI_LANG", "en-IN")

# The brief fixes "today" for every date rule.
TODAY = "20 October 2026"

TOP_K = 5
