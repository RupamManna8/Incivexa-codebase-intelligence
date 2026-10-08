import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from workspace root or current directory
root_env = Path(__file__).resolve().parent.parent.parent / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

# Service Settings
SERVICE_PORT = int(os.getenv("INVESTIGATION_PORT", "8001"))
SERVICE_HOST = os.getenv("INVESTIGATION_HOST", "0.0.0.0")

# MongoDB Settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/incident_manager")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "incident_manager")

# AICredits API
AICREDITS_BASE_URL = os.getenv("AICREDITS_BASE_URL", "https://api.aicredits.in/v1").rstrip("/")
AICREDITS_API_KEY = os.getenv("AICREDITS_API_KEY", "")
AICREDITS_EMBEDDING_MODEL = os.getenv("AICREDITS_EMBEDDING_MODEL", "baai/bge-m3")
AICREDITS_AGENT_MODEL = os.getenv("AICREDITS_AGENT_MODEL", "qwen/qwen3-coder-30b-a3b-instruct")

# Repo Cache Directory
CACHE_DIR = Path(__file__).resolve().parent.parent / ".repo_cache"
INDEX_DIR = Path(__file__).resolve().parent.parent / ".index_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
