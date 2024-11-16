import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    "venv",
    "env",
    ".vscode",
    ".idea",
    "dist",
    "build",
    "__mocks__",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    "logs",
    "temp",
    "tmp",
    "secrets",
    "private",
    "cache",
    "addons",
}
MAX_ADDED_FILES_SIZE = 100000  # ~100KB
