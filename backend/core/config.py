"""
PromptShield - Backend Configuration
All settings come from environment variables (optionally loaded from a .env
file in the project root). Secrets are never given defaults.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class Settings:
    database_url: Optional[str]
    admin_api_key: Optional[str] = field(repr=False, default=None)
    cors_origins: List[str] = field(default_factory=list)
    max_prompt_length: int = 4096
    history_turns: int = 10

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Settings":
        # Absolute path: works no matter which directory uvicorn is started from.
        # Variables already set in the process environment take precedence over the file.
        load_dotenv(env_file or os.path.join(PROJECT_ROOT, ".env"), override=False)
        origins = os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
        )
        return cls(
            database_url=os.getenv("DATABASE_URL") or None,
            admin_api_key=os.getenv("ADMIN_API_KEY") or None,
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            max_prompt_length=int(os.getenv("MAX_PROMPT_LENGTH", "4096")),
            history_turns=int(os.getenv("CHAT_HISTORY_TURNS", "10")),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


def environment_warnings(project_root: str = PROJECT_ROOT) -> List[str]:
    """
    Common configuration mistakes, described without revealing any values:
    a missing .env, an unedited copy of .env.example, or settings saved under
    another file name (e.g. example.env) that the backend never reads.
    """
    env_file = os.path.join(project_root, ".env")
    template = os.path.join(project_root, ".env.example")
    warnings = []

    if not os.path.exists(env_file):
        warnings.append(f".env not found in the project root ({project_root}); copy .env.example to .env.")
    elif os.path.exists(template):
        with open(env_file, "rb") as a, open(template, "rb") as b:
            if a.read() == b.read():
                warnings.append(".env is an unedited copy of .env.example; its DATABASE_URL is a placeholder.")

    for name in sorted(os.listdir(project_root)):
        path = os.path.join(project_root, name)
        if name in (".env", ".env.example") or not os.path.isfile(path):
            continue
        if name.endswith(".env") or name.startswith(".env."):
            with open(path, encoding="utf-8", errors="ignore") as f:
                if any(line.strip().startswith(("DATABASE_URL=", "GEMINI_API_KEY=")) for line in f):
                    warnings.append(f"'{name}' contains PromptShield settings, but the backend only reads '.env'.")
    return warnings
