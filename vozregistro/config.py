"""Carga config.yaml y .env. Ningún otro módulo lee rutas o umbrales por su cuenta."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path or os.environ.get("VOZREGISTRO_CONFIG", ROOT / "config.yaml"))
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / cfg["paths"][key]
