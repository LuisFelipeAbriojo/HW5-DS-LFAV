"""Caché de transcripciones: se calculan una vez (en GPU) y se versionan; todo lo demás las lee de aquí."""
from __future__ import annotations

import json
from pathlib import Path

from .config import load_config, path


def glossary_label(glossary) -> str:
    """True/"hotwords" -> glos, "initial_prompt" -> prompt, False/None/"none" -> plain."""
    if glossary in (False, None, "none"):
        return "plain"
    return "prompt" if glossary == "initial_prompt" else "glos"


def tag(model: str, compute_type: str, device: str, glossary, level: str = "L0") -> str:
    return f"{level}__{model}__{compute_type}__{device}__{glossary_label(glossary)}"


def transcript_path(task: str, clip: str, tg: str, group: str = "real") -> Path:
    return path("cache_dir") / "transcripts" / group / task / tg / f"{clip}.json"


def save(task: str, clip: str, tg: str, data: dict, group: str = "real") -> Path:
    p = transcript_path(task, clip, tg, group)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_tag(task: str, tg: str, group: str = "real") -> dict[str, dict]:
    d = path("cache_dir") / "transcripts" / group / task / tg
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def available_tags(task: str, group: str = "real") -> list[str]:
    d = path("cache_dir") / "transcripts" / group / task
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def main_tag(task: str, level: str = "L0", group: str = "real", cfg: dict | None = None) -> str | None:
    """Etiqueta principal según config; si no hay caché para ella, el primer respaldo disponible."""
    cfg = cfg or load_config()
    ev = cfg["evaluation"]
    have = set(available_tags(task, group))
    for spec in [ev["main"], *ev["fallbacks"]]:
        tg = tag(spec["model"], spec["compute_type"], spec["device"], spec["glossary"], level)
        if tg in have:
            return tg
    same_level = [t for t in sorted(have) if t.startswith(level + "__")]
    return same_level[0] if same_level else None
