"""Utilidades compartidas por los scripts (localizar audios y referencias)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vozregistro.config import load_config, path  # noqa: E402


def find_audio(task: str, clip: str, level: str = "L0", group: str = "real") -> Path | None:
    cfg = load_config()
    if level != "L0":
        p = path("ladder_dir") / group / task / level / f"{clip}.wav"
        return p if p.exists() else None
    base = path("audio_dir" if group == "real" else "synthetic_dir") / task
    for ext in cfg["audio_ext"]:
        p = base / f"{clip}{ext}"
        if p.exists():
            return p
    return None


def clips(task: str) -> list[str]:
    return sorted(p.stem for p in (path("scripts_dir") / task).glob("*.yaml"))


def reference(clip: str) -> str:
    return (path("ground_truth_dir") / "reference" / f"{clip}.txt").read_text(encoding="utf-8")
