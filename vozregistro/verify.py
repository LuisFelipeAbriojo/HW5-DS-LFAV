"""Regla de verificación: todo valor extraído debe existir LITERALMENTE en la transcripción.

"Literal" se evalúa sobre la forma normalizada (textnorm.normalize_for_match) de ambos textos,
de modo que "seis mil quinientos" en el audio y 6500 en el registro coinciden, pero un número
que el modelo "corrigió" y que nadie dijo NO coincide. Si falla, el campo queda sin_verificar.
"""
from __future__ import annotations

import re

from .config import load_config
from .textnorm import contains_phrase, normalize_for_match

_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def status(ok: bool | None, cfg: dict | None = None) -> str:
    v = (cfg or load_config())["verification"]
    if ok is None:
        return "no_extraido"
    return v["status_ok"] if ok else v["status_fail"]


def numbers_in(text_norm: str) -> list[float]:
    return [float(x) for x in _NUM.findall(text_norm or "")]


def has_number(text_norm: str, value: float, tol: float = 0.005) -> bool:
    return any(abs(x - float(value)) <= tol for x in numbers_in(text_norm))


def evidence_in_transcript(evidence, transcript_norm: str) -> bool:
    """Cada fragmento de evidencia (str o lista) debe aparecer tal cual en la transcripción normalizada."""
    if not evidence:
        return False
    frags = evidence if isinstance(evidence, list) else [evidence]
    return all(contains_phrase(transcript_norm, f) for f in frags if f) and any(frags)


def verify_number(value, evidence, transcript_norm: str) -> bool | None:
    if value is None:
        return None
    if not evidence_in_transcript(evidence, transcript_norm):
        return False
    ev = " ".join(evidence) if isinstance(evidence, list) else evidence
    return has_number(normalize_for_match(ev), value)


def verify_phrase(value, transcript_norm: str) -> bool | None:
    if value in (None, ""):
        return None
    return contains_phrase(transcript_norm, str(value)) or contains_phrase(
        transcript_norm.replace(" ", ""), str(value).replace(" ", ""))


def verify_raw_literal(value, raw_transcript: str) -> bool | None:
    """Verificación ingenua SIN capa de normalización (para la ablación del README)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        forms = {str(int(v))} if v.is_integer() else {f"{v:.2f}", f"{v:.2f}".replace(".", ",")}
        return any(f in raw_transcript for f in forms)
    if isinstance(value, list):
        return all(str(x).lower() in raw_transcript.lower() for x in value) if value else None
    return str(value).lower() in raw_transcript.lower()
