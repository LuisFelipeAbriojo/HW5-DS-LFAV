"""Normalización de texto compartida por métricas (WER/CER) y por la regla de verificación.

Regla para WER/CER (se aplica idéntica a referencia e hipótesis):
  1. minúsculas  2. sin tildes ni diéresis (ñ -> n)  3. números en palabras -> dígitos (numbers.py)
  4. sin puntuación (se conserva el punto decimal entre dígitos y el %)  5. espacios colapsados.
"""
from __future__ import annotations

import re

from .config import load_config
from .numbers import normalize_numbers, strip_accents


def soft_normalize(text: str) -> str:
    """Minúsculas + sin tildes + números a dígitos, CONSERVANDO la puntuación (sirve para cortar frases)."""
    return normalize_numbers(strip_accents((text or "").lower()))


def normalize_for_wer(text: str, cfg: dict | None = None) -> str:
    rules = (cfg or load_config())["normalization"]
    t = text or ""
    if rules.get("lowercase", True):
        t = t.lower()
    if rules.get("strip_accents", True):
        t = strip_accents(t)
    if rules.get("numbers_to_digits", True):
        t = normalize_numbers(t)
    if rules.get("strip_punctuation", True):
        t = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", t)
        t = re.sub(r"[^\w\s%<>]", " ", t)
        t = t.replace("<DOT>", ".").replace("<", " ").replace(">", " ")
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_match(text: str) -> str:
    """Forma usada por la verificación literal: igual que WER (sin puntuación)."""
    return normalize_for_wer(text)


def contains_phrase(haystack_norm: str, needle: str) -> bool:
    """¿La frase normalizada aparece literalmente (con límites de palabra) en el texto normalizado?"""
    n = normalize_for_match(needle)
    if not n:
        return False
    return re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", haystack_norm) is not None
