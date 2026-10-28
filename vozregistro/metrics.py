"""Métricas de las dos capas: transcripción (WER/CER) y registro (exactitud por campo, F1 macro, error absoluto)."""
from __future__ import annotations

import math

import jiwer
import numpy as np

from .textnorm import normalize_for_wer


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_for_wer(ref), normalize_for_wer(hyp)
    return float(jiwer.wer(r, h)) if r else float("nan")


def cer(ref: str, hyp: str) -> float:
    r, h = normalize_for_wer(ref), normalize_for_wer(hyp)
    return float(jiwer.cer(r, h)) if r else float("nan")


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    return float(jiwer.wer([normalize_for_wer(r) for r in refs], [normalize_for_wer(h) for h in hyps]))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para una proporción: se comporta bien con n pequeño (nuestro caso)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, iterations: int = 2000, seed: int = 13, stat=np.mean, alpha: float = 0.05):
    """IC por bootstrap (remuestreo de clips). Para diferencias pareadas pasa la diferencia por clip."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.array([stat(rng.choice(arr, size=arr.size, replace=True)) for _ in range(iterations)])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(stat(arr)), float(lo), float(hi))


def macro_f1(y_true: list, y_pred: list, labels: list) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        if tp + fp + fn == 0:
            continue  # clase ausente en verdad y predicción: no aporta
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def abs_error(true, pred):
    if true is None or pred is None:
        return None
    try:
        return abs(float(true) - float(pred))
    except (TypeError, ValueError):
        return None
