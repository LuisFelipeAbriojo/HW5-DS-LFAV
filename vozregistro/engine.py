"""Motor único de las dos tareas. Este módulo NO importa ninguna librería de interfaz.

    from vozregistro.engine import process_audio
    result = process_audio("data/audio/interview/E01.wav", schema="interview")
    result = process_audio("data/audio/order/P01.wav", schema="order", method="llm")

Devuelve siempre el mismo dict: transcript, fields, verified, evidence, validation (solo pedidos),
device, compute_type, model, elapsed_s, error. Los errores no se lanzan: se devuelven en "error".
"""
from __future__ import annotations

import time
import traceback
from datetime import date

from .config import load_config
from .schemas import get_schema


def _ref_date(reference_date, cfg) -> date:
    if isinstance(reference_date, date):
        return reference_date
    return date.fromisoformat(str(reference_date or cfg["interview"]["default_reference_date"]))


def process_transcript(transcript: str, schema: str = "interview", method: str | None = None, *,
                       words: list[dict] | None = None, reference_date=None, cfg: dict | None = None,
                       use_llm_cache: bool = True) -> dict:
    """Extracción + verificación + validación sobre un texto ya transcrito (lo usa la evaluación desde caché)."""
    cfg = cfg or load_config()
    mod = get_schema(schema)
    method = method or cfg["extraction"]["default_method"]
    ref = _ref_date(reference_date, cfg)
    if method == "rules":
        fields, evidence = mod.extract_rules(transcript, cfg, ref)
    elif method == "llm":
        from .extract_llm import extract_llm
        fields, evidence = mod.from_llm(extract_llm(transcript, mod, cfg, use_cache=use_llm_cache), cfg, ref)
    else:
        raise ValueError(f"Método desconocido: {method}")
    verified, verify_detail = mod.verify(fields, evidence, transcript, cfg, ref)
    out = {"schema": schema, "method": method, "transcript": transcript, "fields": fields,
           "evidence": evidence, "verified": verified, "verify_detail": verify_detail}
    if schema == "order":
        from .validate import validate_order
        out["validation"] = validate_order(fields, words, cfg)
    return out


def process_audio(audio_path: str, schema: str = "interview", method: str | None = None, *,
                  model: str | None = None, device: str | None = None, compute_type: str | None = None,
                  glossary: bool = True, reference_date=None, cfg: dict | None = None) -> dict:
    """Audio -> registro estructurado. Única función que llama la app (y el bot/MCP si se agregan)."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    result = {"audio_path": str(audio_path), "schema": schema, "method": method or cfg["extraction"]["default_method"],
              "transcript": None, "fields": None, "verified": None, "evidence": None, "validation": None,
              "device": None, "compute_type": None, "model": None, "elapsed_s": None, "asr": None, "error": None}
    try:
        from .asr import transcribe
        terms = cfg["glossary"].get(schema) if glossary else None
        asr = transcribe(str(audio_path), cfg["asr"], model_size=model, device=device,
                         compute_type=compute_type, glossary_terms=terms,
                         vram_interval=cfg["benchmark"]["vram_poll_seconds"])
        result.update(device=asr["device"], compute_type=asr["compute_type"], model=asr["model"],
                      asr={k: v for k, v in asr.items() if k not in ("text", "words")})
        result["transcript"] = asr["text"]
        rec = process_transcript(asr["text"], schema, method, words=asr["words"],
                                 reference_date=reference_date, cfg=cfg)
        result.update({k: rec[k] for k in ("fields", "verified", "evidence", "verify_detail", "method")})
        result["validation"] = rec.get("validation")
        result["words"] = asr["words"]
    except Exception as exc:  # el motor nunca tumba a quien lo llama
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return result
