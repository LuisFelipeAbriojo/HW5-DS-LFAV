"""Ruta LLM: un modelo de lenguaje devuelve JSON estructurado con valor + evidencia literal por campo.

Proveedores (config.yaml -> extraction.llm.provider):
  ollama     modelo abierto local (por defecto qwen2.5:7b-instruct), gratis y sin API key; el esquema JSON
             se impone con el parámetro `format` de Ollama (decodificación restringida por gramática).
  anthropic  Claude por API (requiere ANTHROPIC_API_KEY).

Las respuestas se cachean por hash (transcripción + prompt + esquema + proveedor + modelo) para que la
evaluación y el CI no vuelvan a llamar al modelo.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import path


def _cache_file(key: str) -> Path:
    d = path("cache_dir") / "llm"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _user_message(transcript: str) -> str:
    return f"Transcripción:\n<transcripcion>\n{transcript}\n</transcripcion>"


def _call_ollama(system: str, transcript: str, json_schema: dict, lcfg: dict) -> str:
    from ollama import Client  # import perezoso: la ruta de reglas no necesita el cliente

    o = lcfg["ollama"]
    resp = Client(host=o["host"]).chat(
        model=o["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": _user_message(transcript)}],
        format=json_schema,
        options={"temperature": o["temperature"], "seed": o["seed"], "num_ctx": o["num_ctx"]},
    )
    return resp["message"]["content"]


def _call_anthropic(system: str, transcript: str, json_schema: dict, lcfg: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    kwargs = dict(model=lcfg["model"], max_tokens=lcfg["max_tokens"], system=system,
                  messages=[{"role": "user", "content": _user_message(transcript)}],
                  output_config={"effort": lcfg["effort"], "format": {"type": "json_schema", "schema": json_schema}})
    if lcfg.get("fallbacks"):
        kwargs["extra_headers"] = {"anthropic-beta": lcfg["fallback_beta"]}
        kwargs["extra_body"] = {"fallbacks": lcfg["fallbacks"]}
    response = client.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        raise RuntimeError(f"El modelo rechazó la solicitud: {getattr(response, 'stop_details', None)}")
    return next(b.text for b in response.content if b.type == "text")


def _fill_missing(data: dict, schema: dict) -> dict:
    """Un modelo pequeño puede omitir claves: se completan con null/[] para que nada falle aguas abajo.
    Un campo completado así no trae evidencia, de modo que la verificación lo marca como no extraído."""
    for key, spec in schema["properties"].items():
        if key in data:
            continue
        if spec.get("type") == "array":
            data[key] = []
        elif spec.get("type") == "object":
            ev = spec["properties"].get("evidencia", {})
            data[key] = {"valor": [] if spec["properties"]["valor"].get("type") == "array" else None,
                         "evidencia": [] if ev.get("type") == "array" else None}
        else:
            data[key] = None
    return data


def extract_llm(transcript: str, schema_mod, cfg: dict, use_cache: bool = True) -> dict:
    lcfg = cfg["extraction"]["llm"]
    provider = lcfg.get("provider", "ollama")
    model = lcfg["ollama"]["model"] if provider == "ollama" else lcfg["model"]
    cats = (cfg["interview"]["english_levels"] if schema_mod.NAME == "interview"
            else cfg["order"]["payment_methods"])
    system = lcfg["prompts"][schema_mod.NAME].replace("{categories}", ", ".join(cats))
    json_schema = schema_mod.llm_schema(cfg)
    key = hashlib.sha256(json.dumps([transcript, system, json_schema, provider, model],
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    cf = _cache_file(key)
    if use_cache and cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))

    call = _call_ollama if provider == "ollama" else _call_anthropic
    data = _fill_missing(json.loads(call(system, transcript, json_schema, lcfg)), json_schema)
    data["_modelo"] = f"{provider}:{model}"
    cf.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data
