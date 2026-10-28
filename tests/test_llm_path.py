"""La ruta LLM se prueba sin red: se simula la respuesta JSON y se comprueba que verificación y validación
tratan igual a las dos rutas (un valor sin evidencia literal queda sin_verificar)."""
import vozregistro.extract_llm as ex
from vozregistro.engine import process_transcript

TEXT = ("Mi último puesto fue analista de datos. Tengo cuatro años de experiencia. Uso Python y SQL. "
        "Mi pretensión es de seis mil quinientos soles. Podría empezar en dos semanas. Mi inglés es intermedio.")


def fake(value_salary, evidence_salary):
    return {"anios_experiencia": {"valor": 4, "evidencia": "cuatro años de experiencia"},
            "ultimo_puesto": {"valor": "analista de datos", "evidencia": "analista de datos"},
            "tecnologias": {"valor": ["Python", "SQL"], "evidencia": ["Python", "SQL"]},
            "pretension_salarial": {"valor": value_salary, "evidencia": evidence_salary},
            "moneda": "PEN",
            "disponibilidad": {"valor": "en dos semanas", "evidencia": "en dos semanas"},
            "nivel_ingles": {"valor": "intermedio", "evidencia": "intermedio"}}


def test_llm_verified(monkeypatch):
    monkeypatch.setattr(ex, "extract_llm", lambda *a, **k: fake(6500, "seis mil quinientos soles"))
    r = process_transcript(TEXT, "interview", "llm", reference_date="2026-10-26")
    assert r["fields"]["disponibilidad"] == "2026-11-09"
    assert set(r["verified"].values()) == {"verificado"}


def test_llm_invented_value_is_flagged(monkeypatch):
    monkeypatch.setattr(ex, "extract_llm", lambda *a, **k: fake(7000, "siete mil soles"))
    r = process_transcript(TEXT, "interview", "llm", reference_date="2026-10-26")
    assert r["verified"]["pretension_salarial"] == "sin_verificar"


def test_ollama_provider_without_api_key(monkeypatch, tmp_path):
    """El proveedor por defecto (Ollama) recibe el esquema JSON en `format` y no necesita credenciales."""
    import json

    import ollama

    calls = {}

    class FakeClient:
        def __init__(self, host):
            calls["host"] = host

        def chat(self, model, messages, format, options):
            calls.update(model=model, format=format, options=options)
            return {"message": {"content": json.dumps(fake(6500, "seis mil quinientos soles"))}}

    monkeypatch.setattr(ollama, "Client", FakeClient)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ex, "path", lambda key: tmp_path)
    r = process_transcript(TEXT, "interview", "llm", reference_date="2026-10-26", use_llm_cache=False)
    assert calls["format"]["required"]          # se pasó el esquema
    assert calls["options"]["temperature"] == 0
    assert r["verified"]["pretension_salarial"] == "verificado"


def test_missing_keys_are_filled():
    from vozregistro.config import load_config
    from vozregistro.schemas import interview
    data = ex._fill_missing({"nivel_ingles": {"valor": "basico", "evidencia": "basico"}},
                            interview.llm_schema(load_config()))
    assert data["tecnologias"] == {"valor": [], "evidencia": []}
    assert data["pretension_salarial"] == {"valor": None, "evidencia": None}
