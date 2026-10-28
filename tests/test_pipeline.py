"""Sobre las transcripciones de referencia (ASR perfecto) la extracción por reglas debe recuperar la verdad
de campo y el validador debe atrapar exactamente las llamadas con total erróneo plantado."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vozregistro.engine import process_transcript
from vozregistro.schemas import get_schema
from vozregistro.verify import verify_number
from vozregistro.textnorm import normalize_for_match

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def ground_truth():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_ground_truth.py")], check=True, capture_output=True)


def _gt(task):
    return json.loads((ROOT / "data/ground_truth" / f"{task}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("task", ["interview", "order"])
def test_rules_on_reference(task):
    mod = get_schema(task)
    for clip, g in _gt(task).items():
        text = (ROOT / "data/ground_truth/reference" / f"{clip}.txt").read_text(encoding="utf-8")
        r = process_transcript(text, task, "rules", reference_date=g.get("reference_date"))
        bad = [f for f, s in mod.score(r["fields"], g).items() if not s["ok"]]
        assert not bad, f"{clip}: {bad}"
        if task == "order":
            assert (r["validation"]["cuadra"] is False) == bool(g["traps"]["total_erroneo"]), clip


def test_verification_rejects_corrected_number():
    t = normalize_for_match("Estaba pensando en seis mil, no, seis mil quinientos soles.")
    assert verify_number(6500, "seis mil quinientos soles", t)
    assert not verify_number(6600, "seis mil seiscientos soles", t)   # evidencia inventada
    assert not verify_number(6600, "seis mil quinientos soles", t)    # valor "corregido" por el modelo
