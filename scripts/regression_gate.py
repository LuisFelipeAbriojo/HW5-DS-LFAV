"""Puerta de regresión para CI: recalcula métricas desde la caché y falla si la calidad cae bajo el umbral.

Los runners de GitHub Actions no tienen GPU: este script NUNCA transcribe. Solo lee cache/ y results/.
"""
import json
import subprocess
import sys

from _common import ROOT, load_config, path


def main():
    cfg = load_config()
    gate = cfg["regression_gate"]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--offline"], check=True)
    failures, checked = [], 0
    for task in ("interview", "order"):
        p = path("results_dir") / f"{task}_eval.json"
        if not p.exists():
            continue
        ev = json.loads(p.read_text(encoding="utf-8"))
        e = ev["levels"].get("L0")
        if not e:
            continue
        checked += 1
        if e["wer"]["media"] > gate["max_wer"]:
            failures.append(f"{task}: WER {e['wer']['media']:.3f} > {gate['max_wer']}")
        agg = e["por_metodo"].get("rules")
        if agg and agg["global"]["exactitud"] < gate["min_field_accuracy"]:
            failures.append(f"{task}: exactitud reglas {agg['global']['exactitud']:.2f} < {gate['min_field_accuracy']}")
    if not checked:
        print("No hay transcripciones reales cacheadas todavía: nada que verificar.")
    print("\n".join(failures) if failures else f"OK: {checked} tarea(s) dentro de umbrales")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
