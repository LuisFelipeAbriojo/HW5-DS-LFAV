"""Calcula las dos capas de métricas de ambas tareas DESDE LA CACHÉ (no transcribe; corre sin GPU y en CI).

  python scripts/evaluate.py                      # reglas + LLM (LLM usa caché; si falta y hay API key, la llama)
  python scripts/evaluate.py --methods rules      # solo reglas (sin credenciales)
  python scripts/evaluate.py --offline            # nunca llama a la API: LLM solo si está en caché

Escribe results/interview_eval.json y results/order_eval.json, que es lo que lee la app.
"""
import argparse
import json
import os
from collections import defaultdict

from _common import clips, load_config, path, reference

from vozregistro import cache
from vozregistro.engine import process_transcript
from vozregistro.metrics import bootstrap_ci, macro_f1, wer, cer, wilson_ci
from vozregistro.schemas import get_schema
from vozregistro.verify import verify_raw_literal


def load_gt(task):
    return json.loads((path("ground_truth_dir") / f"{task}.json").read_text(encoding="utf-8"))


def sources(task, level, group, cfg):
    """Transcripciones a evaluar: la referencia humana (techo: ASR perfecto) y la principal cacheada."""
    out = {}
    if level == "L0" and group == "real":
        out["referencia"] = {c: {"text": reference(c), "words": None} for c in clips(task)}
    tg = cache.main_tag(task, level, group, cfg)
    if tg:
        out[tg] = cache.load_tag(task, tg, group)
    return out


def run_methods(text, words, task, methods, gt_clip, cfg, offline):
    res = {}
    for m in methods:
        if m == "llm" and offline:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            r = process_transcript(text, task, m, words=words, reference_date=gt_clip.get("reference_date"), cfg=cfg)
        except Exception as exc:
            res[m] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        mod = get_schema(task)
        sc = mod.score(r["fields"], gt_clip, None)
        raw = {}
        if task == "interview":
            raw = {f: verify_raw_literal(r["fields"].get(f), text) for f in ("anios_experiencia", "pretension_salarial")}
        res[m] = {"fields": r["fields"], "verified": r["verified"], "score": sc, "raw_literal": raw,
                  "validation": r.get("validation")}
    return res


def aggregate(rows, task, method, cfg):
    mod = get_schema(task)
    ok = [r for r in rows if method in r["methods"] and "error" not in r["methods"][method]]
    if not ok:
        return None
    n = len(ok)
    per_field = {}
    for f in mod.FIELDS:
        hits = sum(r["methods"][method]["score"][f]["ok"] for r in ok)
        lo, hi = wilson_ci(hits, n)
        per_field[f] = {"aciertos": hits, "n": n, "exactitud": hits / n, "ic95_wilson": [lo, hi]}
    total_hits = sum(v["aciertos"] for v in per_field.values())
    lo, hi = wilson_ci(total_hits, n * len(mod.FIELDS))
    out = {"n_clips": n, "por_campo": per_field,
           "global": {"aciertos": total_hits, "n": n * len(mod.FIELDS), "exactitud": total_hits / (n * len(mod.FIELDS)),
                      "ic95_wilson": [lo, hi]}}
    cat = next(iter(mod.CATEGORICAL))
    labels = cfg["interview"]["english_levels"] if task == "interview" else cfg["order"]["payment_methods"]
    gts = load_gt(task)
    out[f"macro_f1_{cat}"] = macro_f1([gts[r["clip"]][cat] for r in ok],
                                      [r["methods"][method]["fields"].get(cat) for r in ok], labels)
    errs = defaultdict(list)
    for r in ok:
        for f, s in r["methods"][method]["score"].items():
            for k in ("abs_err", "abs_err_days"):
                if s.get(k) is not None:
                    errs[f"{f}_{k}"].append(s[k])
    out["error_absoluto_medio"] = {k: sum(v) / len(v) for k, v in errs.items()}
    # ¿verificado implica correcto? (lo que la verificación promete)
    ver_ok = ver_n = unv_ok = unv_n = 0
    for r in ok:
        for f in mod.FIELDS:
            st = r["methods"][method]["verified"].get(f)
            good = r["methods"][method]["score"][f]["ok"]
            if st == cfg["verification"]["status_ok"]:
                ver_n += 1
                ver_ok += good
            elif st == cfg["verification"]["status_fail"]:
                unv_n += 1
                unv_ok += good
    out["verificacion"] = {"campos_verificados": ver_n, "exactitud_si_verificado": ver_ok / ver_n if ver_n else None,
                           "campos_sin_verificar": unv_n, "exactitud_si_sin_verificar": unv_ok / unv_n if unv_n else None}
    if task == "interview":
        raw_n = sum(1 for r in ok for v in r["methods"][method]["raw_literal"].values() if v)
        norm_n = sum(1 for r in ok for f in ("anios_experiencia", "pretension_salarial")
                     if r["methods"][method]["verified"].get(f) == cfg["verification"]["status_ok"])
        out["ablacion_normalizacion_numerica"] = {
            "campos_numericos": 2 * n, "verificados_sin_capa": raw_n, "verificados_con_capa": norm_n}
    if task == "order":
        out["validacion"] = validation_table(ok, method, gts)
    return out


def validation_table(rows, method, gts):
    caught = missed = false_alarm = true_ok = unchecked = 0
    diag_right = diag_n = 0
    detail = []
    for r in rows:
        trap = bool(gts[r["clip"]]["traps"]["total_erroneo"])
        v = r["methods"][method]["validation"]
        flagged = v["cuadra"] is False
        if v["cuadra"] is None:
            unchecked += 1
        elif trap and flagged:
            caught += 1
        elif trap and not flagged:
            missed += 1
        elif not trap and flagged:
            false_alarm += 1
        else:
            true_ok += 1
        truth = None
        if flagged:
            sc = r["methods"][method]["score"]
            record_matches = sc["items"]["ok"] and sc["promocion"]["ok"] and sc["total_dicho"]["ok"]
            truth = "error_hablante" if (trap and record_matches) else "error_asr"
            diag_n += 1
            diag_right += v["diagnostico"] == truth
        detail.append({"clip": r["clip"], "trampa_total_erroneo": trap, "cuadra": v["cuadra"],
                       "esperado": v.get("esperado"), "dicho": v.get("dicho"), "diagnostico": v["diagnostico"],
                       "diagnostico_real": truth, "senales": v.get("senales")})
    return {"atrapados": caught, "perdidos": missed, "falsas_alarmas": false_alarm, "correctos_sin_alarma": true_ok,
            "no_chequeables": unchecked, "diagnostico_acierta": diag_right, "diagnostico_n": diag_n, "detalle": detail}


def evaluate_task(task, levels, methods, group, cfg, offline):
    gts = load_gt(task)
    out = {"task": task, "group": group, "levels": {}}
    for level in levels:
        for src, data in sources(task, level, group, cfg).items():
            rows = []
            for clip, d in data.items():
                if clip not in gts:
                    continue
                row = {"clip": clip, "fuente": src}
                if src != "referencia":
                    row.update(wer=wer(reference(clip), d["text"]), cer=cer(reference(clip), d["text"]),
                               device=d.get("device"), s_per_min=d.get("s_per_audio_min"))
                row["methods"] = run_methods(d["text"], d.get("words"), task, methods, gts[clip], cfg, offline)
                rows.append(row)
            key = "REF" if src == "referencia" else level
            entry = {"fuente": src, "clips": rows, "por_metodo": {m: aggregate(rows, task, m, cfg) for m in methods}}
            if src != "referencia" and rows:
                it, seed = cfg["benchmark"]["bootstrap_iterations"], cfg["benchmark"]["seed"]
                m, lo, hi = bootstrap_ci([r["wer"] for r in rows], it, seed)
                entry["wer"] = {"media": m, "ic95": [lo, hi]}
                m, lo, hi = bootstrap_ci([r["cer"] for r in rows], it, seed)
                entry["cer"] = {"media": m, "ic95": [lo, hi]}
            out["levels"][key] = entry
    if task == "interview" and "L0" in out["levels"]:
        out["wer_vs_registro"] = wer_vs_record(out["levels"]["L0"]["clips"], methods)
    return out


def wer_vs_record(rows, methods):
    """Busca el contraejemplo pedido: WER alto con registro perfecto y WER bajo con registro roto."""
    table = []
    for r in rows:
        for m in methods:
            if m in r["methods"] and "score" in r["methods"][m]:
                sc = r["methods"][m]["score"]
                table.append({"clip": r["clip"], "metodo": m, "wer": r["wer"],
                              "campos_ok": sum(v["ok"] for v in sc.values()), "campos": len(sc),
                              "fallidos": [f for f, v in sc.items() if not v["ok"]]})
    if not table:
        return {"tabla": []}
    perfect = [t for t in table if t["campos_ok"] == t["campos"]]
    broken = [t for t in table if t["campos_ok"] < t["campos"]]
    return {"tabla": sorted(table, key=lambda t: t["wer"]),
            "wer_alto_registro_perfecto": max(perfect, key=lambda t: t["wer"]) if perfect else None,
            "wer_bajo_registro_roto": min(broken, key=lambda t: t["wer"]) if broken else None}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=cfg["extraction"]["methods"])
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    res = path("results_dir")
    res.mkdir(exist_ok=True)
    order_levels = cfg["ladder"]["levels"] + list(cfg["ladder"]["hardening"])
    for group in cfg["evaluation"]["groups"]:
        suffix = "" if group == "real" else "_synthetic"
        for task, levels in (("interview", ["L0"]), ("order", order_levels)):
            r = evaluate_task(task, levels, a.methods, group, cfg, a.offline)
            if not r["levels"]:
                continue
            (res / f"{task}_eval{suffix}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str),
                                                           encoding="utf-8")
            for lv, e in r["levels"].items():
                for m, agg in e["por_metodo"].items():
                    if agg:
                        extra = f" WER={e['wer']['media']:.3f}" if "wer" in e else ""
                        val = agg.get("validacion")
                        vtxt = f" atrapados={val['atrapados']} perdidos={val['perdidos']} falsas={val['falsas_alarmas']}" if val else ""
                        print(f"[{group}] {task} {lv:4s} {m:5s} exactitud={agg['global']['exactitud']:.2f}{extra}{vtxt}")


if __name__ == "__main__":
    main()
