"""Fase 2 de la Tarea 1: tamaños de modelo, CPU vs GPU, precisiones y glosario. Correr en GPU (Colab).

  python scripts/benchmark_asr.py                 # todo el plan
  python scripts/benchmark_asr.py --only sizes    # sizes | precisions | cpu_gpu | glossary
  python scripts/benchmark_asr.py --summarize-only   # recalcula métricas desde caché, sin GPU

Cada clip se mide con el modelo recién cargado (se liberan los modelos) para que la VRAM pico incluya los pesos.
"""
import argparse
import json

from _common import clips, find_audio, load_config, path, reference

from vozregistro import cache
from vozregistro.asr import glossary_support, release_models, transcribe
from vozregistro.metrics import bootstrap_ci, cer, wer
from vozregistro.textnorm import contains_phrase, normalize_for_wer

TASK = "interview"


def plan(cfg):
    b, main = cfg["benchmark"], cfg["asr"]["model"]
    gpu_ct = cfg["asr"]["compute_type"]
    return {
        "sizes": [dict(model=m, device="cuda", compute_type=gpu_ct, glossary="none") for m in b["model_sizes"]],
        "precisions": [dict(model=main, device="cuda", compute_type=p, glossary="none") for p in b["precisions"]],
        "cpu_gpu": [dict(model=b["cpu_model"], device="cpu", compute_type=b["cpu_compute_type"], glossary="none"),
                    dict(model=b["cpu_model"], device="cuda", compute_type=gpu_ct, glossary="none")],
        "glossary": [dict(model=main, device="cuda", compute_type=gpu_ct, glossary=g)
                     for g in ("none", "hotwords", "initial_prompt")],
    }


def run_config(c, cfg, force=False):
    terms = None if c["glossary"] == "none" else cfg["glossary"][TASK]
    for clip in clips(TASK):
        audio = find_audio(TASK, clip)
        if audio is None:
            continue
        tg_req = cache.tag(c["model"], c["compute_type"], c["device"], c["glossary"])
        if cache.transcript_path(TASK, clip, tg_req).exists() and not force:
            continue
        release_models()  # cada clip carga el modelo de cero: la VRAM pico incluye los pesos
        r = transcribe(str(audio), cfg["asr"], model_size=c["model"], device=c["device"],
                       compute_type=c["compute_type"], glossary_terms=terms, glossary_mode=c["glossary"],
                       vram_interval=cfg["benchmark"]["vram_poll_seconds"])
        tg = cache.tag(c["model"], c["compute_type"], r["device"], c["glossary"])
        if tg != tg_req:
            print(f"  AVISO: se pidió {tg_req} pero corrió {tg} (¿sin GPU?)")
        cache.save(TASK, clip, tg, {"clip": clip, "level": "L0", "group": "real", **r})
        print(f"  {tg} {clip}: {r['s_per_audio_min']} s/min, device={r['device']}, vram_pico={r['peak_vram_mb']} MB")


def tag_metrics(tg, cfg, gt):
    data = cache.load_tag(TASK, tg)
    if not data:
        return None
    rows = []
    for clip, d in data.items():
        ref = reference(clip)
        techs = gt[clip]["tecnologias"]
        hn = normalize_for_wer(d["text"])
        recall = sum(contains_phrase(hn, t) or contains_phrase(hn.replace(" ", ""), t.replace(" ", ""))
                     for t in techs) / len(techs)
        rows.append({"clip": clip, "wer": wer(ref, d["text"]), "cer": cer(ref, d["text"]),
                     "s_per_min": d["s_per_audio_min"], "peak_vram_mb": d.get("peak_vram_mb"),
                     "delta_vram_mb": d.get("delta_vram_mb"), "process_vram_mb": d.get("process_vram_mb"),
                     "device": d["device"], "compute_type": d["compute_type"], "term_recall": recall})
    it, seed = cfg["benchmark"]["bootstrap_iterations"], cfg["benchmark"]["seed"]
    agg = {"tag": tg, "n_clips": len(rows), "devices": sorted({r["device"] for r in rows}),
           "compute_types": sorted({r["compute_type"] for r in rows})}
    for k in ("wer", "cer", "s_per_min", "term_recall"):
        m, lo, hi = bootstrap_ci([r[k] for r in rows], it, seed)
        agg[k] = {"media": m, "ic95": [lo, hi]}
    vr = [r["peak_vram_mb"] for r in rows if r["peak_vram_mb"] is not None]
    dv = [r["delta_vram_mb"] for r in rows if r["delta_vram_mb"] is not None]
    agg["peak_vram_mb"] = max(vr) if vr else None
    agg["delta_vram_mb_max"] = max(dv) if dv else None
    return {"agg": agg, "clips": rows}


def text_changed(tag_a, tag_b):
    a, b = cache.load_tag(TASK, tag_a), cache.load_tag(TASK, tag_b)
    out = []
    for c in sorted(set(a) & set(b)):
        na, nb = normalize_for_wer(a[c]["text"]), normalize_for_wer(b[c]["text"])
        out.append({"clip": c, "identico": na == nb, "wer_entre_salidas": wer(a[c]["text"], b[c]["text"])})
    return {"a": tag_a, "b": tag_b, "clips": out,
            "pct_identicos": (100 * sum(o["identico"] for o in out) / len(out)) if out else None}


def summarize(cfg):
    gt = json.loads((path("ground_truth_dir") / f"{TASK}.json").read_text(encoding="utf-8"))
    metrics = {tg: tag_metrics(tg, cfg, gt) for tg in cache.available_tags(TASK) if tg.startswith("L0__")}
    out = {"glossary_support_instalado": glossary_support(), "por_config": metrics, "comparaciones": {}}
    it, seed = cfg["benchmark"]["bootstrap_iterations"], cfg["benchmark"]["seed"]
    main, gct = cfg["asr"]["model"], cfg["asr"]["compute_type"]
    base = cache.tag(main, "float16", "cuda", "none")
    for p in cfg["benchmark"]["precisions"]:
        other = cache.tag(main, p, "cuda", "none")
        if p != "float16" and base in metrics and other in metrics:
            out["comparaciones"][f"precision_float16_vs_{p}"] = text_changed(base, other)
    cm = cfg["benchmark"]["cpu_model"]
    tc, tgpu = cache.tag(cm, cfg["benchmark"]["cpu_compute_type"], "cpu", "none"), cache.tag(cm, gct, "cuda", "none")
    if tc in metrics and tgpu in metrics:
        cmp = text_changed(tc, tgpu)
        cmp["aceleracion_gpu"] = metrics[tc]["agg"]["s_per_min"]["media"] / metrics[tgpu]["agg"]["s_per_min"]["media"]
        out["comparaciones"]["cpu_vs_gpu"] = cmp
    # glosario: diferencia PAREADA por clip (mismo audio, mismo modelo) con IC bootstrap
    plain = cache.tag(main, gct, "cuda", "none")
    for g in ("hotwords", "initial_prompt"):
        gl = cache.tag(main, gct, "cuda", g)
        if plain in metrics and gl in metrics:
            pa = {r["clip"]: r for r in metrics[plain]["clips"]}
            ga = {r["clip"]: r for r in metrics[gl]["clips"]}
            common = sorted(set(pa) & set(ga))
            m, lo, hi = bootstrap_ci([ga[c]["wer"] - pa[c]["wer"] for c in common], it, seed)
            m2, lo2, hi2 = bootstrap_ci([ga[c]["term_recall"] - pa[c]["term_recall"] for c in common], it, seed)
            out["comparaciones"][f"glosario_{g}"] = {
                "delta_wer": {"media": m, "ic95": [lo, hi]}, "delta_term_recall": {"media": m2, "ic95": [lo2, hi2]},
                "n_clips": len(common),
                "lectura": "el IC incluye 0: no se puede afirmar efecto" if lo <= 0 <= hi else "el IC excluye 0"}
    res = path("results_dir")
    res.mkdir(exist_ok=True)
    (res / "asr_benchmark.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in metrics.items():
        if v:
            a = v["agg"]
            print(f"{k:55s} WER={a['wer']['media']:.3f} CER={a['cer']['media']:.3f} "
                  f"s/min={a['s_per_min']['media']:.2f} vram_pico={a['peak_vram_mb']} device={a['devices']}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sizes", "precisions", "cpu_gpu", "glossary"])
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not a.summarize_only:
        for name, configs in plan(cfg).items():
            if a.only and name != a.only:
                continue
            print(f"== {name} ==")
            for c in configs:
                run_config(c, cfg, a.force)
    summarize(cfg)


if __name__ == "__main__":
    main()
