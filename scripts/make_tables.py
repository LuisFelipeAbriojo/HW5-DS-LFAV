"""Imprime las tablas de resultados en Markdown (para pegarlas en el README) a partir de results/.

  python scripts/make_tables.py            # grupo real
  python scripts/make_tables.py synthetic  # grupo sintético (reportar aparte)
"""
import json
import sys

from _common import path


def load(name):
    p = path("results_dir") / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "real"
    suf = "" if group == "real" else "_synthetic"
    bench = load("asr_benchmark.json")
    if bench and group == "real":
        print("### Tarea 1 · Transcripción (capa 1)\n")
        print("| modelo | precisión | dispositivo medido | glosario | WER (IC95) | CER | s/min audio | VRAM pico MB | recall términos |")
        print("|---|---|---|---|---|---|---|---|---|")
        for tg, m in sorted(bench["por_config"].items()):
            if not m:
                continue
            a = m["agg"]
            _, model, ct, _, gl = tg.split("__")
            print(f"| {model} | {ct} | {', '.join(a['devices'])} | {gl} | {a['wer']['media']:.3f} "
                  f"({a['wer']['ic95'][0]:.3f}–{a['wer']['ic95'][1]:.3f}) | {a['cer']['media']:.3f} | "
                  f"{a['s_per_min']['media']:.2f} | {a['peak_vram_mb']} | {pct(a['term_recall']['media'])} |")
        for k, v in bench["comparaciones"].items():
            if k.startswith("glosario"):
                d = v["delta_wer"]
                print(f"\n- **{k}**: ΔWER = {d['media']:+.3f} (IC95 {d['ic95'][0]:+.3f} a {d['ic95'][1]:+.3f}), "
                      f"Δrecall términos = {v['delta_term_recall']['media']:+.2f}, n={v['n_clips']} → {v['lectura']}")
            elif "pct_identicos" in v:
                extra = f", aceleración GPU ×{v['aceleracion_gpu']:.1f}" if "aceleracion_gpu" in v else ""
                print(f"- **{k}**: {v['pct_identicos']:.0f}% de clips con texto idéntico tras normalizar{extra}")
    for task in ("interview", "order"):
        ev = load(f"{task}_eval{suf}.json")
        if not ev:
            continue
        print(f"\n### {'Tarea 1 · Registro' if task == 'interview' else 'Tarea 2 · Pedido'} (capa 2) — grupo {group}\n")
        for lv, e in ev["levels"].items():
            for m, agg in e["por_metodo"].items():
                if not agg:
                    continue
                w = f"WER {e['wer']['media']:.3f}" if "wer" in e else "ASR perfecto"
                cells = " | ".join(f"{pct(v['exactitud'])} [{pct(v['ic95_wilson'][0])}–{pct(v['ic95_wilson'][1])}]"
                                   for v in agg["por_campo"].values())
                if lv == next(iter(ev["levels"])) and m == next(iter(e["por_metodo"])):
                    print("| nivel | método | " + " | ".join(agg["por_campo"]) + " | global |" + (" atrapados / perdidos / falsas |" if task == "order" else ""))
                    print("|---|---|" + "---|" * (len(agg["por_campo"]) + 1) + ("---|" if task == "order" else ""))
                val = agg.get("validacion")
                vt = f" {val['atrapados']} / {val['perdidos']} / {val['falsas_alarmas']} |" if val else ""
                print(f"| {lv} ({w}) | {m} | {cells} | {pct(agg['global']['exactitud'])} |{vt}")


if __name__ == "__main__":
    main()
