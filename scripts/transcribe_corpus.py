"""Transcribe un corpus y guarda el resultado en cache/ (corre UNA vez, en GPU; ver notebooks/colab_gpu.ipynb).

  python scripts/transcribe_corpus.py --task interview
  python scripts/transcribe_corpus.py --task order --levels L0 L1 L2 L3 L4
  python scripts/transcribe_corpus.py --task interview --model small --device cpu --compute-type int8
"""
import argparse

from _common import clips, find_audio, load_config

from vozregistro import cache
from vozregistro.asr import resolve_device, transcribe


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["interview", "order"])
    ap.add_argument("--levels", nargs="+", default=["L0"])
    ap.add_argument("--model", default=cfg["asr"]["model"])
    ap.add_argument("--device", default=cfg["asr"]["device"])
    ap.add_argument("--compute-type", default=cfg["asr"]["compute_type"])
    ap.add_argument("--no-glossary", action="store_true")
    ap.add_argument("--group", default="real", choices=["real", "synthetic"])
    ap.add_argument("--force", action="store_true", help="re-transcribir aunque exista en caché")
    a = ap.parse_args()
    terms = None if a.no_glossary else cfg["glossary"][a.task]
    for level in a.levels:
        for clip in clips(a.task):
            audio = find_audio(a.task, clip, level, a.group)
            if audio is None:
                print(f"[{level}] {clip}: sin audio, se omite")
                continue
            expected = cache.tag(a.model, a.compute_type, resolve_device(a.device), not a.no_glossary, level)
            if cache.transcript_path(a.task, clip, expected, a.group).exists() and not a.force:
                print(f"[{expected}] {clip}: ya en caché")
                continue
            r = transcribe(str(audio), cfg["asr"], model_size=a.model, device=a.device,
                           compute_type=a.compute_type, glossary_terms=terms,
                           vram_interval=cfg["benchmark"]["vram_poll_seconds"])
            # etiqueta = precisión pedida + dispositivo REAL leído del modelo (una caída a CPU no se disfraza de cuda)
            tg = cache.tag(a.model, a.compute_type, r["device"], not a.no_glossary, level)
            if tg != expected:
                print(f"  AVISO: se esperaba {expected} y corrió {tg}")
            cache.save(a.task, clip, tg, {"clip": clip, "level": level, "group": a.group, **r}, a.group)
            print(f"[{tg}] {clip}: {r['asr_s']}s para {r['audio_s']}s de audio | device={r['device']} "
                  f"vram_pico={r['peak_vram_mb']}MB")


if __name__ == "__main__":
    main()
