"""Aumentación OPCIONAL con voces sintéticas (Edge TTS, sin API key). Se guarda en data/audio_synthetic/
y se reporta SIEMPRE como grupo aparte: nunca se mezcla con las grabaciones reales.

  python scripts/synthesize_tts.py --task interview
  python scripts/synthesize_tts.py --task order --clips P01 P02
  python scripts/synthesize_tts.py --task interview --locale es-MX   # eje de acento (voces sintéticas, NO acentos humanos)
"""
import argparse
import asyncio
import tempfile
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf
import yaml
from _common import clips, path

from vozregistro.ladder import SR, load

VOICES = {"es-PE": ("es-PE-AlexNeural", "es-PE-CamilaNeural"), "es-MX": ("es-MX-JorgeNeural", "es-MX-DaliaNeural"),
          "es-ES": ("es-ES-AlvaroNeural", "es-ES-ElviraNeural"), "es-AR": ("es-AR-TomasNeural", "es-AR-ElenaNeural"),
          "es-CO": ("es-CO-GonzaloNeural", "es-CO-SalomeNeural")}


async def say(text, voice, out):
    await edge_tts.Communicate(text, voice).save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["interview", "order"])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--locale", default="es-PE", choices=list(VOICES))
    a = ap.parse_args()
    va, vb = VOICES[a.locale]
    outdir = path("synthetic_dir") / (a.task if a.locale == "es-PE" else f"{a.task}_{a.locale}")
    outdir.mkdir(parents=True, exist_ok=True)
    for clip in a.clips or clips(a.task):
        script = yaml.safe_load((path("scripts_dir") / a.task / f"{clip}.yaml").read_text(encoding="utf-8"))
        pieces = []
        with tempfile.TemporaryDirectory() as td:
            for i, turn in enumerate(script["turns"]):
                spk, text = next(iter(turn.items()))
                mp3 = str(Path(td) / f"{i}.mp3")
                asyncio.run(say(text, va if spk == "A" else vb, mp3))
                pieces += [load(mp3), np.zeros(int(0.35 * SR), np.float32)]
        y = np.concatenate(pieces)
        sf.write(str(outdir / f"{clip}.wav"), y, SR)
        print(f"{clip}: {len(y) / SR:.1f}s -> {outdir / (clip + '.wav')}")


if __name__ == "__main__":
    main()
