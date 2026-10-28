"""Genera los niveles L1..L3 (y L4 de endurecimiento) a partir de los audios L0. Reproducible (semillas en config).

  python scripts/build_ladder.py --task order
"""
import argparse
import json

from _common import clips, find_audio, load_config, path

from vozregistro import ladder


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="order", choices=["interview", "order"])
    ap.add_argument("--group", default="real", choices=["real", "synthetic"])
    ap.add_argument("--levels", nargs="+", default=cfg["ladder"]["levels"][1:] + list(cfg["ladder"]["hardening"]))
    a = ap.parse_args()
    report = {}
    for clip in clips(a.task):
        src = find_audio(a.task, clip, "L0", a.group)
        if src is None:
            print(f"{clip}: sin audio L0")
            continue
        x = ladder.load(str(src))
        report[clip] = {}
        for level in a.levels:
            y, sr, meta = ladder.build_level(x, level, cfg, path("noise_file"))
            out = path("ladder_dir") / a.group / a.task / level / f"{clip}.wav"
            ladder.write(out, y, sr)
            report[clip][level] = meta
            print(f"{clip} {level}: {meta}")
    res = path("results_dir")
    res.mkdir(exist_ok=True)
    (res / f"ladder_build_{a.group}_{a.task}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
