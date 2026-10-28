"""Guiones (.yaml) -> tablas de verdad de campo + transcripciones de referencia + distribución de campos.

La verdad de campo no cuesta nada: el guion se escribió ANTES de grabar, así que el texto exacto y los
valores correctos ya existen. Si al grabar alguien se desvía del guion, corrige el .yaml y vuelve a correr esto.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vozregistro.config import load_config, path  # noqa: E402


def load_scripts(task: str) -> list[dict]:
    d = path("scripts_dir") / task
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.yaml"))]


def reference_text(script: dict) -> str:
    return " ".join(next(iter(t.values())) for t in script["turns"])


def main():
    cfg = load_config()
    out = path("ground_truth_dir")
    (out / "reference").mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in ("interview", "order"):
        rows, scripts = [], load_scripts(task)
        for s in scripts:
            ref = reference_text(s)
            (out / "reference" / f"{s['id']}.txt").write_text(ref, encoding="utf-8")
            row = {"clip": s["id"], "palabras_referencia": len(ref.split()), **s["ground_truth"]}
            if task == "order":
                row.update({f"trampa_{k}": v for k, v in s["traps"].items()})
                row["items"] = json.dumps(row["items"], ensure_ascii=False)
            else:
                row["tecnologias"] = "; ".join(row["tecnologias"])
                row["reference_date"] = s["reference_date"]
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{task}.csv", index=False, encoding="utf-8")
        (out / f"{task}.json").write_text(json.dumps({s["id"]: {**s["ground_truth"], "traps": s.get("traps"),
                                                                  "reference_date": s.get("reference_date")}
                                                      for s in scripts}, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
        dist = {}
        for col in df.columns:
            if col in ("clip", "items"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != bool:
                dist[col] = {"min": float(df[col].min()), "max": float(df[col].max()),
                             "media": float(df[col].mean()), "n": int(df[col].count())}
            else:
                vals = Counter(v for cell in df[col].fillna("null").astype(str) for v in cell.split("; "))
                dist[col] = dict(vals.most_common())
        summary[task] = {"clips": len(df), "distribucion": dist}
        print(f"\n== {task}: {len(df)} clips ==")
        print(df.drop(columns=[c for c in ("items",) if c in df]).to_string(index=False))
    (out / "distribution.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
