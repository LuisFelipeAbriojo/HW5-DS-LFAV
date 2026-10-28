"""Esquema de la Tarea 1: ficha de candidato con seis campos."""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from ..textnorm import contains_phrase, normalize_for_match, soft_normalize
from ..verify import evidence_in_transcript, status, verify_number, verify_phrase

NAME = "interview"
FIELDS = ["anios_experiencia", "ultimo_puesto", "tecnologias", "pretension_salarial",
          "disponibilidad", "nivel_ingles"]
CATEGORICAL = {"nivel_ingles"}
NUMERIC = {"anios_experiencia", "pretension_salarial"}

MONTHS = {m: i for i, m in enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                                      "agosto", "septiembre", "octubre", "noviembre", "diciembre"], 1)}
MONTHS["setiembre"] = 9

_SENT = re.compile(r"(?:[^.?!¿¡;\n]|(?<=\d)\.(?=\d))+[.?!;\n]?")  # el punto decimal no corta la frase


def sentences(soft: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(soft) if s.strip(" .?!¿¡;\n")]


# ---------- disponibilidad: dicho relativo, guardado absoluto ----------

def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def resolve_availability(expr: str | None, ref: date, cfg: dict) -> str | None:
    """Expresión dicha -> 'inmediata' o fecha ISO, relativa a la fecha de la entrevista."""
    if not expr:
        return None
    t = soft_normalize(expr)
    if any(re.search(rf"\b{re.escape(k)}\b", t) for k in cfg["interview"]["availability"]["immediate"]):
        return "inmediata"
    m = re.search(r"(?:preaviso de|en|dentro de|despues de)\s+(\d+)\s+(dias?|semanas?|mes(?:es)?)", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit.startswith("dia"):
            return (ref + timedelta(days=n)).isoformat()
        if unit.startswith("semana"):
            return (ref + timedelta(weeks=n)).isoformat()
        return _add_months(ref, n).isoformat()
    m = re.search(r"\b(primero|1ro|\d{1,2}) de (" + "|".join(MONTHS) + r")\b", t)
    if m:
        day = 1 if m.group(1) in ("primero", "1ro") else int(m.group(1))
        month = MONTHS[m.group(2)]
        year = ref.year if (month, day) >= (ref.month, ref.day) else ref.year + 1
        return date(year, month, day).isoformat()
    return None


# ---------- extracción por reglas ----------

def _years(soft: str):
    best, best_score = None, -99
    for m in re.finditer(r"(?:(\w+)\s+)?(?:(\w+)\s+)?(\d+(?:\.\d+)?)\s+anos?\b", soft):
        ctx_start = soft.rfind(".", 0, m.start()) + 1
        ctx_end = soft.find(".", m.end())
        ctx = soft[ctx_start: ctx_end if ctx_end > 0 else len(soft)]
        prev = " ".join(x for x in (m.group(1), m.group(2)) if x)
        score = 0
        score += 2 if "experiencia" in ctx else 0
        score += 2 if "en total" in soft[max(0, m.start() - 12): m.start() + 3] else 0
        score += 1 if re.search(r"\b(llevo|tengo)\b", prev) else 0
        score -= 2 if re.search(r"\b(hace|ultimo|ultimos)\b", prev) else 0
        score -= 1 if re.search(r"\b(practica\w*|marketing|antes|toronto|vivi)\b", ctx) else 0
        if score > best_score:
            best_score, best = score, (float(m.group(3)), m.group(0).strip())
    return best if best else (None, None)


_ROLE_CUES = [r"mi ultimo puesto (?:fue|es|era)", r"mi ultimo cargo (?:fue|es|era)", r"mi cargo actual es",
              r"he sido", r"me desempeno como", r"trabajo como", r"estoy como", r"soy"]
_ROLE_STOP = r"(?=[,.;?!]| en (?:una|un|1|la|el)\b| y | con (?:un|una)\b| desde | hace | aunque |$)"


def _role(soft: str):
    for cue in _ROLE_CUES:
        m = re.search(rf"\b{cue}\s+(?:de |como )?(.+?){_ROLE_STOP}", soft)
        if m and len(m.group(1).split()) <= 7:
            return m.group(1).strip(), m.group(0).strip()
    return None, None


def _technologies(soft: str, cfg: dict):
    norm = normalize_for_match(soft)
    neg = [normalize_for_match(c) for c in cfg["interview"]["negation_cues"]]
    found = []
    for term in cfg["glossary"]["interview"]:
        t = normalize_for_match(term)
        hits = [m.start() for m in re.finditer(rf"(?<![\w.]){re.escape(t)}(?!\w)", norm)]
        if not hits and " " in t:
            hits = [m.start() for m in re.finditer(rf"(?<![\w.]){re.escape(t.replace(' ', ''))}(?!\w)", norm)]
        if not hits:
            continue
        # se descarta si TODAS sus menciones están en una cláusula con negación ("azure lo he tocado muy poco")
        positive = False
        for clause in re.split(r"[,.;?!]", soft):
            cn = normalize_for_match(clause)
            if re.search(rf"(?<![\w.]){re.escape(t)}(?!\w)", cn) or t.replace(" ", "") in cn.replace(" ", ""):
                if not any(re.search(rf"\b{re.escape(c)}\b", cn) for c in neg):
                    positive = True
        if positive:
            found.append((min(hits), term))
    found.sort()
    terms = [t for _, t in found]
    return terms, terms


_SALARY_CUES = r"pretension|expectativa|cuanto esperas|cuanto esperarias|estoy buscando"


def _salary(soft: str, cfg: dict):
    m = re.search(_SALARY_CUES, soft)
    if not m:
        return None, None, None
    window = " ".join(soft[m.start():].split()[:70])
    cur_words = {w: c for c, ws in cfg["interview"]["currencies"].items() for w in ws}
    cur_alt = "|".join(re.escape(w) for w in sorted(cur_words, key=len, reverse=True))
    cands = []
    for a in re.finditer(rf"(?:(\w+)\s+)?(\d+(?:\.\d+)?)(?:\s+({cur_alt})(?!\w))?", window):
        val = float(a.group(2))
        if val < 300 or (a.group(1) in ("gano", "ganaba", "entre")):
            continue
        if a.group(1) == "y" and re.search(r"entre\s+\d+(?:\.\d+)?\s+$", window[:a.start()]):
            continue  # extremo superior de un rango ("entre 4500 y 5000"): no es la cifra final
        cands.append((val, a.group(3), a.group(0).strip()))
    if not cands:
        return None, None, None
    with_cur = [c for c in cands if c[1]]
    val, cur, ev = (with_cur or cands)[-1]
    return val, (cur_words.get(cur) if cur else None), ev


_AVAIL_CUES = r"empezar|incorporar\w*|disponib\w*|lista para|listo para|preaviso|comenzar|arrancar"


def _availability(soft: str, ref: date, cfg: dict):
    sents = sentences(soft)
    idx = [i for i, s in enumerate(sents) if re.search(_AVAIL_CUES, s)]
    scope = sorted({j for i in idx for j in range(i, min(i + 3, len(sents)))})
    imm = cfg["interview"]["availability"]["immediate"]
    for j in scope:
        s = sents[j]
        for k in sorted(imm, key=len, reverse=True):
            mk = re.search(rf"\b{re.escape(k)}\b", s)
            if mk:
                return "inmediata", mk.group(0)
        m = re.search(r"(?:preaviso de|en|dentro de|despues de)\s+\d+\s+(?:dias?|semanas?|mes(?:es)?)"
                      r"|\b(?:primero|1ro|\d{1,2}) de (?:" + "|".join(MONTHS) + r")\b", s)
        if m:
            return resolve_availability(m.group(0), ref, cfg), m.group(0)
    return None, None


def _english(soft: str, cfg: dict):
    sents = sentences(soft)
    kw = cfg["interview"]["english_keywords"]
    for i, s in enumerate(sents):
        if "ingles" not in s:
            continue
        for j in range(i, min(i + 5, len(sents))):
            for level, words in kw.items():
                for w in sorted(words, key=len, reverse=True):
                    m = re.search(rf"\b{re.escape(w)}\b", sents[j])
                    if m:
                        return level, m.group(0)
    return None, None


def extract_rules(transcript: str, cfg: dict, ref: date) -> tuple[dict, dict]:
    soft = soft_normalize(transcript)
    fields, ev = {}, {}
    fields["anios_experiencia"], ev["anios_experiencia"] = _years(soft)
    fields["ultimo_puesto"], ev["ultimo_puesto"] = _role(soft)
    fields["tecnologias"], ev["tecnologias"] = _technologies(soft, cfg)
    sal, cur, ev["pretension_salarial"] = _salary(soft, cfg)
    fields["pretension_salarial"], fields["moneda"] = sal, cur
    fields["disponibilidad"], ev["disponibilidad"] = _availability(soft, ref, cfg)
    fields["nivel_ingles"], ev["nivel_ingles"] = _english(soft, cfg)
    return fields, ev


# ---------- esquema JSON para la ruta LLM ----------

def _vf(value_schema):
    return {"type": "object", "additionalProperties": False, "required": ["valor", "evidencia"],
            "properties": {"valor": value_schema, "evidencia": {"type": ["string", "null"]}}}


def llm_schema(cfg: dict) -> dict:
    levels = cfg["interview"]["english_levels"]
    return {
        "type": "object", "additionalProperties": False, "required": FIELDS + ["moneda"],
        "properties": {
            "anios_experiencia": _vf({"type": ["number", "null"]}),
            "ultimo_puesto": _vf({"type": ["string", "null"]}),
            "tecnologias": {"type": "object", "additionalProperties": False, "required": ["valor", "evidencia"],
                            "properties": {"valor": {"type": "array", "items": {"type": "string"}},
                                           "evidencia": {"type": "array", "items": {"type": "string"}}}},
            "pretension_salarial": _vf({"type": ["number", "null"]}),
            "moneda": {"type": ["string", "null"], "enum": ["PEN", "USD", None]},
            "disponibilidad": _vf({"type": ["string", "null"]}),
            "nivel_ingles": _vf({"type": ["string", "null"], "enum": levels + [None]}),
        },
    }


def from_llm(data: dict, cfg: dict, ref: date) -> tuple[dict, dict]:
    fields, ev = {}, {}
    for f in FIELDS:
        fields[f] = data[f]["valor"]
        ev[f] = data[f]["evidencia"]
    fields["moneda"] = data.get("moneda")
    # la expresión de disponibilidad la resuelve el sistema, no el modelo
    fields["disponibilidad"] = resolve_availability(ev["disponibilidad"] or fields["disponibilidad"], ref, cfg)
    fields["tecnologias"] = fields["tecnologias"] or []
    return fields, ev


# ---------- verificación ----------

def verify(fields: dict, ev: dict, transcript: str, cfg: dict, ref: date) -> tuple[dict, dict]:
    tn = normalize_for_match(transcript)
    out, detail = {}, {}
    out["anios_experiencia"] = verify_number(fields["anios_experiencia"], ev.get("anios_experiencia"), tn)
    out["ultimo_puesto"] = verify_phrase(fields["ultimo_puesto"], tn)
    techs = fields.get("tecnologias") or []
    per = {t: bool(verify_phrase(t, tn)) for t in techs}
    detail["tecnologias"] = per
    out["tecnologias"] = (all(per.values()) if per else None)
    out["pretension_salarial"] = verify_number(fields["pretension_salarial"], ev.get("pretension_salarial"), tn)
    disp, ev_d = fields["disponibilidad"], ev.get("disponibilidad")
    out["disponibilidad"] = None if disp is None else bool(
        evidence_in_transcript(ev_d, tn) and resolve_availability(ev_d, ref, cfg) == disp)
    lvl, ev_l = fields["nivel_ingles"], ev.get("nivel_ingles")
    if lvl is None:
        out["nivel_ingles"] = None
    else:
        words = cfg["interview"]["english_keywords"].get(lvl, [])
        out["nivel_ingles"] = bool(evidence_in_transcript(ev_l, tn) and any(
            contains_phrase(normalize_for_match(ev_l), w) for w in words))
    return {k: status(v, cfg) for k, v in out.items()}, detail


# ---------- comparación con la verdad de campo ----------

def _norm_list(xs):
    return sorted({normalize_for_match(x).replace(" ", "") for x in xs or []})


def score(fields: dict, gt: dict, ref: date | None = None) -> dict:
    """Devuelve, por campo: acierto (bool) y error absoluto cuando aplica."""
    res = {}
    a, b = fields.get("anios_experiencia"), gt["anios_experiencia"]
    res["anios_experiencia"] = {"ok": a is not None and abs(float(a) - float(b)) < 0.01,
                                "abs_err": None if a is None else abs(float(a) - float(b))}
    res["ultimo_puesto"] = {"ok": normalize_for_match(fields.get("ultimo_puesto") or "") == normalize_for_match(gt["ultimo_puesto"])}
    p, t = set(_norm_list(fields.get("tecnologias"))), set(_norm_list(gt["tecnologias"]))
    tp = len(p & t)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(t) if t else 0.0
    res["tecnologias"] = {"ok": p == t, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    s, sg = fields.get("pretension_salarial"), gt["pretension_salarial"]
    res["pretension_salarial"] = {"ok": s is not None and abs(float(s) - float(sg)) < 0.01 and fields.get("moneda") == gt["moneda"],
                                  "abs_err": None if s is None else abs(float(s) - float(sg))}
    d, dg = fields.get("disponibilidad"), str(gt["disponibilidad"])
    err_days = None
    if d and d != "inmediata" and dg != "inmediata":
        err_days = abs((date.fromisoformat(d) - date.fromisoformat(dg)).days)
    res["disponibilidad"] = {"ok": d == dg, "abs_err_days": err_days}
    res["nivel_ingles"] = {"ok": fields.get("nivel_ingles") == gt["nivel_ingles"]}
    return res
