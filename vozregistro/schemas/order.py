"""Esquema de la Tarea 2: registro de pedido (restaurante) que se verifica a sí mismo."""
from __future__ import annotations

import re

from ..textnorm import normalize_for_match, soft_normalize
from ..verify import evidence_in_transcript, has_number, status, verify_number, verify_phrase

NAME = "order"
FIELDS = ["items", "promocion", "direccion_entrega", "metodo_pago", "total_dicho"]
CATEGORICAL = {"metodo_pago"}
NUMERIC = {"total_dicho"}

_SENT = re.compile(r"(?:[^.?!¿¡\n]|(?<=\d)\.(?=\d))+[.?!\n]?")  # el punto decimal no corta la frase
_PRICE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])(\s*(?:soles|sol)\b)?")


def _sents(soft: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(soft) if s.strip(" .?!¿¡\n")]


def _alias_table(cfg: dict) -> list[tuple[str, str]]:
    pairs = [(normalize_for_match(a), prod) for prod, spec in cfg["order"]["menu"].items() for a in spec["aliases"]]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def _promo_table(cfg: dict) -> list[tuple[str, str]]:
    pairs = [(normalize_for_match(a), name) for name, spec in cfg["order"]["promotions"].items() for a in spec["aliases"]]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def normalize_address(text: str | None, cfg: dict) -> str:
    if not text:
        return ""
    t = soft_normalize(text)
    for ab, full in sorted(cfg["order"]["address_abbreviations"].items(), key=lambda kv: len(kv[0]), reverse=True):
        t = re.sub(rf"(?<!\w){re.escape(ab)}(?!\w)", full, t)
    t = normalize_for_match(t)
    t = re.sub(r"\b(\d+) ([a-z])\b(?! \w{2,} \d)", r"\1\2", t)        # "4 b" -> "4b"
    t = re.sub(r"^(?:a la|al|a|la|en)\s+", "", t)
    t = re.sub(r"\ben\b\s*", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------- extracción por reglas ----------

def _mask(text: str, phrases: list[str]) -> str:
    for p in phrases:
        text = re.sub(rf"(?<!\w){re.escape(p)}(?!\w)", lambda m: "#" * len(m.group(0)), text)
    return text


def _items(soft: str, cfg: dict):
    menu = cfg["order"]["menu"]
    size_alias = {normalize_for_match(k): v for k, v in cfg["order"]["size_aliases"].items()}
    size_keys = sorted(size_alias, key=len, reverse=True)
    removal = [normalize_for_match(c) for c in cfg["order"]["removal_cues"]]
    aliases = _alias_table(cfg)
    promos = [p for p, _ in _promo_table(cfg)]
    state: dict[str, dict] = {}   # producto -> item vigente (la última mención manda)
    order: list[str] = []
    for sent in _sents(soft):
        s = normalize_for_match(sent)
        s_masked = _mask(s, promos)  # "martes de pollo" o "gaseosa de regalo" no son ítems
        taken = [False] * len(s_masked)
        mentions = []
        for alias, prod in aliases:
            for m in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", s_masked):
                if any(taken[m.start():m.end()]):
                    continue
                for k in range(m.start(), m.end()):
                    taken[k] = True
                mentions.append((m.start(), m.end(), prod))
        mentions.sort()
        for idx, (st, en, prod) in enumerate(mentions):
            nxt = mentions[idx + 1][0] if idx + 1 < len(mentions) else len(s)
            before = s[max(0, st - 40):st]
            after = s[en:nxt]
            valid_sizes = {normalize_for_match(x): x for x in menu[prod]["sizes"]}
            size = None
            # el alias mismo puede traer el tamaño ("jarra de chicha")
            for zone in (s[st:en] + after[:30], before[-25:]):
                for k in size_keys:
                    if re.search(rf"(?<![\w.]){re.escape(k)}(?!\w)", zone) and normalize_for_match(size_alias[k]) in valid_sizes:
                        size = valid_sizes[normalize_for_match(size_alias[k])]
                        break
                if size:
                    break
            if size is None and len(valid_sizes) == 1:
                size = next(iter(valid_sizes.values()))
            # cantidad = el número más cercano antes del producto, con a lo sumo 3 palabras en medio
            qm = re.search(r"(?<![\w.])(\d+)\s+(?:[a-z]+\s+){0,3}$", before)
            qty = int(qm.group(1)) if qm and int(qm.group(1)) < 50 else None
            # los tamaños con cifras ("1.5 litros", "de 500") no son precios
            after_p = _mask(after, [k for k in size_keys if re.search(r"\d", k)])
            pm = None
            for cand in _PRICE.finditer(after_p):
                val = float(cand.group(1))
                tail = after_p[cand.end():cand.end() + 4]
                if "%" in tail or (val == int(val) and val < 50 and not cand.group(2) and "." not in cand.group(1)
                                   and not re.search(r"\b(a|esta|cuesta|sale|de)\s*$", after_p[:cand.start()])):
                    continue
                pm = val
                break
            removed = any(re.search(rf"\b{re.escape(c)}\s+(?:\w+\s+){{0,2}}$", before) for c in removal)
            if removed:
                state.pop(prod, None)
                continue
            if prod not in state and qty is None and pm is None:
                continue  # mención de contexto ("viene con papas"), no un pedido
            item = state.get(prod, {"producto": prod, "tamano": None, "cantidad": 1, "precio_unitario": None,
                                    "evidencia": []})
            if qty is not None:
                item["cantidad"] = qty
            if size is not None:
                item["tamano"] = size
            if pm is not None:
                item["precio_unitario"] = pm
            if s not in item["evidencia"]:
                item["evidencia"].append(s)
            if prod not in state:
                order.append(prod)
            state[prod] = item
    items = [state[p] for p in order if p in state]
    evidence = [it.pop("evidencia") for it in items]
    return items, evidence


def _promo(soft: str, cfg: dict):
    s = normalize_for_match(soft)
    best = None
    for alias, name in _promo_table(cfg):
        for m in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", s):
            if best is None or m.start() > best[0]:
                best = (m.start(), name, alias)
    return (best[1], best[2]) if best else (None, None)


def _address(soft: str, cfg: dict):
    sents = _sents(soft)
    cues = [normalize_for_match(c) for c in cfg["order"]["address_cues"]]
    for i, sent in enumerate(sents):
        s = normalize_for_match(sent)
        if "direccion" in s and sent.rstrip().endswith("?") and i + 1 < len(sents):
            cand = sents[i + 1]
            if re.search(r"\d", cand):
                return normalize_address(cand, cfg), normalize_for_match(cand)
        for c in sorted(cues, key=len, reverse=True):
            m = re.search(rf"\b{re.escape(c)}\b\s+(.*)", s)
            if m and re.search(r"\d", m.group(1)):
                raw = m.group(1) if not c.startswith("a la ") and not c.startswith("al ") else c.split(" ", 1)[1] + " " + m.group(1)
                return normalize_address(raw, cfg), normalize_for_match(raw)
    return None, None


def _payment(soft: str, cfg: dict):
    methods = cfg["order"]["payment_methods"]
    sents = _sents(soft)
    for i, sent in enumerate(sents):
        if re.search(r"pag\w*|forma de pago|metodo de pago", normalize_for_match(sent)):
            # la ÚLTIMA mención tras la pregunta manda ("en efectivo... no, mejor por yape")
            last = None
            for j in range(i, min(i + 4, len(sents))):
                for m in re.finditer(rf"\b({'|'.join(methods)})\w*\b", normalize_for_match(sents[j])):
                    last = (m.group(1), m.group(0))
            if last:
                return last
    for mth in methods:
        m = re.search(rf"\b{mth}\w*\b", normalize_for_match(soft))
        if m:
            return mth, m.group(0)
    return None, None


def _total(soft: str, cfg: dict):
    s = normalize_for_match(soft)
    best = None
    for cue in cfg["order"]["total_cues"]:
        c = normalize_for_match(cue)
        for m in re.finditer(rf"\b{re.escape(c)}\b", s):
            tail = s[m.end():m.end() + 70]
            pm = _PRICE.search(tail)
            if pm and (best is None or m.start() > best[0]):
                best = (m.start(), float(pm.group(1)), (c + tail[:pm.end()]).strip())
    return (best[1], best[2]) if best else (None, None)


def extract_rules(transcript: str, cfg: dict, ref=None) -> tuple[dict, dict]:
    soft = soft_normalize(transcript)
    fields, ev = {}, {}
    fields["items"], ev["items"] = _items(soft, cfg)
    fields["promocion"], ev["promocion"] = _promo(soft, cfg)
    fields["direccion_entrega"], ev["direccion_entrega"] = _address(soft, cfg)
    fields["metodo_pago"], ev["metodo_pago"] = _payment(soft, cfg)
    fields["total_dicho"], ev["total_dicho"] = _total(soft, cfg)
    return fields, ev


# ---------- esquema JSON para la ruta LLM ----------

def llm_schema(cfg: dict) -> dict:
    products = list(cfg["order"]["menu"])
    item = {"type": "object", "additionalProperties": False,
            "required": ["producto", "tamano", "cantidad", "precio_unitario", "evidencia"],
            "properties": {"producto": {"type": "string", "enum": products},
                           "tamano": {"type": ["string", "null"]},
                           "cantidad": {"type": "integer"},
                           "precio_unitario": {"type": ["number", "null"]},
                           "evidencia": {"type": "array", "items": {"type": "string"}}}}

    def vf(schema):
        return {"type": "object", "additionalProperties": False, "required": ["valor", "evidencia"],
                "properties": {"valor": schema, "evidencia": {"type": ["string", "null"]}}}

    return {"type": "object", "additionalProperties": False, "required": FIELDS,
            "properties": {"items": {"type": "array", "items": item},
                           "promocion": vf({"type": ["string", "null"], "enum": list(cfg["order"]["promotions"]) + [None]}),
                           "direccion_entrega": vf({"type": ["string", "null"]}),
                           "metodo_pago": vf({"type": ["string", "null"], "enum": cfg["order"]["payment_methods"] + [None]}),
                           "total_dicho": vf({"type": ["number", "null"]})}}


def from_llm(data: dict, cfg: dict, ref=None) -> tuple[dict, dict]:
    items, ev_items = [], []
    for it in data["items"]:
        ev_items.append(it.pop("evidencia", []))
        items.append(it)
    fields = {"items": items}
    ev = {"items": ev_items}
    for f in FIELDS[1:]:
        fields[f] = data[f]["valor"]
        ev[f] = data[f]["evidencia"]
    fields["direccion_entrega"] = normalize_address(fields["direccion_entrega"], cfg) or None
    return fields, ev


# ---------- verificación ----------

def verify(fields: dict, ev: dict, transcript: str, cfg: dict, ref=None) -> tuple[dict, dict]:
    tn = normalize_for_match(transcript)
    aliases = _alias_table(cfg)
    per_item = []
    for it, e in zip(fields.get("items") or [], ev.get("items") or []):
        evn = normalize_for_match(" ".join(e or []))
        ok = bool(e) and evidence_in_transcript(e, tn)
        ok = ok and any(a in evn for a, p in aliases if p == it["producto"])
        ok = ok and (it["cantidad"] == 1 or has_number(evn, it["cantidad"]))
        ok = ok and it.get("precio_unitario") is not None and has_number(evn, it["precio_unitario"])
        per_item.append(ok)
    out = {"items": (all(per_item) if per_item else None)}
    promo = fields.get("promocion")
    out["promocion"] = None if promo is None else any(
        verify_phrase(a, tn) for a in cfg["order"]["promotions"].get(promo, {}).get("aliases", []))
    addr = fields.get("direccion_entrega")
    out["direccion_entrega"] = None if not addr else normalize_address(addr, cfg) in normalize_address(transcript, cfg)
    pay = fields.get("metodo_pago")
    out["metodo_pago"] = None if pay is None else bool(re.search(rf"\b{pay}\w*\b", tn))
    out["total_dicho"] = verify_number(fields.get("total_dicho"), ev.get("total_dicho"), tn)
    return {k: status(v, cfg) for k, v in out.items()}, {"items": per_item}


# ---------- comparación con la verdad de campo ----------

def _item_key(it: dict) -> tuple:
    return (it["producto"], normalize_for_match(str(it.get("tamano") or "")), int(it["cantidad"]),
            round(float(it["precio_unitario"] or 0), 2))


def score(fields: dict, gt: dict, ref=None, cfg: dict | None = None) -> dict:
    pred = {_item_key(i) for i in fields.get("items") or []}
    true = {_item_key(i) for i in gt["items"]}
    tp = len(pred & true)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(true) if true else 0.0
    res = {"items": {"ok": pred == true, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}}
    res["promocion"] = {"ok": (fields.get("promocion") or None) == (gt["promocion"] or None)}
    pa = normalize_for_match(fields.get("direccion_entrega") or "")
    ga = normalize_for_match(gt["direccion_entrega"])
    res["direccion_entrega"] = {"ok": pa == ga,
                                "digits_ok": re.findall(r"\d+\w?", pa) == re.findall(r"\d+\w?", ga)}
    res["metodo_pago"] = {"ok": fields.get("metodo_pago") == gt["metodo_pago"]}
    t, tg = fields.get("total_dicho"), gt["total_dicho"]
    res["total_dicho"] = {"ok": t is not None and abs(float(t) - float(tg)) < 0.01,
                          "abs_err": None if t is None else abs(float(t) - float(tg))}
    return res
